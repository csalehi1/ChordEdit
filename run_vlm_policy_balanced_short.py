import argparse
import csv
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

# Model families whose chat template needs get_patched_chat_template() below —
# their stock templates only render `text` content in the system message and
# silently drop `image` content there, causing a feature-count mismatch.
# Qwen2.5-VL, LLaVA-OneVision, and InternVL templates loop over system-message
# content role-agnostically and don't need this patch.
QWEN3_VL_MODEL_TYPES = {"qwen3_vl", "qwen3_vl_moe"}


def get_patched_chat_template(processor):
    """Patch the Qwen3-VL chat template to render image tokens in the system message."""
    template = processor.chat_template

    # `\\n` is a literal backslash-n matching the Jinja source, not a newline.
    old = (
        "            {%- for content in messages[0].content %}\n"
        "                {%- if 'text' in content %}\n"
        "                    {{- content.text }}\n"
        "                {%- endif %}\n"
        "            {%- endfor %}\n"
        "        {%- endif %}\n"
        "        {{- '<|im_end|>\\n' }}"
    )
    new = (
        "            {%- for content in messages[0].content %}\n"
        "                {%- if content.type == 'image' or 'image' in content or 'image_url' in content -%}\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "                {%- elif 'text' in content %}\n"
        "                    {{- content.text }}\n"
        "                {%- endif %}\n"
        "            {%- endfor %}\n"
        "        {%- endif %}\n"
        "        {{- '<|im_end|>\\n' }}"
    )

    if old not in template:
        raise ValueError(
            "Could not find the expected system content loop in the Qwen3-VL chat "
            "template. The template may have changed — re-check chat_template.json."
        )

    return template.replace(old, new, 1)

SYSTEM_PROMPT = """You are predicting the best ChordEdit timestep bucket for an image edit.

Choose the bucket that best balances:
1. making the requested edit visible, and
2. preserving the original image.

Buckets:
LOW = t_start 0.0, 0.1, 0.2, 0.3, or 0.4: weak edit, strongest preservation.
MID = t_start 0.5 or 0.6: moderate edit.
HIGH = t_start 0.7, 0.8, 0.9, or 1.0: strong edit, more source override.

Choose LOW when preservation likely matters more than forcing a strong edit.
Choose MID when both LOW and HIGH seem plausible.
Choose HIGH only when the edit would likely be weak or absent without strong source override.
"""

def parse_bucket(text):
    # Prefer JSON if model follows instructions. Try the raw text first, then
    # fall back to the first {...} block in case the model wrapped it in a
    # markdown code fence or added surrounding prose.
    json_candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_candidates.append(match.group(0))

    for candidate in json_candidates:
        try:
            obj = json.loads(candidate)
            bucket = str(obj.get("bucket", "")).strip().upper()
            reason = str(obj.get("reason", "")).strip()
        except Exception:
            continue
        if bucket in {"LOW", "MID", "HIGH"}:
            return bucket.lower(), reason

    # Conservative fallback: only accept regex parsing if exactly one unique
    # bucket label appears. This avoids misparsing explanations like
    # "LOW is too weak, MID is better" as LOW just because LOW appears first.
    matches = re.findall(r"\b(LOW|MID|HIGH)\b", text.upper())
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0].lower(), text.strip()

    return "parse_error", text.strip()


DATA_ROOT = Path("/shared/ssd_30T/zarageddes/llm_timestep_policy/data")

# Fixed labeled examples shown to Qwen before each query.
# These examples are drawn from the corrected full700 oracle using whole-image PSNR,
# CLIP Edit, and per-example normalized combined scores.
PROMPT_EXAMPLES = [
    {
        "file_id": "323000000007",
        "bucket": "LOW",
        "reason": (
            "This is a small, localized object removal. The plate, chicken, "
            "table, and overall scene should stay almost unchanged, so a low "
            "timestep is appropriate."
        ),
    },
    {
        "file_id": "621000000004",
        "bucket": "MID",
        "reason": (
            "This changes the color of the main subject while preserving the "
            "kitten's shape, pose, background, and scene layout, so a moderate "
            "timestep is appropriate."
        ),
    },
    {
        "file_id": "613000000003",
        "bucket": "HIGH",
        "reason": (
            "This changes a large background surface from white to red. Because "
            "the edit affects a broad region and requires a strong color "
            "transformation, a high timestep is appropriate."
        ),
    },
    # Extended pool for --num_prompt_examples > 3 (up to 10 total, roughly balanced
    # across buckets: 4 LOW, 3 MID, 3 HIGH).
    {
        "file_id": "311000000001",
        "bucket": "LOW",
        "reason": (
            "This is a small, localized object removal — the headphones are "
            "removed from the cat icon. The cat's pose, shape, and background "
            "should stay almost unchanged, so a low timestep is appropriate."
        ),
    },
    {
        "file_id": "211000000001",
        "bucket": "LOW",
        "reason": (
            "This is a small, localized addition — a gold chain and a star are "
            "added to the cat's head. The cat's pose, fur, and background should "
            "stay almost unchanged, so a low timestep is appropriate."
        ),
    },
    {
        "file_id": "612000000003",
        "bucket": "LOW",
        "reason": (
            "This only recolors the umbrella from pink to yellow. The woman, her "
            "pose, the rain, and the background should stay almost unchanged, so "
            "a low timestep is appropriate."
        ),
    },
    {
        "file_id": "511000000001",
        "bucket": "MID",
        "reason": (
            "This changes the robot dog's pose from standing to sitting while "
            "preserving its design, colors, and background, so a moderate "
            "timestep is appropriate."
        ),
    },
    {
        "file_id": "112000000006",
        "bucket": "MID",
        "reason": (
            "This replaces the laptop with a notebook while preserving the "
            "man's pose, expression, desk, and background, so a moderate "
            "timestep is appropriate."
        ),
    },
    {
        "file_id": "912000000005",
        "bucket": "HIGH",
        "reason": (
            "This transforms the entire image into an oil painting style, "
            "changing the texture and rendering across the whole scene, so a "
            "high timestep is appropriate."
        ),
    },
    {
        "file_id": "811000000003",
        "bucket": "HIGH",
        "reason": (
            "This changes the entire background behind the cat, affecting a "
            "large portion of the image, so a high timestep is appropriate."
        ),
    },
]

def _format_file_id_12(raw_file_id):
    try:
        return f"{int(float(raw_file_id)):012d}"
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid file_id={raw_file_id!r}; expected a numeric PIE-Bench file_id"
        ) from e


@lru_cache(maxsize=None)
def _build_image_index(data_root):
    """Walk annotation_images/ once and index every image by its 12-digit file_id."""
    image_root = Path(data_root) / "annotation_images"
    index = {}
    for path in list(image_root.rglob("*.jpg")) + list(image_root.rglob("*.png")):
        index.setdefault(path.stem, []).append(path)
    return index


def resolve_source_image_path_for_file_id(raw_file_id, data_root=str(DATA_ROOT)):
    """Find the PIE-Bench source image path for a file_id, via a cached index."""
    file_id = _format_file_id_12(raw_file_id)
    image_root = Path(data_root) / "annotation_images"
    matches = _build_image_index(data_root).get(file_id, [])

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise FileExistsError(
            f"Found multiple source images for file_id={file_id} under {image_root}: {matches}"
        )

    raise FileNotFoundError(
        f"Could not find source image for file_id={file_id} under {image_root}"
    )


def resolve_source_image_path(row, data_root=DATA_ROOT):
    """Find the PIE-Bench source image for a row.

    In this data copy, source images are stored under:
        DATA_ROOT / "annotation_images" / <category> / <source_type> / <image_type> / <file_id>.jpg
    """
    return resolve_source_image_path_for_file_id(row["file_id"], str(data_root))


def get_oracle_bucket(row):
    """Return the ground-truth timestep bucket as LOW/MID/HIGH."""
    bucket = str(row["timestep_bucket"]).strip().upper()
    if bucket not in {"LOW", "MID", "HIGH"}:
        raise ValueError(f"Invalid timestep_bucket={row['timestep_bucket']!r}")
    return bucket


def _normalize_file_id_for_match(value):
    try:
        return str(int(float(value)))
    except Exception:
        s = str(value).strip()
        return s.lstrip("0") or "0"


def build_prompt_example_pool(df, max_examples=3):
    """Validate and return the fixed hardcoded prompt examples."""
    file_id_to_rows = {}
    for _, row in df.iterrows():
        file_id_to_rows.setdefault(
            _normalize_file_id_for_match(row["file_id"]),
            [],
        ).append(row)

    examples = []
    for spec in PROMPT_EXAMPLES:
        if len(examples) >= max_examples:
            break

        wanted = _normalize_file_id_for_match(spec["file_id"])
        matches = file_id_to_rows.get(wanted, [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one row for prompt example file_id={spec['file_id']}, "
                f"found {len(matches)}."
            )

        row = matches[0]
        expected_bucket = spec["bucket"].strip().upper()
        actual_bucket = get_oracle_bucket(row)

        if actual_bucket != expected_bucket:
            raise ValueError(
                f"Prompt example file_id={spec['file_id']} has oracle bucket "
                f"{actual_bucket}, but PROMPT_EXAMPLES says {expected_bucket}."
            )

        examples.append({
            "row": row,
            "bucket": expected_bucket,
            "reason": spec["reason"],
            "file_id": wanted,
            "image_path": resolve_source_image_path(row),
        })

    return examples


def choose_prompt_examples(example_pool, current_file_id):
    """Return examples from the pool, excluding the current query image."""
    current_file_id = _normalize_file_id_for_match(current_file_id)
    return [
        example
        for example in example_pool
        if example["file_id"] != current_file_id
    ]


_CATEGORY_LEVEL_WORDS = {
    "LOW": "small, localized",
    "MID": "moderate",
    "HIGH": "substantial, broad",
}


def _generate_category_example_reason(row, category, bucket):
    """Auto-generate a reason from the example's own editing_instruction.

    Used by same-category example selection, where examples are picked
    dynamically per query rather than hand-curated, so a bespoke reason isn't
    available.
    """
    instruction = str(row.get("editing_instruction", "")).strip().rstrip(".")
    level_word = _CATEGORY_LEVEL_WORDS[bucket]
    return (
        f"{instruction}. This is a {level_word} edit within the '{category}' "
        f"category, so a {bucket.lower()} timestep is appropriate."
    )


def choose_category_examples(df, query_row, num_examples=3):
    """Dynamically pick in-context examples from the query's own edit category.

    Selects examples from `df` sharing the query's `editing_type_name`
    (ground-truth PIE-Bench category label), excluding the query itself,
    cycling round-robin through LOW/MID/HIGH so a 3-example request yields
    one example per bucket (matching the fixed-pool convention). Reasons are
    auto-generated from each example's own editing_instruction rather than
    hand-written, since examples aren't known ahead of time.
    """
    category = query_row["editing_type_name"]
    current_file_id = _normalize_file_id_for_match(query_row["file_id"])

    candidates = df[
        (df["editing_type_name"] == category)
        & (df["file_id"].apply(_normalize_file_id_for_match) != current_file_id)
    ]

    by_bucket = {"LOW": [], "MID": [], "HIGH": []}
    for _, row in candidates.sort_values("file_id").iterrows():
        by_bucket[get_oracle_bucket(row)].append(row)

    examples = []
    bucket_order = ["LOW", "MID", "HIGH"]
    cursors = {b: 0 for b in bucket_order}
    while len(examples) < num_examples:
        added_any = False
        for bucket in bucket_order:
            if len(examples) >= num_examples:
                break
            rows = by_bucket[bucket]
            cursor = cursors[bucket]
            if cursor >= len(rows):
                continue
            row = rows[cursor]
            cursors[bucket] += 1
            added_any = True
            examples.append({
                "row": row,
                "bucket": bucket,
                "reason": _generate_category_example_reason(row, category, bucket),
                "file_id": _normalize_file_id_for_match(row["file_id"]),
                "image_path": resolve_source_image_path(row),
            })
        if not added_any:
            break  # exhausted all buckets in this category

    return examples


def _input_text(r):
    payload = {
        "source prompt": r.get("original_prompt", ""),
        "target prompt": r.get("editing_prompt", ""),
    }
    return "Input: " + json.dumps(payload, ensure_ascii=False) + "\n"


def build_messages(row, image_path, examples):
    """Build messages with ICL examples embedded in the system message.

    Structure:
      system: [task instructions] + N × ([example image] [Input] [Output]) + [JSON format instruction]
      user:   [query image] [Input]

    For Qwen3-VL models this requires get_patched_chat_template() — the stock
    template silently drops image content from the system role, causing a
    feature-count mismatch. Other supported model families render system-role
    images natively and don't need the patch (see QWEN3_VL_MODEL_TYPES).
    """
    system_content = [{"type": "text", "text": SYSTEM_PROMPT}]

    if examples:
        system_content.append({"type": "text", "text": "There are some examples with the source image, source prompt, target prompt:\n"})

    for spec in examples:
        ex = spec["row"]
        system_content.append({"type": "image", "url": str(spec["image_path"])})
        system_content.append({"type": "text", "text": _input_text(ex)})
        output_json = json.dumps({"bucket": spec["bucket"], "reason": spec["reason"]}, ensure_ascii=False)
        system_content.append({"type": "text", "text": f"Output: {output_json}\n"})

    system_content.append({"type": "text", "text": '\nReturn only valid JSON:\n{"bucket": "LOW|MID|HIGH", "reason": "one short sentence"}\n'})

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": _input_text(row) + "Output: "},
            ],
        },
    ]
    return messages

OUTPUT_FIELDNAMES = [
    "file_id", "editing_type_name", "original_prompt", "editing_prompt",
    "editing_instruction", "oracle_bucket", "oracle_t_start", "model_id",
    "pred_bucket", "reason", "raw_output",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help=(
            "Any of: Qwen/Qwen3-VL-{4B,8B}-Instruct, Qwen/Qwen2.5-VL-7B-Instruct, "
            "llava-hf/llava-onevision-qwen2-7b-ov-hf, OpenGVLab/InternVL3-8B-hf."
        ),
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_prompt_examples", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=140)
    parser.add_argument(
        "--example_selection",
        choices=["fixed", "same_category"],
        default="fixed",
        help=(
            "'fixed': use the hand-curated PROMPT_EXAMPLES pool for every query "
            "(default). 'same_category': dynamically pick --num_prompt_examples "
            "examples per query from rows sharing the query's editing_type_name "
            "(ground-truth PIE-Bench category label), one per LOW/MID/HIGH bucket, "
            "with auto-generated reasons."
        ),
    )
    parser.add_argument(
        "--exclude_editing_types",
        default="",
        help=(
            "Comma-separated editing_type_name values to drop from evaluation, "
            "e.g. 'random'. Useful with --example_selection same_category, since "
            "'random' isn't a coherent edit type to match examples against."
        ),
    )
    args = parser.parse_args()

    full_df = pd.read_csv(args.input_csv)
    exclude_types = {t.strip() for t in args.exclude_editing_types.split(",") if t.strip()}
    # example_source_df is the pool same_category examples are drawn from — filtered
    # by exclude_editing_types (so e.g. 'random' rows can never be picked as ICL
    # examples) but NOT sliced by --max_examples, so a quick test run doesn't starve
    # the example pool.
    example_source_df = full_df
    df = full_df
    if exclude_types:
        example_source_df = full_df[~full_df["editing_type_name"].isin(exclude_types)].copy()
        df = example_source_df
        print(f"Excluding editing_type_name in {sorted(exclude_types)}: "
              f"{len(full_df) - len(df)} rows dropped, {len(df)} remain.")
    if args.max_examples is not None:
        df = df.head(args.max_examples).copy()

    prompt_example_pool = []
    if args.num_prompt_examples > 0 and args.example_selection == "fixed":
        prompt_example_pool = build_prompt_example_pool(
            full_df,
            max_examples=args.num_prompt_examples,
        )
        if exclude_types:
            excluded_in_pool = [
                ex["file_id"] for ex in prompt_example_pool
                if full_df.loc[
                    full_df["file_id"].apply(_normalize_file_id_for_match) == ex["file_id"],
                    "editing_type_name",
                ].iloc[0] in exclude_types
            ]
            if excluded_in_pool:
                print(
                    f"Note: --exclude_editing_types {sorted(exclude_types)} does NOT "
                    f"apply to the fixed example pool; these file_ids remain in it "
                    f"despite matching an excluded category: {excluded_in_pool}"
                )

    print(f"Loading model: {args.model_id}")
    model_type = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True).model_type
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    patched_template = (
        get_patched_chat_template(processor) if model_type in QWEN3_VL_MODEL_TYPES else None
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    num_written = 0
    num_failed = 0

    with open(args.output_csv, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for _, row in df.iterrows():
            try:
                image_path = resolve_source_image_path(row)
                examples = []
                if args.num_prompt_examples > 0:
                    if args.example_selection == "same_category":
                        examples = choose_category_examples(
                            example_source_df, row, num_examples=args.num_prompt_examples
                        )
                        if len(examples) < args.num_prompt_examples:
                            print(
                                f"Warning: file_id={row['file_id']} (category="
                                f"{row['editing_type_name']!r}) got only {len(examples)} / "
                                f"{args.num_prompt_examples} same-category examples."
                            )
                    else:
                        examples = choose_prompt_examples(prompt_example_pool, row["file_id"])

                messages = build_messages(row, image_path, examples)

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    chat_template=patched_template,
                )

                inputs.pop("token_type_ids", None)
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]

                text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                pred_bucket, reason = parse_bucket(text)

                writer.writerow({
                    "file_id": row["file_id"],
                    "editing_type_name": row.get("editing_type_name", ""),
                    "original_prompt": row.get("original_prompt", ""),
                    "editing_prompt": row.get("editing_prompt", ""),
                    "editing_instruction": row.get("editing_instruction", ""),
                    "oracle_bucket": str(row["timestep_bucket"]).strip().lower(),
                    "oracle_t_start": row["t_start"],
                    "model_id": args.model_id,
                    "pred_bucket": pred_bucket,
                    "reason": reason,
                    "raw_output": text,
                })
                out_file.flush()
                num_written += 1

                correct = pred_bucket == str(row["timestep_bucket"]).strip().lower()
                print(f"[{num_written}/{len(df)}] file_id={row['file_id']} pred={pred_bucket} oracle={row['timestep_bucket']} correct={correct}")
            except Exception as e:
                num_failed += 1
                print(f"Warning: skipping file_id={row.get('file_id')} due to error: {e}")

    out = pd.read_csv(args.output_csv)

    valid = out[out["pred_bucket"].isin(["low", "mid", "high"])].copy()
    acc = (valid["pred_bucket"] == valid["oracle_bucket"]).mean() if len(valid) else 0.0

    print(f"\nSaved: {args.output_csv}")
    if num_failed:
        print(f"Skipped {num_failed} row(s) due to errors (see warnings above).")
    print(f"Valid predictions: {len(valid)} / {len(out)}")
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion table:")
    print(pd.crosstab(valid["oracle_bucket"], valid["pred_bucket"], rownames=["oracle"], colnames=["pred"]))


if __name__ == "__main__":
    main()
