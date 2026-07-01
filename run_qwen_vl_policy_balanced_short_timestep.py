import argparse
import csv
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


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

SYSTEM_PROMPT = """You are predicting the best ChordEdit diffusion timestep t_start for an image edit.

Choose the t_start that best balances:
1. making the requested edit visible, and
2. preserving the original image.

t_start must be exactly one of: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

Choose a low t_start when preservation likely matters more than forcing a strong edit.
Choose a mid t_start when both a weak and a strong edit seem plausible.
Choose a high t_start only when the edit would likely be too weak without strong source override.
"""

ALLOWED_T_START = [round(i * 0.1, 1) for i in range(11)]


def _snap_to_allowed_t_start(value, tol=0.05):
    """Snap a float to the nearest allowed t_start value, or return None if too far off."""
    closest = min(ALLOWED_T_START, key=lambda a: abs(a - value))
    return closest if abs(closest - value) <= tol else None


def parse_timestep(text):
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
            t_start = _snap_to_allowed_t_start(float(obj.get("t_start")))
            reason = str(obj.get("reason", "")).strip()
        except Exception:
            continue
        if t_start is not None:
            return t_start, reason

    # Conservative fallback: only accept regex parsing if exactly one unique
    # allowed t_start value appears. This avoids misparsing explanations like
    # "0.5 is too weak, 0.8 is better" as 0.5 just because it appears first.
    # Requires a decimal point so bare digits elsewhere in the text (list
    # markers, "step 1", percentages) can't be mistaken for 0.0/1.0.
    raw_matches = re.findall(r"\b\d\.\d+\b", text)
    snapped = []
    for raw in raw_matches:
        value = _snap_to_allowed_t_start(float(raw), tol=1e-6)
        if value is not None:
            snapped.append(value)
    unique = list(dict.fromkeys(snapped))
    if len(unique) == 1:
        return unique[0], text.strip()

    return "parse_error", text.strip()


DATA_ROOT = Path("/shared/ssd_30T/zarageddes/llm_timestep_policy/data")

# Fixed labeled examples shown to Qwen before each query.
# These examples are drawn from the corrected full700 oracle using whole-image PSNR,
# CLIP Edit, and per-example normalized combined scores. t_start values were checked
# against policy_dataset_full700_sdturbo_wholepsnr_clipedit_perexample.csv.
PROMPT_EXAMPLES = [
    {
        "file_id": "323000000007",
        "t_start": 0.1,
        "reason": (
            "This is a small, localized object removal. The plate, chicken, "
            "table, and overall scene should stay almost unchanged, so a low "
            "timestep of 0.1 is appropriate."
        ),
    },
    {
        "file_id": "621000000004",
        "t_start": 0.5,
        "reason": (
            "This changes the color of the main subject while preserving the "
            "kitten's shape, pose, background, and scene layout, so a moderate "
            "timestep of 0.5 is appropriate."
        ),
    },
    {
        "file_id": "613000000003",
        "t_start": 0.8,
        "reason": (
            "This changes a large background surface from white to red. Because "
            "the edit affects a broad region and requires a strong color "
            "transformation, a high timestep of 0.8 is appropriate."
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


def get_oracle_t_start(row):
    """Return the ground-truth diffusion timestep t_start, snapped to the allowed grid."""
    t_start = _snap_to_allowed_t_start(float(row["t_start"]))
    if t_start is None:
        raise ValueError(f"Invalid t_start={row['t_start']!r}")
    return t_start


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
        expected_t_start = float(spec["t_start"])
        actual_t_start = get_oracle_t_start(row)

        if abs(actual_t_start - expected_t_start) > 1e-6:
            raise ValueError(
                f"Prompt example file_id={spec['file_id']} has oracle t_start "
                f"{actual_t_start}, but PROMPT_EXAMPLES says {expected_t_start}."
            )

        examples.append({
            "row": row,
            "t_start": expected_t_start,
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

    Requires get_patched_chat_template() — the default Qwen3-VL template silently
    drops image content from the system role, causing a feature-count mismatch.
    """
    system_content = [{"type": "text", "text": SYSTEM_PROMPT}]

    if examples:
        system_content.append({"type": "text", "text": "There are some examples with the source image, source prompt, target prompt:\n"})

    for spec in examples:
        ex = spec["row"]
        system_content.append({"type": "image", "url": str(spec["image_path"])})
        system_content.append({"type": "text", "text": _input_text(ex)})
        output_json = json.dumps({"t_start": round(spec["t_start"], 1), "reason": spec["reason"]}, ensure_ascii=False)
        system_content.append({"type": "text", "text": f"Output: {output_json}\n"})

    system_content.append({"type": "text", "text": (
        '\nReturn only valid JSON:\n'
        '{"t_start": <one of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0>, '
        '"reason": "one short sentence"}\n'
    )})

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
    "editing_instruction", "oracle_bucket", "oracle_t_start",
    "qwen_pred_t_start", "qwen_reason", "raw_output",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_prompt_examples", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=140)
    parser.add_argument("--gpu_id", type=int, default=0, help="Single CUDA device index to run on.")
    args = parser.parse_args()

    full_df = pd.read_csv(args.input_csv)
    df = full_df
    if args.max_examples is not None:
        df = full_df.head(args.max_examples).copy()

    prompt_example_pool = []
    if args.num_prompt_examples > 0:
        prompt_example_pool = build_prompt_example_pool(
            full_df,
            max_examples=args.num_prompt_examples,
        )

    device_map = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_id} on {device_map}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    patched_template = get_patched_chat_template(processor)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    outputs = []
    num_failed = 0

    with open(args.output_csv, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for _, row in df.iterrows():
            try:
                image_path = resolve_source_image_path(row)
                examples = []
                if args.num_prompt_examples > 0:
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

                pred_t_start, reason = parse_timestep(text)
                oracle_t_start = get_oracle_t_start(row)

                record = {
                    "file_id": row["file_id"],
                    "editing_type_name": row.get("editing_type_name", ""),
                    "original_prompt": row.get("original_prompt", ""),
                    "editing_prompt": row.get("editing_prompt", ""),
                    "editing_instruction": row.get("editing_instruction", ""),
                    "oracle_bucket": str(row["timestep_bucket"]).strip().lower(),
                    "oracle_t_start": oracle_t_start,
                    "qwen_pred_t_start": pred_t_start,
                    "qwen_reason": reason,
                    "raw_output": text,
                }
                outputs.append(record)
                writer.writerow(record)
                out_file.flush()

                correct = isinstance(pred_t_start, float) and abs(pred_t_start - oracle_t_start) < 1e-6
                print(f"[{len(outputs)}/{len(df)}] file_id={row['file_id']} pred={pred_t_start} oracle={oracle_t_start} correct={correct}")
            except Exception as e:
                num_failed += 1
                print(f"Warning: skipping file_id={row.get('file_id')} due to error: {e}")

    out = pd.DataFrame(outputs)

    valid = out[out["qwen_pred_t_start"].apply(lambda v: isinstance(v, float))].copy()

    if len(valid):
        exact_acc = (abs(valid["qwen_pred_t_start"] - valid["oracle_t_start"]) < 1e-6).mean()
        mae = (valid["qwen_pred_t_start"] - valid["oracle_t_start"]).abs().mean()
    else:
        exact_acc = 0.0
        mae = float("nan")

    print(f"\nSaved: {args.output_csv}")
    if num_failed:
        print(f"Skipped {num_failed} row(s) due to errors (see warnings above).")
    print(f"Valid predictions: {len(valid)} / {len(out)}")
    print(f"Exact-match accuracy: {exact_acc:.4f}")
    print(f"Mean absolute error: {mae:.4f}")

    print("\nConfusion table (oracle t_start vs qwen predicted t_start):")
    print(pd.crosstab(valid["oracle_t_start"], valid["qwen_pred_t_start"], rownames=["oracle"], colnames=["qwen"]))


if __name__ == "__main__":
    main()
