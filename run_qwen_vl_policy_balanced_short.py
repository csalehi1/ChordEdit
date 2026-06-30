import argparse
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
    # Prefer JSON if model follows instructions.
    try:
        obj = json.loads(text)
        bucket = str(obj.get("bucket", "")).strip().upper()
        reason = str(obj.get("reason", "")).strip()
        if bucket in {"LOW", "MID", "HIGH"}:
            return bucket.lower(), reason
    except Exception:
        pass

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
]

def _format_file_id_12(raw_file_id):
    try:
        return f"{int(float(raw_file_id)):012d}"
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid file_id={raw_file_id!r}; expected a numeric PIE-Bench file_id"
        ) from e


@lru_cache(maxsize=None)
def resolve_source_image_path_for_file_id(raw_file_id, data_root=str(DATA_ROOT)):
    """Find and cache the PIE-Bench source image path for a file_id."""
    file_id = _format_file_id_12(raw_file_id)
    data_root = Path(data_root)
    image_root = data_root / "annotation_images"
    matches = list(image_root.rglob(f"{file_id}.jpg")) + list(image_root.rglob(f"{file_id}.png"))

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
        system_content.append({"type": "text", "text": (
            f'Output: {{"bucket": "{spec["bucket"]}", "reason": "{spec["reason"]}"}}\n'
        )})

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_prompt_examples", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=140)
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

    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    patched_template = get_patched_chat_template(processor)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    outputs = []

    for _, row in df.iterrows():
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

        pred_bucket, reason = parse_bucket(text)

        outputs.append({
            "file_id": row["file_id"],
            "editing_type_name": row.get("editing_type_name", ""),
            "original_prompt": row.get("original_prompt", ""),
            "editing_prompt": row.get("editing_prompt", ""),
            "editing_instruction": row.get("editing_instruction", ""),
            "oracle_bucket": str(row["timestep_bucket"]).strip().lower(),
            "oracle_t_start": row["t_start"],
            "qwen_pred_bucket": pred_bucket,
            "qwen_reason": reason,
            "raw_output": text,
        })

        correct = pred_bucket == str(row["timestep_bucket"]).strip().lower()
        print(f"[{len(outputs)}/{len(df)}] file_id={row['file_id']} pred={pred_bucket} oracle={row['timestep_bucket']} correct={correct}")

    out = pd.DataFrame(outputs)
    out.to_csv(args.output_csv, index=False)

    valid = out[out["qwen_pred_bucket"].isin(["low", "mid", "high"])].copy()
    acc = (valid["qwen_pred_bucket"] == valid["oracle_bucket"]).mean() if len(valid) else 0.0

    print(f"\nSaved: {args.output_csv}")
    print(f"Valid predictions: {len(valid)} / {len(out)}")
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion table:")
    print(pd.crosstab(valid["oracle_bucket"], valid["qwen_pred_bucket"], rownames=["oracle"], colnames=["qwen"]))


if __name__ == "__main__":
    main()
