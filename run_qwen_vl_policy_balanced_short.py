import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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

There are some examples with the source image, source prompt, target prompt, and edit instruction.

Return only valid JSON:
{"bucket": "LOW|MID|HIGH", "reason": "one short sentence"}
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

def resolve_source_image_path(row, data_root=DATA_ROOT):
    """Find the PIE-Bench source image for a row.

    In this data copy, source images are stored under:
        DATA_ROOT / "annotation_images" / <category> / <source_type> / <image_type> / <file_id>.jpg
    """
    raw_file_id = row["file_id"]

    try: 
        file_id = f"{int(float(raw_file_id)):012d}"
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid file_id={raw_file_id!r}; expected a numeric PIE-Bench file_id"
        ) from e

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

def choose_prompt_examples(df, current_index=None, max_examples=3):
    """Look up the fixed hardcoded prompt examples in df."""
    examples = []

    current_file_id = None
    if current_index is not None and current_index in df.index:
        current_file_id = _normalize_file_id_for_match(df.loc[current_index, "file_id"])

    for spec in PROMPT_EXAMPLES:
        if len(examples) >= max_examples:
            break

        wanted = _normalize_file_id_for_match(spec["file_id"])

        # Do not use the current query image as its own example.
        if wanted == current_file_id:
            continue

        matches = df[df["file_id"].map(_normalize_file_id_for_match) == wanted]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one row for prompt example file_id={spec['file_id']}, "
                f"found {len(matches)}."
            )

        row = matches.iloc[0]
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
        })

    return examples


def build_example_user_content(row, image_path, examples):
    """Build a multimodal Qwen-VL user message."""
    content = []

    def input_text(r):
        return (
            'Input: { '
            f'"source prompt": "{r.get("original_prompt", "")}", '
            f'"target prompt": "{r.get("editing_prompt", "")}", '
            f'"edit instruction": "{r.get("editing_instruction", "")}" '
            '}\n'
        )

    for k, spec in enumerate(examples, 1):
        ex = spec["row"]
        content.extend([
            {"type": "text", "text": f"Example {k}\nInput source image:"},
            {"type": "image", "url": str(resolve_source_image_path(ex))},
            {
                "type": "text",
                "text": (
                    input_text(ex)
                    + f'Output: {{"bucket": "{spec["bucket"]}", '
                    + f'"reason": "{spec["reason"]}"}}\n'
                ),
            },
        ])

    prompt = (
        "Now predict the best ChordEdit timestep bucket for this new input."
        if examples
        else "Predict the best ChordEdit timestep bucket for this input."
    )

    content.extend([
        {"type": "text", "text": f"{prompt}\nInput source image:"},
        {"type": "image", "url": str(image_path)},
        {"type": "text", "text": input_text(row) + "Return only valid JSON."},
    ])

    return content

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

    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    outputs = []

    for i, row in df.iterrows():
        image_path = resolve_source_image_path(row)
        examples = []
        if args.num_prompt_examples > 0:
            examples = choose_prompt_examples(full_df, i, max_examples=args.num_prompt_examples)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": build_example_user_content(row, image_path, examples),
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
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
            "oracle_bucket": row["timestep_bucket"],
            "oracle_t_start": row["t_start"],
            "qwen_pred_bucket": pred_bucket,
            "qwen_reason": reason,
            "raw_output": text,
        })

        correct = pred_bucket == str(row["timestep_bucket"]).lower()
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
