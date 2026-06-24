import argparse
import os
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ---------------------------------------------------------------------
# Prompt variant: shorter balanced prompt
#
# Full-700 result:
#   Output CSV:
#   /shared/ssd_30T/zarageddes/llm_timestep_policy/qwen_vl_4b_policy_predictions_full700_balanced_short.csv
#
#   Accuracy: 33.57%
#
# Notes:
#   This shorter prompt was designed to keep the main calibration cautions:
#   do not overuse HIGH for semantically meaningful edits, and do not overuse
#   LOW for small-looking edits that change salient color/material/style/identity.
#   It was close to the long prompt but still worse, mainly because predictions
#   collapsed too much toward MID.
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """You are predicting the best ChordEdit timestep bucket for an image edit.

Choose the bucket that best balances:
1. making the requested edit visible, and
2. preserving the original image.

Buckets:
LOW = t_start 0.3 or 0.4: weak edit, strongest preservation.
MID = t_start 0.5 or 0.6: moderate edit.
HIGH = t_start 0.7, 0.8, or 0.9: strong edit, more source override.

Choose LOW when preservation likely matters more than forcing a strong edit.
Choose MID when both LOW and HIGH seem plausible.
Choose HIGH only when the edit would likely be weak or absent without strong source override.

There are some examples with the source image, source prompt, target prompt.

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

# Optional fixed few-shot sets for ablations.
# Select with: QWEN_FEWSHOT_SET=color_ladder or QWEN_FEWSHOT_SET=mixed
QWEN_FEWSHOT_POOL_CSV = os.environ.get("QWEN_FEWSHOT_POOL_CSV")
QWEN_FEWSHOT_POOL_CSV = Path(QWEN_FEWSHOT_POOL_CSV) if QWEN_FEWSHOT_POOL_CSV else None

MANUAL_FEWSHOT_SETS = {
    "color_ladder": [
        "613000000003",  # LOW: make wall red, color edit
        "614000000001",  # MID: roses red to purple, color edit
        "611000000002",  # HIGH: bear brown to black, color edit
    ],
    "mixed": [
        "613000000003",  # LOW: make wall red, color edit
        "111000000001",  # MID: cat to tiger, object edit
        "611000000002",  # HIGH: bear brown to black, color edit
    ],
}

_FEWSHOT_POOL_CACHE = None


def resolve_source_image_path(row, data_root=DATA_ROOT):
    """Find the PIE-Bench source image for a row."""
    raw_file_id = row["file_id"]
    try:
        file_id = f"{int(float(raw_file_id)):012d}"
    except Exception:
        file_id = str(raw_file_id).strip().zfill(12)

    # First try common direct locations.
    candidates = [
        data_root / "annotation_images" / f"{file_id}.jpg",
        data_root / "annotation_images" / f"{file_id}.png",
        data_root / f"{file_id}.jpg",
        data_root / f"{file_id}.png",
    ]

    # Then try any path-like columns if present.
    for col in ["image_path", "source_image", "source_path", "input_image"]:
        if col in row and str(row[col]) not in ["", "nan", "None"]:
            q = Path(str(row[col]))
            candidates.append(q if q.is_absolute() else data_root / q)

    for q in candidates:
        if q.exists():
            return q

    # PIE-Bench may be nested by edit type/category, so fall back to recursive search.
    matches = list(data_root.rglob(f"{file_id}.jpg")) + list(data_root.rglob(f"{file_id}.png"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find source image for file_id={file_id} under {data_root}")



def get_oracle_bucket(row):
    """Return oracle bucket as LOW/MID/HIGH from whichever label column exists."""
    for col in ["timestep_bucket", "oracle_bucket", "bucket", "label", "best_bucket", "target_bucket", "oracle", "oracle_label"]:
        if col in row and str(row[col]) not in ["", "nan", "None"]:
            bucket = str(row[col]).strip().upper()
            if bucket in {"LOW", "MID", "HIGH"}:
                return bucket
    raise KeyError("Could not find oracle bucket column in row")


def make_example_reason(bucket):
    if bucket == "LOW":
        return "Preservation matters more and the edit should be possible with a weaker update."
    if bucket == "MID":
        return "The edit needs a moderate update while still preserving the source."
    return "The edit likely needs a stronger update to become visible."


def _normalize_file_id_for_match(value):
    try:
        return str(int(float(value)))
    except Exception:
        s = str(value).strip()
        return s.lstrip("0") or "0"


def _load_fewshot_pool(fallback_df):
    global _FEWSHOT_POOL_CACHE
    if _FEWSHOT_POOL_CACHE is not None:
        return _FEWSHOT_POOL_CACHE

    if QWEN_FEWSHOT_POOL_CSV is not None and QWEN_FEWSHOT_POOL_CSV.exists():
        _FEWSHOT_POOL_CACHE = pd.read_csv(QWEN_FEWSHOT_POOL_CSV)
    else:
        _FEWSHOT_POOL_CACHE = fallback_df

    return _FEWSHOT_POOL_CACHE


def choose_fewshot_examples(df, current_index, max_examples=3):
    """Choose few-shot examples.

    Default behavior: automatic LOW/MID/HIGH examples.
    If QWEN_FEWSHOT_SET is set, use a fixed exemplar set from MANUAL_FEWSHOT_SETS.
    """
    current_row = df.loc[current_index] if current_index in df.index else None
    current_file_id = None
    if current_row is not None and "file_id" in current_row:
        current_file_id = _normalize_file_id_for_match(current_row["file_id"])

    mode = os.environ.get("QWEN_FEWSHOT_SET", "").strip()
    pool_df = _load_fewshot_pool(df) if mode in MANUAL_FEWSHOT_SETS else df

    examples = []
    used_file_ids = {current_file_id}

    # Manual ablation mode.
    if mode in MANUAL_FEWSHOT_SETS:
        for wanted in MANUAL_FEWSHOT_SETS[mode]:
            wanted_norm = _normalize_file_id_for_match(wanted)
            if wanted_norm in used_file_ids:
                continue

            matches = pool_df[pool_df["file_id"].map(_normalize_file_id_for_match) == wanted_norm]
            if len(matches) == 0:
                raise ValueError(
                    f"Manual few-shot example file_id={wanted} was not found in the few-shot pool. "
                    "Set QWEN_FEWSHOT_POOL_CSV explicitly or choose examples present in the input CSV."
                )

            ex = matches.iloc[0]
            examples.append(ex)
            used_file_ids.add(wanted_norm)

            if len(examples) >= max_examples:
                return examples[:max_examples]

    # Fallback/default: one LOW, one MID, one HIGH.
    for bucket in ["LOW", "MID", "HIGH"]:
        if len(examples) >= max_examples:
            break

        for _, ex in pool_df.iterrows():
            fid = _normalize_file_id_for_match(ex["file_id"])
            if fid in used_file_ids:
                continue
            try:
                ex_bucket = get_oracle_bucket(ex)
            except Exception:
                continue
            if ex_bucket == bucket:
                examples.append(ex)
                used_file_ids.add(fid)
                break

    return examples[:max_examples]

def build_fewshot_user_content(row, image_path, examples):
    """Build a multimodal Qwen-VL message with real source-image examples."""
    content = []

    for k, ex in enumerate(examples, 1):
        ex_image_path = resolve_source_image_path(ex)
        ex_bucket = get_oracle_bucket(ex)
        ex_reason = make_example_reason(ex_bucket)

        content.append({
            "type": "text",
            "text": f"Example {k}\nInput source image:"
        })
        content.append({"type": "image", "url": str(ex_image_path)})
        content.append({
            "type": "text",
            "text": (
                'Input: { '
                f'"source prompt": "{ex.get("original_prompt", "")}", '
                f'"target prompt": "{ex.get("editing_prompt", "")}" '
                '}\n'
                f'Output: {{"bucket": "{ex_bucket}", "reason": "{ex_reason}"}}\n'
            )
        })

    content.append({
        "type": "text",
        "text": "Now predict the best ChordEdit timestep bucket for this new input.\nInput source image:"
    })
    content.append({"type": "image", "url": str(image_path)})
    content.append({
        "type": "text",
        "text": (
            'Input: { '
            f'"source prompt": "{row.get("original_prompt", "")}", '
            f'"target prompt": "{row.get("editing_prompt", "")}", '
            f'"edit instruction": "{row.get("editing_instruction", "")}" '
            '}\n'
            "Return only valid JSON."
        )
    })

    return content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=140)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.max_examples is not None:
        df = df.head(args.max_examples).copy()

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
        examples = choose_fewshot_examples(df, i, max_examples=3)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": build_fewshot_user_content(row, image_path, examples),
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
