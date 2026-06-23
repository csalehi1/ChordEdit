import argparse
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

Do not choose HIGH just because the instruction sounds semantically meaningful.
Many meaningful edits still work best at LOW if preserving the source layout, shape, pose, and composition is important.

Do not choose LOW just because the edited region looks small.
A small-looking edit may need MID or HIGH if it changes a defining color, material, texture, style, or identity of a salient object.

Choose LOW when preservation likely matters more than forcing a strong edit.
Choose MID when both LOW and HIGH seem plausible.
Choose HIGH only when the edit would likely be weak or absent without strong source override.

Use the source image, original prompt, edit prompt, and edit instruction.

Return only valid JSON:
{"bucket": "LOW|MID|HIGH", "reason": "one short sentence"}
"""


def build_user_prompt(row):
    return f"""Edit type: {row.get("editing_type_name", "")}

Original prompt:
{row.get("original_prompt", "")}

Target/edit prompt:
{row.get("editing_prompt", "")}

Editing instruction:
{row.get("editing_instruction", "")}

Choose the timestep bucket."""


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

    # Fallback: regex search.
    m = re.search(r"\b(LOW|MID|HIGH)\b", text.upper())
    if m:
        return m.group(1).lower(), text.strip()

    return "parse_error", text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="/shared/ssd_30T/zarageddes/llm_timestep_policy/policy_dataset_strat20_clean.csv")
    parser.add_argument("--output_csv", default="/shared/ssd_30T/zarageddes/llm_timestep_policy/qwen_vl_policy_predictions_strat20_clean.csv")
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
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": build_user_prompt(row)}],
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
