import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SYSTEM_PROMPT = """You are choosing a ChordEdit timestep bucket for an image edit.

Choose the bucket by estimating how much the edit must change the original image.

Buckets:
LOW = t_start 0.3 or 0.4
MID = t_start 0.5 or 0.6
HIGH = t_start 0.7, 0.8, or 0.9

Do not choose based only on whether the instruction sounds easy or hard. Judge the edit along these three axes:

1. Region scope:

* small/local region = lower strength
* main subject or large salient region = higher strength
* whole image or global scene/style = highest strength

2. Visual identity change:

* small attribute/detail change = lower strength
* object/category/pose/background change = moderate strength
* change to the defining color, material, style, or identity of a salient object = higher strength

3. Preservation need:

* if the original layout, object shape, pose, or scene should stay nearly the same, lower the strength
* if preserving the source would prevent the requested edit from appearing, raise the strength

Choose LOW when the edit is mostly local or attribute-level and the original image should stay very similar.

Choose MID when the edit needs a clear visible change but should still preserve most source structure.

Choose HIGH when the edit must strongly change a salient object, defining color/material, global style, lighting, background, or scene identity, and lower strengths would likely leave the edit weak or absent.

Examples of edits that may be LOW:

* small object addition/removal
* slight pose, expression, or state change
* changing a local detail or pattern
* object replacement where preserving the original shape/layout is important

Examples of edits that may be MID:

* local object replacement
* noticeable pose/state/expression change
* changing an important object detail
* background change that should preserve the main subject

Examples of edits that may be HIGH:

* changing the global style of the image
* changing the defining color or material of a main object
* changing the main subject into a substantially different subject
* changing the scene, atmosphere, or lighting in a way that affects the whole image

Use the examples as guidance, not fixed rules. The same edit type can be LOW, MID, or HIGH depending on scope, visual identity change, and preservation need.

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
