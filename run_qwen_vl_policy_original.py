import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SYSTEM_PROMPT = """You are choosing a ChordEdit timestep bucket for image editing.

Your goal is to imitate an empirical oracle that selected the timestep by balancing:
1. edit success, and
2. preservation of unedited image regions.

Buckets:
LOW = t_start 0.3 or 0.4
MID = t_start 0.5 or 0.6
HIGH = t_start 0.7, 0.8, or 0.9

Important calibration:
The oracle often chooses LOW when preservation dominates. LOW does not mean the prompt change is trivial. LOW means the edit can plausibly be attempted while keeping most of the source image intact.

Do not classify by semantic distance alone.
Do not classify by human-perceived edit size alone.
Do not assume local reconstruction automatically means MID.
Do not assume material, background, pose, or object-identity words automatically mean HIGH.

Use this decision process:

1. Is the edited region probably bounded/local?
If yes, LOW is a strong candidate even if the concept changes.

2. Is the edit mostly preserving composition, pose, layout, and background?
If yes, prefer LOW unless the target clearly needs stronger generation.

3. Would HIGH likely over-edit the source image?
If yes, avoid HIGH.

4. Would LOW likely under-edit because the target requires visible structural synthesis?
If yes, consider MID.

5. Is the main subject, large region, global style, or image distribution changing?
If yes, consider HIGH.

Bucket rules:

Choose LOW when:
- the edit is bounded or local,
- the original composition should stay almost identical,
- preservation is more important than strong semantic force,
- or the change affects a limited object/detail/background region.

LOW can be correct for:
- local material changes such as plastic, crystal, knitted, golden, sculpture, or toy-like changes when the region is limited,
- local background changes such as mud to grass, branch to cave, space to desert, or sky condition changes when the main subject remains unchanged,
- object substitutions that preserve size, pose, layout, or cartoon/illustration style,
- pose/orientation/state edits that are bounded, such as folded wings, opened lid, facing backward, or upside-down painting,
- deletions of small or separate objects such as planes, eggs, sunglasses, or one object in a group.

Choose MID when:
- LOW may under-edit but HIGH may over-edit,
- the edit needs visible local synthesis,
- and the scene composition should remain mostly unchanged.

MID is common for:
- facial expression or body-part changes,
- local inpainting after deletion,
- additions/deletions that noticeably affect local composition,
- moderate pose/state changes,
- local structure changes that are more than a tiny detail but not global.

Choose HIGH when:
- the main subject identity changes in a way that likely needs strong generation,
- the color/material/style change affects a large salient object or much of the image,
- the whole artistic style or scene distribution changes,
- the target requires a major shape/geometry change,
- or LOW/MID are likely too weak to make the edit visible.

Special caution:
Color changes on salient objects can be HIGH, but small/local color changes can be LOW or MID.
Material changes on salient objects can be HIGH, but bounded/local material changes can be LOW.
Background changes can be LOW if the subject and composition are preserved.
Object substitutions can be LOW if visually compatible and layout-preserving.

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
