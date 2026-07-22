"""
Quick sanity check before committing to a full run: loads Qwen-Image-Bench
once and runs ONE (source, edited, instruction) triple through both the
full-metrics prompt and the tournament/selection prompt, comparing
enable_thinking=True vs False. Prints raw model output and whether the
existing JSON-extraction logic can parse it, plus real generated-token
counts for each setting -- useful both as a smoke test and for estimating
throughput before scaling up.

Expects to run on a host with access to /shared/ssd_30T/zarageddes/ (this
was built/tested on the "seribizon" research box).
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from evaluation.full_metrics_judge import QwenFullMetricsJudge

DATA_ROOT = Path("/shared/ssd_30T/zarageddes/ultraedit_100_v2_dataroot")
MAPPING_PATH = DATA_ROOT / "mapping_file.json"
MODEL_ID = "Qwen/Qwen-Image-Bench"
TEST_MAX_NEW_TOKENS = 4096


def cell_filename(ts, te):
    def fmt(v):
        return f"{v:.1f}".replace(".", "p")
    return f"t_start_{fmt(ts)}__t_end_{fmt(te)}.jpg"


def main():
    mapping = json.loads(MAPPING_PATH.read_text())
    sid = sorted(mapping.keys())[0]
    item = mapping[sid]
    instruction = item["editing_instruction"]
    src_image = Image.open(DATA_ROOT / item["image_path"]).convert("RGB")
    tgt_path = DATA_ROOT / sid / "cells" / cell_filename(0.5, 0.5)
    tgt_image = Image.open(tgt_path).convert("RGB")

    print(f"Testing on sample {sid}, instruction: {instruction!r}", flush=True)
    print(f"Loading {MODEL_ID}...", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    print(f"Model device map: {model.hf_device_map}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"

    def run_raw(prompt_text, images_with_captions, label, enable_thinking):
        content = [{"type": "text", "text": prompt_text}]
        for caption, image in images_with_captions:
            content.append({"type": "text", "text": caption})
            content.append({"type": "image", "image": image})
        messages = [[{"role": "user", "content": content}]]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
            enable_thinking=enable_thinking,
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=TEST_MAX_NEW_TOKENS, repetition_penalty=1.05, do_sample=False,
            )
        n_generated = generated_ids.shape[1] - inputs.input_ids.shape[1]
        trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
        text = processor.batch_decode([trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(f"\n{'=' * 20} {label} (enable_thinking={enable_thinking}) {'=' * 20}")
        print(f"Generated {n_generated} tokens (budget was {TEST_MAX_NEW_TOKENS})")
        print(f"--- RAW OUTPUT ---\n{text}\n--- END RAW OUTPUT ---")

        parsed = QwenFullMetricsJudge._extract_json_block(text)
        print(f"JSON-parseable with existing extraction logic: {parsed is not None}")
        if parsed:
            print(f"Parsed keys: {list(parsed.keys())}")
        return n_generated, parsed

    fm_template = (REPO_ROOT / "evaluation" / "prompts" / "full_metrics_online.txt").read_text(encoding="utf-8")
    fm_prompt = fm_template.replace("[text instruction]", instruction).replace("[input image]", "").replace("[edited image]", "")
    sel_template = (REPO_ROOT / "evaluation" / "prompts" / "selection_judge.txt").read_text(encoding="utf-8")
    sel_prompt = sel_template.replace("[text instruction]", instruction).replace("[n_candidates]", "2")

    for thinking in (True, False):
        run_raw(fm_prompt, [("Input Image:", src_image), ("Edited Image:", tgt_image)], "FULL-METRICS JUDGE", thinking)
        run_raw(sel_prompt, [("Source Image:", src_image), ("Candidate 1:", tgt_image), ("Candidate 2:", tgt_image)], "SELECTION/TOURNAMENT JUDGE", thinking)


if __name__ == "__main__":
    main()
