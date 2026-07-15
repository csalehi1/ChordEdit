import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATA = Path("/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_10000")
OUT = Path("/data/home/salehi/UltraEdit_SAM3_mask_10000")
SAM3_REPO = Path("/shared/ssd_30T/salehi/UltraEdit_SAM3_mask_smoketest/sam3")
THRESHOLDS = [0.5, 0.3, 0.15, 0.05]

SYSTEM_SOURCE = """You identify which region of a SOURCE image an edit will affect.
Given a source image and an edit instruction, name the object(s) or region(s)
in the SOURCE image that the edit modifies, replaces, removes, or occupies.
Rules:
- Each concept is a short noun phrase (1-4 words) naming something visibly
  segmentable in the SOURCE image: an object ("cat", "red car door") or a
  region ("sky", "grass", "wall"). Never a color alone, never written text.
- For "add X" edits, name the source region where X will appear (e.g.
  "add a balloon in the sky" -> ["sky"]). The added object itself is NOT in
  the source image, so never name it.
- If the edit changes text, name the object carrying the text ("sign").
- The list must NEVER be empty: if unsure, name the most plausible region
  the instruction refers to.
Respond with JSON only: {"source_concepts": [...]}"""

SYSTEM_GLOBAL = """You decide whether an image edit is GLOBAL or LOCAL.
GLOBAL edits change the whole image rather than a specific object or region:
style transfer, weather or season change (e.g. "winter wonderland", "make it
snow"), lighting or color grading, background/backdrop/setting/scene
replacement, or placing the subject in a new environment.
LOCAL edits add, remove, replace, or alter specific object(s) or region(s),
including text changes. "Add X" edits are LOCAL even when X lands in the sky
or background (a rainbow, birds, stars).
Respond with JSON only: {"global_edit": true} or {"global_edit": false}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)

    mapping = json.load(open(DATA / "mapping_file.json"))
    sids = sorted(mapping)
    if args.limit:
        sids = sids[: args.limit]
    sids = [s for i, s in enumerate(sids) if i % args.nshards == args.shard]

    import sys
    sys.path.insert(0, str(SAM3_REPO))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3 = Sam3Processor(build_sam3_image_model(
        bpe_path=str(SAM3_REPO / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")))

    qwen = qwen_proc = None

    def qwen_chat(rec, system):
        nonlocal qwen, qwen_proc
        if qwen is None:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map=args.device)
            qwen_proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        from qwen_vl_utils import process_vision_info
        caption = rec["original_prompt"].replace("[", "").replace("]", "")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image", "image": str(DATA / rec["image_path"])},
                {"type": "text", "text": f'Instruction: "{rec["editing_instruction"]}"\nSource caption: "{caption}"'},
            ]},
        ]
        text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _ = process_vision_info(messages)
        inputs = qwen_proc(text=[text], images=images, return_tensors="pt").to(qwen.device)
        out = qwen.generate(**inputs, max_new_tokens=128, do_sample=False)
        reply = qwen_proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return json.loads(reply.strip().removeprefix("```json").removesuffix("```").strip())

    def requery_qwen(rec):
        try:
            c = qwen_chat(rec, SYSTEM_SOURCE)
            assert isinstance(c.get("source_concepts"), list) and c["source_concepts"]
            return c["source_concepts"]
        except Exception:
            return []

    def is_global_edit(rec):
        try:
            return bool(qwen_chat(rec, SYSTEM_GLOBAL).get("global_edit"))
        except Exception:
            return False  # when unsure, exclude rather than keep a wrong empty mask

    def sam3_union(image, concepts, threshold):
        union = np.zeros((image.height, image.width), dtype=bool)
        info = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam3.set_image(image)
            sam3.set_confidence_threshold(threshold, state=state)
            for concept in dict.fromkeys(concepts):
                out = sam3.set_text_prompt(prompt=concept, state=state)
                masks, scores = out["masks"], out["scores"]
                info[concept] = {"n_instances": int(masks.shape[0]),
                                 "scores": [round(float(s), 4) for s in scores]}
                if masks.shape[0]:
                    union |= masks.any(dim=0).squeeze(0).cpu().numpy()
        return union, info

    def iou(a, b):
        un = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / un) if un else 0.0

    for n, sid in enumerate(sids):
        if (OUT / f"{sid}_srcmask.json").exists():
            continue
        rec = mapping[sid]
        src_path = OUT / f"{sid}_concepts_source.json"
        if src_path.exists():
            concepts = json.load(open(src_path))["source_concepts"]
        else:
            concepts = json.load(open(OUT / f"{sid}_concepts.json"))["source_concepts"]
            if not concepts:
                concepts = requery_qwen(rec)
            json.dump({"source_concepts": concepts}, open(src_path, "w"), indent=2)

        src_img = Image.open(DATA / rec["image_path"]).convert("RGB")
        mask, method, info = None, None, {}
        if concepts:
            for thr in THRESHOLDS:
                u, info = sam3_union(src_img, concepts, thr)
                if u.any():
                    mask, method = u, ("sam3" if thr == 0.5 else f"sam3_t{thr}")
                    break
        if mask is None:
            if is_global_edit(rec):
                mask, method = np.zeros((src_img.height, src_img.width), dtype=bool), "empty_global"
            else:
                json.dump({"concepts": info, "mask_frac": 0.0, "method": "excluded_empty"},
                          open(OUT / f"{sid}_srcmask.json", "w"), indent=2)
                print(f"[shard {args.shard}] {n + 1}/{len(sids)} {sid} method=excluded_empty", flush=True)
                continue

        Image.fromarray(mask.astype(np.uint8) * 255).save(OUT / f"{sid}_srcmask.png")
        ref = np.array(Image.open(DATA / rec["downloaded_mask_image_path"]).convert("L")) > 127
        if ref.shape != mask.shape:
            ref = np.array(Image.fromarray(ref).resize(mask.shape[::-1], Image.NEAREST))
        stats = {"concepts": info, "mask_frac": round(float(mask.mean()), 4),
                 "method": method, "iou_vs_downloaded": round(iou(mask, ref), 4)}
        json.dump(stats, open(OUT / f"{sid}_srcmask.json", "w"), indent=2)
        if n % 100 == 0:
            print(f"[shard {args.shard}] {n + 1}/{len(sids)} {sid} method={method}", flush=True)

    print(f"[shard {args.shard}] done", flush=True)


if __name__ == "__main__":
    main()
