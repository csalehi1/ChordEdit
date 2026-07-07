"""Recompute masked grid metrics for the UltraEdit_Region_1000 ChordEdit grid.

For every sample (1000) and every (t_start, t_end) cell (11x11 = 121), computes:
  - psnr_unedit_part: PIE-Bench masked PSNR — cell and source multiplied by
    (1 - mask) before PSNR (background preservation).
  - clip_similarity_target_image_edit_part: PIE-Bench masked CLIP-Edit —
    cell multiplied by mask, CLIP ViT-L/14 similarity to the target prompt.
  - psnr_whole: unmasked PSNR, kept as a cross-check against the existing
    shard CSVs (which stored whole-image PSNR under the column name `psnr`).

Conventions match /data/home/salehi/projects/PnPInversion_clean/evaluation
(matrics_calculator.py): PSNR data_range=1.0 over the full zeroed tensor,
CLIP score = 100 * cosine(image_features, text_features). Masks are
binarized at >127 (UltraEdit masks are JPEGs; white = edit region).

Usage:
  python compute_masked_grid_metrics_ultraedit1000.py [--limit N] [--device cuda:1]
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
from transformers import CLIPModel, CLIPProcessor

DATASET_ROOT = Path("/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_1000")
CELLS_ROOT = Path("/shared/ssd_30T/mirick/generated/ultra_edit/UltraEdit_Region_1000")
INPUTS_CSV = CELLS_ROOT / "id_to_inputs_ultraeditregion1000.csv"
OUT_CSV = Path("/shared/ssd_30T/salehi/id_to_metrics_ultraeditregion1000_masked.csv")

T_VALUES = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0
CLIP_BATCH = 64


def t_tag(v: float) -> str:
    return f"{v:.1f}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="only first N samples (smoke test)")
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(8)

    device = torch.device(args.device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    with open(INPUTS_CSV) as f:
        samples = list(csv.DictReader(f))
    if args.limit:
        samples = samples[: args.limit]
    samples = [s for i, s in enumerate(samples) if i % args.nshards == args.shard]

    out_path = Path(args.out)
    done_ids = set()
    if out_path.exists():  # resume support
        with open(out_path) as f:
            done_ids = {row["sample_id"] for row in csv.DictReader(f)}
        print(f"resuming: {len(done_ids)} samples already scored")

    write_header = not out_path.exists()
    fout = open(out_path, "a", newline="")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow([
            "sample_id", "category", "t_start", "t_end", "t_delta",
            "psnr_unedit_part", "clip_similarity_target_image_edit_part",
            "psnr_whole", "cell_path",
        ])

    for si, row in enumerate(samples):
        sid = row["sample_id"]
        if sid in done_ids:
            continue
        src = np.array(
            Image.open(DATASET_ROOT / row["image_path"]).convert("RGB"), dtype=np.float32
        ) / 255.0
        mask_img = np.array(Image.open(DATASET_ROOT / row["mask_image_path"]).convert("L"))
        mask = (mask_img > 127).astype(np.float32)[..., None]  # HxWx1, 1 = edit region
        unedit = 1.0 - mask
        target_prompt = row["target_prompt"].replace("[", "").replace("]", "")

        with torch.no_grad():
            txt_in = clip_processor(text=[target_prompt], return_tensors="pt", padding=True, truncation=True)
            txt_feat = clip_model.get_text_features(
                input_ids=txt_in["input_ids"].to(device),
                attention_mask=txt_in["attention_mask"].to(device),
            )
            if hasattr(txt_feat, "pooler_output"):
                txt_feat = txt_feat.pooler_output
            txt_feat = txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)

        src_bg = torch.tensor(src * unedit).permute(2, 0, 1).unsqueeze(0).to(device)

        cells, masked_pils, rows_meta = [], [], []
        for ts in T_VALUES:
            for te in T_VALUES:
                rel = f"{sid}/cells/t_start_{t_tag(ts)}__t_end_{t_tag(te)}.jpg"
                img = np.array(Image.open(CELLS_ROOT / rel).convert("RGB"), dtype=np.float32) / 255.0
                cells.append(img)
                masked_pils.append(Image.fromarray(np.uint8(img * 255 * mask)))
                rows_meta.append((ts, te, rel))

        # masked CLIP-Edit, batched
        clip_scores = []
        with torch.no_grad():
            for b in range(0, len(masked_pils), CLIP_BATCH):
                batch = masked_pils[b : b + CLIP_BATCH]
                px = clip_processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
                feat = clip_model.get_image_features(pixel_values=px)
                if hasattr(feat, "pooler_output"):
                    feat = feat.pooler_output
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                clip_scores.extend((100.0 * (feat @ txt_feat.T).squeeze(-1)).tolist())

        # PSNRs
        for (ts, te, rel), cell, clip_s in zip(rows_meta, cells, clip_scores):
            cell_t = torch.tensor(cell).permute(2, 0, 1).unsqueeze(0).to(device)
            src_t = torch.tensor(src).permute(2, 0, 1).unsqueeze(0).to(device)
            psnr_whole = psnr_metric(cell_t, src_t).item()
            if unedit.sum() == 0:
                psnr_bg = float("nan")
            else:
                cell_bg = torch.tensor(cell * unedit).permute(2, 0, 1).unsqueeze(0).to(device)
                psnr_bg = psnr_metric(cell_bg, src_bg).item()
            writer.writerow([sid, "annotation_images", ts, te, 0.0, psnr_bg, clip_s, psnr_whole, "/" + rel])
        fout.flush()
        if si % 25 == 0:
            print(f"[{si + 1}/{len(samples)}] {sid} done", flush=True)

    fout.close()
    print("done ->", out_path)


if __name__ == "__main__":
    main()
