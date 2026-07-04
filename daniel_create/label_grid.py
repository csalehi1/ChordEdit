"""Label an already-generated image grid set with PSNR + CLIP metrics.

Reads the cells written by generate_grid.py, scores every cell with two metrics
copied from /data/home/mirick/PnPInversion, and writes:

    <output-root>/result.csv                 (or result_shard<NN>.csv when sharded)
    <output-root>/<sample_id>/grid_psnr.png  whole-image PSNR overlay
    <output-root>/<sample_id>/grid_clip.png  mask-restricted CLIP-edited overlay

Metrics (inlined so everything runs in the chordedit env):
  psnr                                    whole-image PSNR, source vs. edited.
  clip_similarity_target_image_edit_part  CLIP similarity (100 * cosine) of the
                                          masked edit region to the target prompt.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np

import settings
from common import (
    cell_filename,
    ensure_dir,
    load_mask,
    load_samples,
    resolve_under,
    strip_brackets,
)
from grid_render import save_metric_grids

LOGGER = logging.getLogger("metric_grid")


class InlineMetrics:
    """PnPInversion's psnr / clip_similarity metrics, inlined for a single env.

    PSNR mirrors MetricsCalculator.calculate_psnr (torchmetrics PSNR, data_range=1.0,
    whole image). CLIP reproduces torchmetrics CLIPScore (100 * cosine on CLIP
    image/text embeds) via a direct CLIP forward.
    """

    def __init__(self, device: str, clip_model_id: str = settings.CLIP_MODEL_ID):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    def calculate_psnr_batch(self, img_gt, imgs_pred) -> list[float]:
        """Per-image PSNR (data_range=1.0) for many predictions vs one source."""
        torch = self._torch
        gt = torch.tensor(np.array(img_gt).astype(np.float32) / 255).permute(2, 0, 1).to(self.device)
        preds = torch.stack(
            [torch.tensor(np.array(im).astype(np.float32) / 255).permute(2, 0, 1) for im in imgs_pred]
        ).to(self.device)
        mse = ((preds - gt.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3))
        return (-10.0 * torch.log10(mse)).cpu().tolist()

    def calculate_clip_similarity_batch(self, imgs, txt: str, mask=None, batch_size: int = 64) -> list[float]:
        """CLIP edit-part similarity (100 * cosine) for many images vs one prompt."""
        from PIL import Image

        torch = self._torch
        masked = []
        for img in imgs:
            arr = np.array(img)
            if mask is not None:
                arr = np.uint8(arr * mask)
            masked.append(Image.fromarray(arr))

        scores: list[float] = []
        for start in range(0, len(masked), batch_size):
            chunk = masked[start : start + batch_size]
            inputs = self.clip_processor(
                text=[txt], images=chunk, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.clip_model(**inputs)
            img_emb = out.image_embeds / out.image_embeds.norm(p=2, dim=-1, keepdim=True)
            txt_emb = out.text_embeds / out.text_embeds.norm(p=2, dim=-1, keepdim=True)
            scores.extend((100 * (img_emb * txt_emb).sum(dim=-1)).cpu().tolist())
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=settings.DEFAULT_DATA_ROOT, help="Dataset root (for source images + masks).")
    parser.add_argument("--output-root", default=settings.DEFAULT_OUTPUT_ROOT, help="Where generated cells live and result.csv goes.")
    parser.add_argument("--chord-edit-mode", choices=["default", "sym"], default=settings.CHORD_EDIT_MODE)
    parser.add_argument("--clip-model", default=settings.CLIP_MODEL_ID, help="CLIP model id for the edit-part metric.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only score the first N samples (per shard).")
    parser.add_argument("--overwrite", action="store_true", help="Re-score samples already present in a result CSV.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split samples across this many GPU workers.")
    parser.add_argument("--shard", type=int, default=0, help="Which shard this process handles (0-based).")
    args = parser.parse_args()
    if not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, {args.num_shards}); got {args.shard}")
    return args


def _load_cells(cells_dir: Path, values):
    """Load generated cells in grid order. Returns (specs, images) or None if incomplete."""
    from PIL import Image

    specs = []
    images = []
    for t_end in values:
        for t_start in values:
            cell_path = cells_dir / cell_filename(t_start, t_end)
            if not cell_path.exists():
                return None
            with Image.open(cell_path) as img:
                images.append(img.convert("RGB").resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)))
            specs.append((t_start, t_end, cell_path))
    return specs, images


def _finite_scores(rows, column):
    """Extract finite scores from a list of rows for a given column."""
    scores = {}
    for row in rows:
        value = row[column]
        if isinstance(value, (int, float)) and not (isinstance(value, float) and value != value):
            scores[(round(row["t_start"], 1), round(row["t_end"], 1))] = float(value)
    return scores


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    args = parse_args()

    import torch
    from PIL import Image

    # Enable TF32 for faster GPU operations.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root).expanduser().resolve()
    mapping_path = data_root / "mapping_file.json"
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_dir(output_root)

    # Load the grid values and t_delta from settings.
    values = settings.grid_values(args.chord_edit_mode)
    t_delta = settings.T_DELTA

    # Load the samples for this shard.
    csv_path = output_root / (settings.CSV_NAME if args.num_shards == 1 else f"result_shard{args.shard:02d}.csv")
    samples = load_samples(mapping_path, args.max_samples, args.shard, args.num_shards)

    LOGGER.info("Data root: %s (%d sample(s))", data_root, len(samples))
    if args.num_shards > 1:
        LOGGER.info("Shard %d/%d -> %s", args.shard, args.num_shards, csv_path.name)
    LOGGER.info("Scoring cells under %s", output_root)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading inlined PnPInversion metrics (PSNR + CLIP) on %s ...", device)
    metrics = InlineMetrics(device, args.clip_model)

    # Resume against every result CSV in the output dir, result.csv and shard CSVs.
    done_ids = set()
    if not args.overwrite:
        for existing in sorted(output_root.glob("result*.csv")):
            with existing.open("r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("sample_id"):
                        done_ids.add(row["sample_id"])

    # Open the CSV file for writing.
    write_header = not csv_path.exists()
    csv_file = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=settings.CSV_FIELDS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    # Score the cells for each sample.
    for index, (sample_id, meta) in enumerate(samples, start=1):
        # Skip samples that have already been scored.
        if sample_id in done_ids:
            LOGGER.info("[%d/%d] %s already scored; skipping.", index, len(samples), sample_id)
            continue

        # Load the cells for this sample.
        cells_dir = output_root / sample_id / "cells"
        loaded = _load_cells(cells_dir, values)
        if loaded is None:
            LOGGER.warning("[%d/%d] %s cells missing/incomplete; run generate_grid.py first.", index, len(samples), sample_id)
            continue
        cell_specs, generated_list = loaded

        # Extract the sample metadata.
        image_path = resolve_under(data_root, meta[settings.FIELD_IMAGE_PATH])
        category = Path(meta[settings.FIELD_IMAGE_PATH]).parent.name
        source_prompt = strip_brackets(meta.get(settings.FIELD_SOURCE_PROMPT, ""))
        target_prompt = strip_brackets(meta.get(settings.FIELD_TARGET_PROMPT, ""))
        LOGGER.info("[%d/%d] %s (%s): %r", index, len(samples), sample_id, category, target_prompt)

        # Load the source image for the metrics.
        with Image.open(image_path) as img:
            src_for_metric = img.convert("RGB").resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        mask = load_mask(data_root, meta)

        # Calculate the PSNR and CLIP scores for the cells.
        psnr_list = metrics.calculate_psnr_batch(src_for_metric, generated_list)
        if mask.sum() == 0:
            clip_list = ["nan"] * len(generated_list)
        else:
            clip_list = metrics.calculate_clip_similarity_batch(generated_list, target_prompt, mask)

        rows = [
            {
                "sample_id": sample_id,
                "category": category,
                "t_start": t_start,
                "t_end": t_end,
                "t_delta": t_delta,
                "psnr": psnr,
                "clip_similarity_target_image_edit_part": clip_edit,
                "cell_path": str(cell_path),
            }
            for (t_start, t_end, cell_path), psnr, clip_edit in zip(cell_specs, psnr_list, clip_list)
        ]
        writer.writerows(rows)
        csv_file.flush()

        title = f'{sample_id} ({category})\nSource: "{source_prompt}"\nTarget: "{target_prompt}"'
        scores_by_metric = {
            "grid_psnr.png": ("Whole PSNR", "psnr", _finite_scores(rows, "psnr")),
            "grid_clip.png": ("CLIP-Edited", "clip", _finite_scores(rows, "clip_similarity_target_image_edit_part")),
        }
        save_metric_grids(cells_dir, output_root / sample_id, values, t_delta, title, scores_by_metric)
        LOGGER.info("[%d/%d] Scored %s (%d cells)", index, len(samples), sample_id, len(rows))

    # Close the CSV file.
    csv_file.close()
    LOGGER.info("Done. Results in %s", csv_path)


if __name__ == "__main__":
    main()
