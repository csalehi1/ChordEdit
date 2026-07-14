"""
Evaluate the metrics of a grid of cells for every sample in a dataset.
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import csv
import logging
import numpy as np
import re
import time
import torch
from collections import defaultdict
from pathlib import Path
from PIL import Image

from evaluation.matrics_calculator import MetricsCalculator

LOGGER = logging.getLogger("grid_eval")

IMAGE_SIZE = 512
DEVICE = "cuda"
METRICS = [
    "psnr_unedit_part",
    "lpips_unedit_part",
    "clip_similarity_target_image_edit_part",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", required=True)
    # parser.add_argument("--model-root", required=True)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def _format_mask_image(mask_image: Image.Image) -> np.ndarray:
    # Convert the mask image to grayscale and resize.
    mask_image = mask_image.convert("L")
    if mask_image.size != (IMAGE_SIZE, IMAGE_SIZE):
        mask_image = mask_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert the mask image to a binary array.
    mask_array = np.array(mask_image)
    mask_array = (mask_array > 127).astype(np.float64)

    # Force the border of the mask to be 1 to avoid annotation errors in boundaries.
    # Matched from https://github.com/cure-lab/PnPInversion/blob/07f97f448150e2ca220bebd54c8f687c5c50c67a/evaluation/evaluate.py#L20
    mask_array[0, :]  = 1
    mask_array[-1, :] = 1
    mask_array[:, 0]  = 1
    mask_array[:, -1] = 1

    # Matched from https://github.com/cure-lab/PnPInversion/blob/07f97f448150e2ca220bebd54c8f687c5c50c67a/evaluation/evaluate.py#L258
    mask_array = mask_array[:, :, np.newaxis].repeat(3, axis=2)
    return mask_array


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    args = parse_args()
    max_samples = args.max_samples
    generated_root = Path(args.generated_root).expanduser().resolve()
    inputs_path = generated_root / f"id_to_inputs_{generated_root.name.replace('_','').lower()}{f'_n{max_samples}' if max_samples else ''}.csv"
    result_path = generated_root / f"id_to_metrics_{generated_root.name.replace('_','').lower()}{f'_n{max_samples}' if max_samples else ''}.csv"

    # Collect samples from the inputs CSV.
    samples: dict[str, dict[str, str]] = {}
    with open(inputs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples[row["sample_id"]] = {k: v for k, v in row.items() if k != "sample_id"}

    # Collect cells from the generated root.
    cells: defaultdict[str, list[tuple[float, float, Path]]] = defaultdict(list)
    for sample_id in samples:
        cells_dir = generated_root / sample_id / "cells"
        if not cells_dir.is_dir():
            continue
        for path in sorted(cells_dir.iterdir()):
            match = re.search(r"t_start_(\d+p\d+)__t_end_(\d+p\d+)\.jpg$", path.name)
            if match is None:
                raise ValueError(f"Invalid cell filename: {path.name}")
            t_start, t_end = (float(x.replace("p", ".")) for x in match.groups())
            cells[sample_id].append((t_start, t_end, path.absolute()))

    LOGGER.info("Found %d samples and %d total cells", len(samples), sum(len(v) for v in cells.values()))

    metrics_calculator = MetricsCalculator(DEVICE)

    # OPT: Access the internal torchmetrics calculators and the CLIP model/processor
    # directly from MetricsCalculator, so we can call them with pre-built GPU tensors
    # instead of going through the PIL->numpy->tensor conversion on every call.
    psnr_calc = metrics_calculator.psnr_metric_calculator
    lpips_calc = metrics_calculator.lpips_metric_calculator
    clip_model = metrics_calculator.clip_metric_calculator.model
    clip_processor = metrics_calculator.clip_metric_calculator.processor

    # Write the header to the result file.
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["sample_id", "t_start", "t_end"] + METRICS)

    # Evaluate every sample.
    for sample_idx, (sample_id, cell_list) in enumerate(cells.items()):
        sample_start = time.perf_counter()
        sample_meta = samples[sample_id]
        source_image = Image.open(sample_meta["image_path"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        mask_image = Image.open(sample_meta["mask_image_path"]).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        mask_array = _format_mask_image(mask_image)

        # OPT 1: Precompute masked source tensors once per sample.
        # The source image and inverse mask are constant across all cells in a sample.
        # We bypass MetricsCalculator's PIL-accepting methods and call its internal
        # torchmetrics calculators (psnr_metric_calculator, lpips_metric_calculator)
        # directly with pre-built GPU tensors, eliminating redundant PIL->numpy->tensor
        # conversion and mask application that would otherwise repeat for every cell.
        src_np = np.array(source_image).astype(np.float32) / 255.0
        inv_mask = (1.0 - mask_array).astype(np.float32)
        has_unedit_part = inv_mask.sum() > 0
        has_edit_part = mask_array.sum() > 0

        src_psnr_tensor = torch.empty(0)
        src_lpips_tensor = torch.empty(0)
        text_features = torch.empty(0)

        if has_unedit_part:
            src_masked_np = src_np * inv_mask
            src_psnr_tensor = torch.tensor(src_masked_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            src_lpips_tensor = src_psnr_tensor * 2 - 1

        # OPT 2: Encode the target prompt through CLIP's text encoder once per sample.
        # We access the CLIPScore metric's internal model (clip_metric_calculator.model)
        # to extract text features, then compute image-text cosine similarity manually
        # for each cell, avoiding N-1 redundant text forward passes per sample.
        if has_edit_part:
            target_prompt = sample_meta["target_prompt"]
            with torch.no_grad():
                text_processed = clip_processor(
                    text=[target_prompt], return_tensors="pt", padding=True, truncation=True,
                )
                text_features = clip_model.get_text_features(
                    text_processed["input_ids"].to(DEVICE),
                )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Evaluate every cell.
        for t_start, t_end, cell_path in cell_list:
            target_image = Image.open(cell_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

            # OPT 3: Convert the target image to numpy once per cell and derive all
            # tensor variants, instead of repeating PIL->numpy->tensor 3x per metric.
            target_arr = np.array(target_image)

            if has_unedit_part:
                target_np = target_arr.astype(np.float32) / 255.0
                target_masked_np = target_np * inv_mask
                # Calculate PSNR between the target image and the source image.
                target_psnr_tensor = torch.tensor(target_masked_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                psnr_score = psnr_calc(target_psnr_tensor, src_psnr_tensor).cpu().item()
                # Calculate LPIPS between the target image and the source image.
                target_lpips_tensor = target_psnr_tensor * 2 - 1
                lpips_score = lpips_calc(target_lpips_tensor, src_lpips_tensor).cpu().item()
            else:
                # Fallback to NaN for metrics that require unedited parts.
                psnr_score = "nan"
                lpips_score = "nan"

            if has_edit_part:
                # Calculate CLIP score between the target image and the source image.
                target_clip_arr = np.uint8(target_arr * mask_array)
                image_for_clip = torch.tensor(target_clip_arr).permute(2, 0, 1)
                with torch.no_grad():
                    image_processed = clip_processor(images=[image_for_clip], return_tensors="pt")
                    image_features = clip_model.get_image_features(image_processed["pixel_values"].to(DEVICE))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    clip_score = (100.0 * (image_features * text_features).sum(axis=-1)).item()
            else:
                # Fallback to NaN for metrics that require edited parts.
                clip_score = "nan"

            row = [sample_id, f"{t_start:.1f}", f"{t_end:.1f}", psnr_score, lpips_score, clip_score]
            with open(result_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        elapsed = time.perf_counter() - sample_start
        LOGGER.info("[%d/%d] Evaluated %s (%d cells in %.2fs)", sample_idx + 1, len(cells), sample_id, len(cell_list), elapsed)

    LOGGER.info("Done. Metrics in %s", result_path.absolute())


if __name__ == "__main__":
    main()