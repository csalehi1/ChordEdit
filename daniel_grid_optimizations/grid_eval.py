"""
Evaluate the metrics of a grid of cells for every sample in a dataset.

Expects:

  <generated-root>/
    id_to_inputs_<suffix>.csv
    {sample_id}/cells/t_start_*__t_end_*.jpg

Writes id_to_metrics_<suffix>.csv with columns:
  sample_id,t_start,t_end,t_delta,<metrics...>,cell_path
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
import settings
from _helpers import cell_filename, metrics_fieldnames

LOGGER = logging.getLogger("grid_eval")

IMAGE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_BATCH_SIZE = 32
LPIPS_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    # parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-psnr", action="store_true", default=False)
    parser.add_argument("--include-lpips", action="store_true", default=False)
    parser.add_argument("--include-clip", action="store_true", default=False)
    return parser.parse_args()


def _format_mask_image(mask_image: Image.Image) -> np.ndarray:
    # Convert the mask image to grayscale and resize.
    mask_image = mask_image.convert("L")
    if mask_image.size != (IMAGE_SIZE, IMAGE_SIZE):
        mask_image = mask_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Convert the mask image to a binary array.
    mask_array = np.array(mask_image)
    # OPT 1: Use float32 instead of float64 to halve mask memory and avoid
    # extra .astype(np.float32) casts downstream when building GPU tensors.
    mask_array = (mask_array > 127).astype(np.float32)

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
    # Check that the generated root is a directory.
    if not generated_root.is_dir():
        raise FileNotFoundError(f"generated-root is not a directory: {generated_root}")
    # Check that the inputs CSV exists.
    inputs_path = generated_root / f"id_to_inputs_{generated_root.name.lower().replace('_', '').replace('-', '')}.csv"
    if not inputs_path.is_file():
        raise FileNotFoundError(f"Missing inputs CSV (expected generated layout): {inputs_path}")
    # Resolve the result path for the generated root.
    suffix = generated_root.name.lower().replace("_", "").replace("-", "")
    result_path = (
        Path(args.result_path).expanduser().resolve()
        if args.result_path
        else generated_root / f"id_to_metrics_{suffix}{f'_n{max_samples}' if max_samples is not None else ''}.csv"
    )

    # Build the list of metrics to evaluate. If no --include-* flags are
    # specified, all metrics are included (the default behavior).
    include_psnr = args.include_psnr
    include_lpips = args.include_lpips
    include_clip = args.include_clip
    if not (include_psnr or include_lpips or include_clip):
        include_psnr = include_lpips = include_clip = True

    metrics: list[str] = []
    if include_psnr:
        metrics.append("psnr_unedit_part")
    if include_lpips:
        metrics.append("lpips_unedit_part")
    if include_clip:
        metrics.append("clip_similarity_target_image_edit_part")
    fieldnames = metrics_fieldnames(metrics)
    t_delta = settings.T_DELTA

    # Collect samples from the inputs CSV.
    samples: dict[str, dict[str, str]] = {}
    with open(inputs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != settings.ID_TO_INPUTS_FIELDS:
            raise ValueError(
                f"Unexpected id_to_inputs columns {reader.fieldnames}; "
                f"expected {settings.ID_TO_INPUTS_FIELDS}"
            )
        for row in reader: # type: ignore
            samples[row["sample_id"]] = {k: v for k, v in row.items() if k != "sample_id"} # type: ignore[index]

    # Collect cells from {generated_root}/{sample_id}/cells/.
    cells: defaultdict[str, list[tuple[float, float, Path]]] = defaultdict(list)
    for sample_id in samples:
        cells_dir = generated_root / sample_id / settings.CELLS_DIRNAME
        if not cells_dir.is_dir():
            continue
        for path in sorted(cells_dir.iterdir()):
            match = re.search(r"t_start_(\d+p\d+)__t_end_(\d+p\d+)\.jpg$", path.name)
            if match is None:
                raise ValueError(f"Invalid cell filename: {path.name}")
            t_start, t_end = (float(x.replace("p", ".")) for x in match.groups())
            cells[sample_id].append((t_start, t_end, path))

    if max_samples is not None:
        sample_ids_to_keep = list(cells.keys())[:max_samples]
        cells = defaultdict(list, {sid: cells[sid] for sid in sample_ids_to_keep})

    LOGGER.info("Found %d samples and %d total cells", len(cells), sum(len(v) for v in cells.values()))

    metrics_calculator = MetricsCalculator(DEVICE)

    # OPT 2: Access the internal torchmetrics calculators and the CLIP model/processor
    # directly from MetricsCalculator, so we can call them with pre-built GPU tensors
    # instead of going through the PIL->numpy->tensor conversion on every call.
    psnr_calc = metrics_calculator.psnr_metric_calculator
    lpips_calc = metrics_calculator.lpips_metric_calculator
    clip_model = metrics_calculator.clip_metric_calculator.model

    # OPT 3: Replace the default slow processor with the fast (Rust-based) variant.
    from transformers import AutoProcessor
    clip_processor = AutoProcessor.from_pretrained(
        metrics_calculator.clip_metric_calculator.model.config._name_or_path,
        use_fast=True,
    )

    # Write the header to the result file.
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(fieldnames)

    # OPT 4: Keep the result file open for the entire evaluation instead of
    # re-opening and closing it for every single cell row, eliminating thousands
    # of open()/close() syscall pairs. Flush after each sample for crash safety.
    with open(result_path, "a", newline="", encoding="utf-8") as result_file:
        result_writer = csv.writer(result_file)

        # OPT 5: Wrap the entire evaluation in a single torch.no_grad() context
        # instead of entering/exiting it per-cell for CLIP. No metric computation
        # here requires gradients, so one outer context eliminates repeated
        # context-manager overhead and ensures PSNR/LPIPS also skip grad tracking.
        with torch.no_grad():

            for sample_idx, (sample_id, cell_list) in enumerate(cells.items()):
                sample_start = time.perf_counter()
                sample_meta = samples[sample_id]
                source_image = Image.open(sample_meta["image_path"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                mask_image = Image.open(sample_meta["mask_image_path"])
                mask_array = _format_mask_image(mask_image)

                # OPT 6: Precompute masked source tensors once per sample.
                # The source image and inverse mask are constant across all cells in a sample.
                # We bypass MetricsCalculator's PIL-accepting methods and call its internal
                # torchmetrics calculators (psnr_metric_calculator, lpips_metric_calculator)
                # directly with pre-built GPU tensors, eliminating redundant PIL->numpy->tensor
                # conversion and mask application that would otherwise repeat for every cell.
                src_np = np.array(source_image).astype(np.float32) / 255.0
                # OPT 1 (cont.): mask_array is already float32 from _format_mask_image,
                # so no extra .astype(np.float32) cast is needed here.
                inv_mask = 1.0 - mask_array
                has_unedit_part = inv_mask.sum() > 0
                has_edit_part = mask_array.sum() > 0

                src_psnr_tensor = torch.empty(0)
                src_lpips_tensor = torch.empty(0)
                text_features = torch.empty(0)

                # OPT 7: torch.from_numpy() shares memory with the numpy array
                # instead of copying like torch.tensor(), avoiding a redundant
                # CPU-side memcpy before the .to(DEVICE) GPU transfer.
                if (include_psnr or include_lpips) and has_unedit_part:
                    src_masked_np = src_np * inv_mask
                    src_psnr_tensor = torch.from_numpy(src_masked_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                    src_lpips_tensor = src_psnr_tensor * 2 - 1

                # OPT 8: Encode the target prompt through CLIP's text encoder once per sample.
                # We access the CLIPScore metric's internal model (clip_metric_calculator.model)
                # to extract text features, then compute image-text cosine similarity manually
                # for each cell, avoiding N-1 redundant text forward passes per sample.
                if include_clip and has_edit_part:
                    target_prompt = sample_meta["target_prompt"]
                    text_processed = clip_processor(
                        text=[target_prompt], return_tensors="pt", padding=True, truncation=True,
                    )
                    text_out = clip_model.get_text_features(
                        text_processed["input_ids"].to(DEVICE),
                    )
                    # transformers >=5 (chordedit env) returns BaseModelOutputWithPooling
                    # instead of a plain tensor; pie_eval's older transformers returned a tensor directly.
                    text_features = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # OPT 9: Batch CLIP and LPIPS neural-network forward passes across
                # all cells in a sample. Instead of running these networks once per cell
                # (N separate GPU kernel launches at batch_size=1), we collect all cell
                # images in a first pass (computing the cheap PSNR metric inline), then
                # run LPIPS (SqueezeNet) and CLIP (ViT-L/14) in sub-batches.
                cell_meta = []
                psnr_scores: list = []
                lpips_target_tensors = []
                clip_image_tensors = []

                for t_start, t_end, cell_path in cell_list:
                    target_image = Image.open(cell_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                    # OPT 9: Convert the target image to numpy once per cell and derive
                    # all tensor variants, instead of repeating PIL->numpy->tensor per metric.
                    target_arr = np.array(target_image)

                    if (include_psnr or include_lpips) and has_unedit_part:
                        target_np = target_arr.astype(np.float32) / 255.0
                        target_masked_np = target_np * inv_mask
                        target_psnr_tensor = torch.from_numpy(target_masked_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                        if include_psnr:
                            psnr_scores.append(psnr_calc(target_psnr_tensor, src_psnr_tensor).cpu().item())
                        if include_lpips:
                            lpips_target_tensors.append(target_psnr_tensor * 2 - 1)
                    elif include_psnr:
                        psnr_scores.append("nan")

                    if include_clip and has_edit_part:
                        target_clip_arr = np.uint8(target_arr * mask_array)
                        clip_image_tensors.append(torch.from_numpy(target_clip_arr).permute(2, 0, 1))

                    cell_meta.append(
                        (t_start, t_end, f"{sample_id}/{settings.CELLS_DIRNAME}/{cell_filename(t_start, t_end)}")
                    )

                # OPT 9 (cont.): Batched LPIPS -- run SqueezeNet on stacked target tensors.
                # We call lpips_calc.net directly to get per-sample LPIPS scores;
                # the torchmetrics wrapper would average across the batch.
                lpips_scores: list = []
                if include_lpips:
                    if has_unedit_part and lpips_target_tensors:
                        lpips_scores = []
                        for i in range(0, len(lpips_target_tensors), LPIPS_BATCH_SIZE):
                            batch = torch.cat(lpips_target_tensors[i:i + LPIPS_BATCH_SIZE], dim=0)
                            src_batch = src_lpips_tensor.expand(batch.shape[0], -1, -1, -1)
                            scores = lpips_calc.net(batch, src_batch).squeeze()
                            if scores.dim() == 0:
                                lpips_scores.append(scores.cpu().item())
                            else:
                                lpips_scores.extend(scores.cpu().tolist())
                    else:
                        lpips_scores = ["nan"] * len(cell_meta)

                # OPT 9 (cont.): Batched CLIP -- run the CLIP image encoder on all collected
                # masked cell images at once. clip_processor handles resizing and
                # normalization; clip_model.get_image_features encodes the whole batch
                # in one forward pass. Cosine similarities are computed vectorially.
                clip_scores: list = []
                if include_clip:
                    if has_edit_part and clip_image_tensors:
                        clip_scores = []
                        for i in range(0, len(clip_image_tensors), CLIP_BATCH_SIZE):
                            batch_images = clip_image_tensors[i:i + CLIP_BATCH_SIZE]
                            image_processed = clip_processor(images=batch_images, return_tensors="pt")
                            image_out = clip_model.get_image_features(
                                image_processed["pixel_values"].to(DEVICE)
                            )
                            # See text_features comment above re: transformers >=5.
                            image_features = image_out.pooler_output if hasattr(image_out, "pooler_output") else image_out
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            batch_scores = (100.0 * (image_features @ text_features.T)).squeeze(-1)
                            if batch_scores.dim() == 0:
                                clip_scores.append(batch_scores.cpu().item())
                            else:
                                clip_scores.extend(batch_scores.cpu().tolist())
                    else:
                        clip_scores = ["nan"] * len(cell_meta)

                for idx, (t_start, t_end, rel_cell_path) in enumerate(cell_meta):
                    row: list = [sample_id, f"{t_start:.1f}", f"{t_end:.1f}", f"{t_delta:.1f}"]
                    if include_psnr:
                        row.append(psnr_scores[idx])
                    if include_lpips:
                        row.append(lpips_scores[idx])
                    if include_clip:
                        row.append(clip_scores[idx])
                    row.append(rel_cell_path)
                    result_writer.writerow(row)
                result_file.flush()

                elapsed = time.perf_counter() - sample_start
                LOGGER.info("[%d/%d] Evaluated %s (%d cells in %.2fs)",
                            sample_idx + 1, len(cells), sample_id, len(cell_list), elapsed)

    LOGGER.info("Done. Metrics in %s", result_path.absolute())


if __name__ == "__main__":
    main()
