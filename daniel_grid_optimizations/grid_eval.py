"""
Evaluate the metrics of a grid of cells for every sample in a dataset.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import List

import settings
from _helpers import cell_filename, metrics_fieldnames

LOGGER = logging.getLogger("grid_eval")

IMAGE_SIZE = 512
CLIP_BATCH_SIZE = 32
LPIPS_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--inputs-path", default=None)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-psnr", action="store_true", default=False)
    parser.add_argument("--include-lpips", action="store_true", default=False)
    parser.add_argument("--include-clip", action="store_true", default=False)
    return parser.parse_args()


def _format_mask_image(mask_image: "Image.Image") -> "np.ndarray":
    import numpy as np
    from PIL import Image

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


def run_shard(
    *,
    generated_root: Path,
    inputs_path: Path,
    result_path: Path,
    metrics: List[str],
    include_psnr: bool,
    include_lpips: bool,
    include_clip: bool,
    max_samples: int | None,
    shard: int,
    num_shards: int,
    gpu: int,
) -> None:
    """Evaluate one round-robin shard of samples on a single GPU."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from PIL import Image
    from evaluation.matrics_calculator import MetricsCalculator

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    DEVICE = torch.device("cuda:0")
    fieldnames = metrics_fieldnames(metrics)
    t_delta = settings.T_DELTA

    # Collect samples from the inputs CSV.
    all_samples: dict[str, dict[str, str]] = {}
    with open(inputs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != settings.ID_TO_INPUTS_FIELDS:
            raise ValueError(
                f"Unexpected id_to_inputs columns {reader.fieldnames}; "
                f"expected {settings.ID_TO_INPUTS_FIELDS}"
            )
        for row in reader:
            all_samples[row["sample_id"]] = {k: v for k, v in row.items() if k != "sample_id"}

    # Collect cells from {generated_root}/grids/{sample_id}/cells/.
    all_cells: defaultdict[str, list[tuple[float, float, Path]]] = defaultdict(list)
    for sample_id in all_samples:
        cells_dir = generated_root / settings.GRIDS_DIRNAME / sample_id / settings.CELLS_DIRNAME
        if not cells_dir.is_dir():
            continue
        for path in sorted(cells_dir.iterdir()):
            match = re.search(r"t_start_(\d+p\d+)__t_end_(\d+p\d+)\.jpg$", path.name)
            if match is None:
                raise ValueError(f"Invalid cell filename: {path.name}")
            t_start, t_end = (float(x.replace("p", ".")) for x in match.groups())
            all_cells[sample_id].append((t_start, t_end, path))

    # Apply max_samples cap, then round-robin shard.
    sample_ids = list(all_cells.keys())
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    shard_ids = sample_ids[shard::num_shards]

    LOGGER.info(
        "GPU %d: Started shard %d/%d (%d sample%s)",
        gpu, shard + 1, num_shards, len(shard_ids), "s" * (len(shard_ids) != 1),
    )

    metrics_calculator = MetricsCalculator(DEVICE)

    # OPT 2: Access the internal torchmetrics calculators and the CLIP model/processor
    # directly from MetricsCalculator, so we can call them with pre-built GPU tensors
    # instead of going through the PIL->numpy->tensor conversion on every call.
    psnr_calc = metrics_calculator.psnr_metric_calculator
    lpips_calc = metrics_calculator.lpips_metric_calculator
    clip_model = metrics_calculator.clip_metric_calculator.model

    # OPT 3: Split text tokenization from image preprocessing so the image side can
    # use TorchvisionBackend explicitly. `use_fast=True` on the combined AutoProcessor
    # silently no-ops on this transformers version (both paths resolve to the same
    # slow CPU-bound class) -- TorchvisionBackend is only actually fast when it runs
    # on GPU-resident tensors, which requires calling it separately from tokenization.
    from transformers import AutoTokenizer, AutoImageProcessor
    clip_model_name = metrics_calculator.clip_metric_calculator.model.config._name_or_path
    clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    clip_image_processor = AutoImageProcessor.from_pretrained(
        clip_model_name, use_fast=True, backend="torchvision",
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
            for sample_idx, sample_id in enumerate(shard_ids):
                sample_start = time.perf_counter()
                cell_list = all_cells[sample_id]
                sample_meta = all_samples[sample_id]
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
                    text_processed = clip_tokenizer(
                        [target_prompt], return_tensors="pt", padding=True, truncation=True,
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
                        # Move to GPU now (not after preprocessing) so TorchvisionBackend's
                        # resize/normalize below actually run on GPU tensors -- its fast path
                        # only engages for GPU-resident input; on CPU tensors it's slower than
                        # the PIL backend (measured: 2.7s vs 0.076s for a 32-image batch).
                        clip_image_tensors.append(torch.from_numpy(target_clip_arr).permute(2, 0, 1).to(DEVICE))

                    cell_meta.append(
                        (
                            t_start,
                            t_end,
                            f"{settings.GRIDS_DIRNAME}/{sample_id}/{settings.CELLS_DIRNAME}/{cell_filename(t_start, t_end)}",
                        )
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
                # masked cell images at once. clip_image_processor handles resizing and
                # normalization; clip_model.get_image_features encodes the whole batch
                # in one forward pass. Cosine similarities are computed vectorially.
                clip_scores: list = []
                if include_clip:
                    if has_edit_part and clip_image_tensors:
                        clip_scores = []
                        for i in range(0, len(clip_image_tensors), CLIP_BATCH_SIZE):
                            batch_images = clip_image_tensors[i:i + CLIP_BATCH_SIZE]
                            image_processed = clip_image_processor(images=batch_images, return_tensors="pt")
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
                LOGGER.info("GPU %d: [%d/%d] Evaluated %s (%d cells in %.2fs)",
                            gpu, sample_idx + 1, len(shard_ids), sample_id, len(cell_list), elapsed)

    LOGGER.info("GPU %d: Finished shard %d/%d.", gpu, shard + 1, num_shards)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    args = parse_args()
    max_samples = args.max_samples
    gpus = args.gpus
    generated_root = Path(args.generated_root).expanduser().resolve()
    # Check that the generated root is a directory.
    if not generated_root.is_dir():
        raise FileNotFoundError(f"generated-root is not a directory: {generated_root}")
    suffix = generated_root.name.lower().replace("_", "").replace("-", "")
    # Check that the inputs CSV exists.
    inputs_path = generated_root / f"id_to_inputs_{generated_root.name.lower().replace('_', '').replace('-', '')}.csv"
    if not inputs_path.is_file():
        raise FileNotFoundError(f"Missing inputs CSV (expected generated layout): {inputs_path}")
    metrics_path = (
        Path(args.result_path).expanduser().resolve()
        if args.result_path
        else generated_root / f"id_to_metrics_{suffix}{f'_n{max_samples}' if max_samples is not None else ''}.csv"
    )

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

    # OPT 10: Multi-GPU sharding -- distribute samples round-robin across GPUs.
    # Each GPU runs an independent spawned process with its own CUDA context,
    # MetricsCalculator, and CLIP/LPIPS models. Every shard writes to its own
    # temporary CSV (avoiding write races), and main() merges them at the end.
    # With N GPUs the wall-clock time is ~1/N of single-GPU evaluation.
    shard_paths = [
        metrics_path.with_suffix(f".shard{shard}.csv")
        for shard in range(len(gpus))
    ]

    kwargs = dict(
        generated_root=generated_root,
        inputs_path=inputs_path,
        metrics=metrics,
        include_psnr=include_psnr,
        include_lpips=include_lpips,
        include_clip=include_clip,
        max_samples=max_samples,
        num_shards=len(gpus),
    )

    if len(gpus) == 1:
        run_shard(result_path=shard_paths[0], shard=0, gpu=gpus[0], **kwargs)
    else:
        ctx = get_context("spawn")
        processes = [
            ctx.Process(target=run_shard, kwargs={**kwargs, "result_path": shard_paths[shard], "shard": shard, "gpu": gpu})
            for shard, gpu in enumerate(gpus)
        ]
        for process in processes:
            process.start()
            # Stagger by a second so .
            time.sleep(1.0)
        failed = False
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failed = True
        if failed:
            raise SystemExit("One or more eval shards failed.")

    # Merge per-shard CSVs into the final result file.
    with open(metrics_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(fieldnames)
        for shard_path in shard_paths:
            with open(shard_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    writer.writerow([row[col] for col in fieldnames])
            shard_path.unlink()

    LOGGER.info("Done. Metrics in %s", metrics_path.absolute())


if __name__ == "__main__":
    main()
