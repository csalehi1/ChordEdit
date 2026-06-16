import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_METRICS = [
    "structure_distance",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
]


def mask_decode(encoded_mask, image_shape=(512, 512)):
    """PIE-Bench/PnPInversion RLE mask decoder."""
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,), dtype=np.float32)

    for i in range(0, len(encoded_mask), 2):
        start = int(encoded_mask[i])
        run_len = int(encoded_mask[i + 1])
        splice_len = min(run_len, length - start)
        if splice_len > 0:
            mask_array[start:start + splice_len] = 1.0

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])

    # Match PnPInversion boundary handling.
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def to_float(value):
    if isinstance(value, str):
        return math.nan
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return math.nan


def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask, src_prompt, tgt_prompt):
    """Same metric choices as PnPInversion evaluation/evaluate.py."""
    if metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)

    if metric == "psnr_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_psnr(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)

    if metric == "lpips_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_lpips(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)

    if metric == "mse_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_mse(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)

    if metric == "ssim_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        return metrics_calculator.calculate_ssim(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)

    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)

    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)

    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)

    raise ValueError(f"Unknown metric: {metric}")


def build_image_index(method_root):
    """Index generated jpgs by PIE relative path, e.g. 0_random_140/000000000000.jpg."""
    index = {}
    for p in method_root.rglob("*.jpg"):
        if len(p.parts) >= 2:
            rel = Path(p.parts[-2]) / p.parts[-1]
            index[str(rel)] = p
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnp-root", type=Path, default=Path.home() / "research/PnPInversion")
    parser.add_argument("--pie-root", type=Path, default=Path.home() / "datasets/PIE-Bench_v1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/shared/ssd_30T/zarageddes/chordedit_original_ablations/output"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("/shared/ssd_30T/zarageddes/chordedit_original_ablations/results_piebench_protocol"),
    )
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, str(args.pnp_root))
    from evaluation.matrics_calculator import MetricsCalculator

    args.result_root.mkdir(parents=True, exist_ok=True)

    mapping_path = args.pie_root / "mapping_file.json"
    src_root = args.pie_root / "annotation_images"

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    items = list(mapping.items())
    if args.max_samples is not None:
        items = items[: args.max_samples]

    print(f"Loaded {len(items)} PIE samples")
    print(f"Methods: {args.methods}")

    metrics_calculator = MetricsCalculator(args.device)

    summary_rows = []
    per_image_path = args.result_root / "per_image_metrics.csv"

    with open(per_image_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "file_id", "image_path"] + DEFAULT_METRICS)

        for method in args.methods:
            method_root = args.output_root / method / "annotation_images"
            if not method_root.exists():
                raise FileNotFoundError(f"Missing method annotation_images folder: {method_root}")

            index = build_image_index(method_root)
            print(f"\nEvaluating {method}: indexed {len(index)} generated jpgs")

            method_values = {metric: [] for metric in DEFAULT_METRICS}
            used = 0
            missing = 0

            for file_id, item in items:
                rel_image_path = item["image_path"]
                src_path = src_root / rel_image_path
                tgt_path = index.get(rel_image_path)

                if tgt_path is None:
                    missing += 1
                    continue

                src_image = Image.open(src_path).convert("RGB")
                tgt_image = Image.open(tgt_path).convert("RGB")

                if tgt_image.size[0] != tgt_image.size[1]:
                    tgt_image = tgt_image.crop(
                        (
                            tgt_image.size[0] - 512,
                            tgt_image.size[1] - 512,
                            tgt_image.size[0],
                            tgt_image.size[1],
                        )
                    )

                mask = mask_decode(item["mask"])
                mask = mask[:, :, np.newaxis].repeat(3, axis=2)

                original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
                editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")

                row = [method, file_id, rel_image_path]
                for metric in DEFAULT_METRICS:
                    val = calculate_metric(
                        metrics_calculator,
                        metric,
                        src_image,
                        tgt_image,
                        mask,
                        mask,
                        original_prompt,
                        editing_prompt,
                    )
                    val = to_float(val)
                    row.append(val)
                    if not math.isnan(val):
                        method_values[metric].append(val)

                writer.writerow(row)
                used += 1

            summary = {
                "method": method,
                "rows": used,
                "missing": missing,
            }
            for metric in DEFAULT_METRICS:
                vals = method_values[metric]
                summary[metric] = float(np.mean(vals)) if vals else math.nan

            summary_rows.append(summary)
            print(f"Finished {method}: rows={used}, missing={missing}")

    summary_path = args.result_root / "summary_metrics.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["method", "rows", "missing"] + DEFAULT_METRICS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\nDone.")
    print(f"Per-image metrics: {per_image_path}")
    print(f"Summary metrics:   {summary_path}")


if __name__ == "__main__":
    main()
