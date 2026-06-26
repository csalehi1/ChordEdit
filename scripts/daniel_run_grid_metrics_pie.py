"""
Generate a t_start x t_end x t_delta grid of edits for the first image in each
PIE-Bench category folder (1_... to 9_...), score every generated image with a
whole-image PSNR (vs. the source) and a mask-restricted CLIP-edited score, and
dump everything to id_to_metrics_sdturbo_top.csv.

Image generation reuses the factorized optimization from
scripts/daniel_run_grid_ablation.py (run_factorized_grid): one VAE encode,
one transport per unique t_start, then cleanup+decode per (t_start, t_end) cell.

Grid:
  t_start in [0.0, 1.0] step 0.1   (11 values)
  t_end   in [0.0, 1.0] step 0.1   (11 values)
  t_delta in {0.15, 0.0}
  => 11 * 11 * 2 = 242 images per source image, 9 source images.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))         # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

import argparse
import csv
import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from run_local_ablation import (
    LocalRecord,
    dtype_from_precision,
    ensure_dir,
    resolve_component_paths,
)
from run_grid_ablation import (
    make_axis_grid,
    save_source_copy,
    t_delta_conditions,
    value_slug,
)
from daniel_run_grid_ablation import run_factorized_grid
from run_pie_bench import DEFAULT_EDIT_CONFIG


LOGGER = logging.getLogger("grid_metrics_pie")

# --- Fixed task configuration --------------------------------------------- #
PIE_ROOT = Path("/data/home/mirick/datasets/PIE-Bench_v1")
MODEL_ROOT = "/data/home/mirick/models/sd-turbo"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATION_ROOT = REPO_ROOT / "ablation_outputs"
DEFAULT_OUTPUT_NAME = "grid_metrics_sdturbo_top"
DEFAULT_CSV_NAME = "id_to_metrics_sdturbo_top.csv"
DEFAULT_CATEGORY_PREFIXES = [str(i) for i in range(1, 10)]  # 1..9
DEFAULT_T_END_VALUES = [round(i / 10.0, 1) for i in range(11)]
DEFAULT_T_START_VALUES = [round(i / 10.0, 1) for i in range(11)]
SEED = 42
IMAGE_SIZE = 512


def cell_filename(t_start: float, t_end: float) -> str:
    return f"t_start_{value_slug(t_start)}__t_end_{value_slug(t_end)}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate t_start x t_end x t_delta grids over PIE-Bench images and score "
            "each with whole-image PSNR + mask-restricted CLIP-edited similarity."
        )
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=DEFAULT_CATEGORY_PREFIXES,
        help="Category folder prefixes to sample from (e.g. 0, or 1 2 3). Default: 1..9.",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=1,
        help="How many images (smallest sample id first) to take from each category. Default: 1.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Base name of the output folder inside ablation_outputs/.",
    )
    parser.add_argument(
        "--append-datetime",
        action="store_true",
        help="Append a _YYYYmmdd_HHMMSS suffix to the output folder (and root CSV).",
    )
    parser.add_argument(
        "--t-start-values",
        type=float,
        nargs="+",
        default=None,
        help="t_start sweep values. Default: 0.0..1.0 step 0.1.",
    )
    parser.add_argument(
        "--t-end-values",
        type=float,
        nargs="+",
        default=None,
        help="t_end sweep values. Default: 0.0..1.0 step 0.1.",
    )
    parser.add_argument(
        "--t-delta-values",
        type=float,
        nargs="+",
        default=None,
        help="t_delta conditions. Default: base config t_delta plus 0.0.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. cuda:2 or cpu.",
    )
    return parser.parse_args()


def strip_brackets(text: str) -> str:
    """PIE-Bench prompts mark edited words with [ ]; remove the markers."""
    return text.replace("[", "").replace("]", "").strip()


def select_records(
    mapping: Dict[str, Any],
    category_prefixes: List[str],
    max_per_category: Optional[int],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Select records grouped by category folder prefix, smallest sample id first.

    Within each requested prefix, take up to ``max_per_category`` images
    (sorted by sample id). Categories are returned in the requested order.
    """
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for sample_id in sorted(mapping.keys()):
        meta = mapping[sample_id]
        image_path = meta.get("image_path")
        if not image_path:
            continue
        category = str(image_path).split("/")[0]
        prefix = category.split("_")[0]
        if prefix not in category_prefixes:
            continue
        grouped.setdefault(prefix, []).append((sample_id, meta))

    selected: List[Tuple[str, Dict[str, Any]]] = []
    for prefix in category_prefixes:
        items = grouped.get(prefix, [])
        if max_per_category is not None:
            items = items[:max_per_category]
        selected.extend(items)
    return selected


def mask_decode(encoded_mask: List[int], image_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Decode the PIE-Bench run-length mask into an HxW {0,1} array.

    Mirrors the reference PIE-Bench decoder, including forcing a 1-pixel border
    to 1 to absorb annotation boundary errors.
    """
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded_mask), 2):
        start = encoded_mask[i]
        splice_len = min(encoded_mask[i + 1], length - start)
        mask_array[start : start + splice_len] = 1.0
    mask_array = mask_array.reshape(image_shape)
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1
    return mask_array


def compute_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Whole-image PSNR (dB) on uint8 RGB arrays, data range 255."""
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    mse = np.mean((ref - cand) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


class ClipScorer:
    def __init__(self, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    def edited_score(self, image: "Any", text: str, mask: np.ndarray) -> float:
        """Cosine similarity between the masked edit region and the target text."""
        from PIL import Image

        torch = self._torch
        rgb = np.array(image.convert("RGB"))
        masked = (rgb * mask[:, :, None]).astype(np.uint8)
        masked_pil = Image.fromarray(masked)

        inputs = self.processor(
            text=[text],
            images=masked_pil,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return float((img_emb * txt_emb).sum(dim=-1).item())


def source_as_model_array(pipeline: Any, source_image: Any) -> np.ndarray:
    """Source image preprocessed exactly as the VAE sees it (center-cropped,
    resized to IMAGE_SIZE), returned as a uint8 RGB array for PSNR."""
    pixel_values = pipeline._prepare_image_tensor(source_image)  # [-1, 1]
    arr = ((pixel_values[0].float().clamp(-1.0, 1.0) + 1.0) / 2.0)
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import torch
    from PIL import Image

    from pipeline_chord import ChordEditPipeline

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{args.output_name}_{timestamp}" if args.append_datetime else args.output_name
    output_root = ABLATION_ROOT / folder_name
    csv_stem = Path(DEFAULT_CSV_NAME).stem
    csv_suffix = Path(DEFAULT_CSV_NAME).suffix
    root_csv_name = f"{csv_stem}_{timestamp}{csv_suffix}" if args.append_datetime else DEFAULT_CSV_NAME
    csv_path = REPO_ROOT / root_csv_name

    mapping_path = PIE_ROOT / "mapping_file.json"
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    selected = select_records(mapping, list(args.categories), args.max_per_category)
    LOGGER.info(
        "Selected %d image(s) from categories %s (max %s each)",
        len(selected),
        args.categories,
        args.max_per_category,
    )
    LOGGER.info("Output folder: %s", output_root)

    image_root = PIE_ROOT / "annotation_images"
    base_config = dict(DEFAULT_EDIT_CONFIG)
    t_start_values = (
        list(args.t_start_values) if args.t_start_values is not None else list(DEFAULT_T_START_VALUES)
    )
    t_end_values = (
        list(args.t_end_values) if args.t_end_values is not None else list(DEFAULT_T_END_VALUES)
    )
    delta_values = (
        list(args.t_delta_values)
        if args.t_delta_values is not None
        else t_delta_conditions(base_config)
    )
    LOGGER.info("t_start values: %s", t_start_values)
    LOGGER.info("t_end values: %s", t_end_values)
    LOGGER.info("t_delta conditions: %s", delta_values)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    component_paths = resolve_component_paths(MODEL_ROOT, "sd")
    torch_dtype = dtype_from_precision("fp32")
    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        model_type="sd",
        default_edit_config=base_config,
        device=device,
        torch_dtype=torch_dtype,
        image_size=IMAGE_SIZE,
        use_center_crop=True,
        compute_dtype=torch.float32,
        use_attention_mask=False,
        use_safety_checker=False,
        chord_edit_mode="default",
    )

    clip_scorer = ClipScorer(device)

    ensure_dir(output_root)
    rows: List[Dict[str, Any]] = []

    for (sample_id, meta) in selected:
        image_rel = meta["image_path"]
        category = image_rel.split("/")[0]
        image_path = image_root / image_rel
        source_prompt = strip_brackets(meta.get("original_prompt", ""))
        target_prompt = strip_brackets(meta.get("editing_prompt", ""))
        edit_prompt = strip_brackets(meta.get("editing_instruction", ""))

        record = LocalRecord(
            sample_name=category,
            image_path=image_path,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_prompt=edit_prompt,
            edit_id=sample_id,
        )

        with Image.open(image_path) as img:
            source_image = img.convert("RGB")

        mask = mask_decode(meta["mask"], (IMAGE_SIZE, IMAGE_SIZE))
        source_arr = source_as_model_array(pipeline, source_image)

        sample_dir = output_root / f"{category}_{sample_id}"
        ensure_dir(sample_dir)
        save_source_copy(source_image, sample_dir / "source.png", overwrite=True)

        LOGGER.info("Processing %s (%s): target=%r", sample_id, category, target_prompt)

        for t_delta in delta_values:
            condition_dir = sample_dir / f"t_delta_{value_slug(t_delta)}"
            cells_dir = condition_dir / "cells"
            ensure_dir(cells_dir)

            images = run_factorized_grid(
                pipeline=pipeline,
                source_image=source_image,
                record=record,
                base_config=base_config,
                t_start_values=t_start_values,
                t_end_values=t_end_values,
                t_delta=t_delta,
                seed=SEED,
            )

            grid_cells: List[List[Image.Image]] = []
            for t_end in t_end_values:
                row_images: List[Image.Image] = []
                for t_start in t_start_values:
                    generated = images[(t_start, t_end)]
                    filename = cell_filename(t_start, t_end)
                    out_path = cells_dir / filename
                    generated.save(out_path)
                    row_images.append(generated)

                    gen_arr = np.array(generated.convert("RGB"))
                    whole_psnr = compute_psnr(source_arr, gen_arr)
                    clip_edited = clip_scorer.edited_score(generated, target_prompt, mask)

                    image_id = (
                        f"{sample_id}_ts{value_slug(t_start)}_te{value_slug(t_end)}"
                        f"_td{value_slug(t_delta)}"
                    )
                    rows.append(
                        {
                            "id": image_id,
                            "sample_id": sample_id,
                            "category": category,
                            "t_start": t_start,
                            "t_end": t_end,
                            "t_delta": t_delta,
                            "whole_psnr": round(whole_psnr, 6),
                            "clip_edited": round(clip_edited, 6),
                            "image_path": str(out_path),
                        }
                    )
                grid_cells.append(row_images)

            grid_path = condition_dir / "grid_t_start_x_t_end.png"
            make_axis_grid(
                title=f"{category}_{sample_id} | x=t_start, y=t_end | t_delta={t_delta:g}",
                cells=grid_cells,
                t_start_values=t_start_values,
                t_end_values=t_end_values,
                destination=grid_path,
                cell_size=160,
            )
            LOGGER.info(
                "  t_delta=%g done (%d cells)",
                t_delta,
                len(t_start_values) * len(t_end_values),
            )

    fieldnames = [
        "id",
        "sample_id",
        "category",
        "t_start",
        "t_end",
        "t_delta",
        "whole_psnr",
        "clip_edited",
        "image_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Keep a copy (canonical name) alongside the generated grids too.
    with (output_root / DEFAULT_CSV_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Wrote %d rows to %s", len(rows), csv_path)


if __name__ == "__main__":
    main()
