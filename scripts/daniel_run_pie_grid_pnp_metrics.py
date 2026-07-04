"""
Run the optimized (factorized) t_start by t_end grid ablation over a PIE-Bench /
UltraEdit style folder, then score every generated cell with two metrics copied
from /data/home/mirick/PnPInversion:

  psnr                                    whole-image PSNR, source vs. edited.
  clip_similarity_target_image_edit_part  CLIP similarity of the masked edit
                                          region to the (bracket-stripped) target
                                          prompt.

The metric math is inlined here (see InlineMetrics) so everything runs in a
single env (chordedit). PSNR uses torchmetrics exactly like PnPInversion; CLIP
reproduces torchmetrics' CLIPScore (100 * cosine on openai/clip-vit-large-patch14
image_embeds / text_embeds), computed via a plain CLIP forward so it works with
this env's transformers. Verified bit-for-bit against PnPInversion's pie_eval env.

Generation reuses run_factorized_grid from daniel_run_grid_ablation.py (one VAE
encode + one transport per unique t_start + cleanup/decode per cell). Rows are
flushed to CSV after each source image, so each image is scored right after it is
generated and partial runs are never lost.

Supported input layouts (auto-detected per record from mapping_file.json):
  PIE-Bench : image_path relative to <root>/annotation_images, run-length "mask".
  UltraEdit : image_path / mask_image_path relative to <root>, jpg mask image.
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
import os

import numpy as np

from daniel_create_grid_image import _apply_overlay, _build_base_grid, _save_grid_figure
from daniel_run_grid_ablation import (
    base_edit_config,
    grid_values,
    require_factorizable_config,
    run_factorized_grid,
)
from run_local_ablation import (
    LocalRecord,
    dtype_from_precision,
    ensure_dir,
    param_slug,
    resolve_component_paths,
)

LOGGER = logging.getLogger("pie_grid_pnp_metrics")

DEFAULT_DATA_ROOT = "/shared/ssd_30T/mirick/datasets/PIE-Bench_v1"
DEFAULT_MODEL_ROOT = "/shared/ssd_30T/mirick/models/sd-turbo"
DEFAULT_OUTPUT_ROOT = "outputs/pie_grid_pnp_metrics"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
IMAGE_SIZE = 512
SEED = 42
CSV_FIELDS = [
    "sample_id",
    "category",
    "t_start",
    "t_end",
    "t_delta",
    "psnr",
    "clip_similarity_target_image_edit_part",
    "cell_path",
]


class InlineMetrics:
    """PnPInversion's psnr / clip_similarity metrics, inlined for a single env.

    - calculate_psnr mirrors MetricsCalculator.calculate_psnr (torchmetrics PSNR,
      data_range=1.0, whole image).
    - calculate_clip_similarity reproduces MetricsCalculator.calculate_clip_similarity
      (torchmetrics CLIPScore == 100 * cosine on CLIP image/text embeds), computed
      with a direct CLIP forward so it works with this env's transformers.
    """

    def __init__(self, device: str, clip_model_id: str = CLIP_MODEL_ID):
        import torch
        from torchmetrics.image import PeakSignalNoiseRatio
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()  # type: ignore[union-attr]
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    def calculate_psnr(self, img_pred, img_gt) -> float:
        torch = self._torch
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        pred = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        gt = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.psnr_metric_calculator(pred, gt).cpu().item()

    def calculate_clip_similarity(self, img, txt: str, mask=None) -> float:
        return self.calculate_clip_similarity_batch([img], txt, mask)[0]

    def calculate_psnr_batch(self, img_gt, imgs_pred) -> list[float]:
        """Per-image PSNR (data_range=1.0) for many predictions against one source.

        Matches torchmetrics PeakSignalNoiseRatio applied per cell: for data_range=1
        PSNR = -10 * log10(mean_channels_pixels((pred - gt) ** 2)). Computed in one
        batched GPU op instead of N torchmetrics calls.
        """
        torch = self._torch
        gt = torch.tensor(np.array(img_gt).astype(np.float32) / 255).permute(2, 0, 1).to(self.device)
        preds = torch.stack(
            [torch.tensor(np.array(im).astype(np.float32) / 255).permute(2, 0, 1) for im in imgs_pred]
        ).to(self.device)
        mse = ((preds - gt.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3))
        return (-10.0 * torch.log10(mse)).cpu().tolist()

    def calculate_clip_similarity_batch(self, imgs, txt: str, mask=None, batch_size: int = 64) -> list[float]:
        """CLIP edit-part similarity (100 * cosine) for many images vs one prompt.

        Same math as calculate_clip_similarity, but masks all images up front and
        runs the CLIP image tower in batches (one prompt encode per chunk) instead of
        one forward per image.
        """
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
                text=[txt], images=chunk, return_tensors="pt", padding=True, truncation=True  # pyright: ignore[reportCallIssue]
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.clip_model(**inputs)
            img_emb = out.image_embeds / out.image_embeds.norm(p=2, dim=-1, keepdim=True)
            txt_emb = out.text_embeds / out.text_embeds.norm(p=2, dim=-1, keepdim=True)
            scores.extend((100 * (img_emb * txt_emb).sum(dim=-1)).cpu().tolist())
        return scores


def mask_decode(encoded_mask, image_shape=(IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    """PIE-Bench run-length mask -> HxW {0,1} (border forced to 1), from PnPInversion."""
    length = image_shape[0] * image_shape[1]
    mask = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded_mask), 2):
        start = encoded_mask[i]
        run = min(encoded_mask[i + 1], length - start)
        mask[start : start + run] = 1.0
    mask = mask.reshape(image_shape)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 1
    return mask


def save_sample_grids(
    cells_dir: Path,
    sample_dir: Path,
    values,
    t_delta: float,
    title: str,
    rows,
) -> None:
    """Build a clean grid plus PSNR / CLIP overlay grids for one sample.

    Reuses the grid builders from daniel_create_grid_image.py; cells must be named
    with param_slug (t_start_0p0__t_end_1p0.jpg), which is what this script writes.
    """
    built = _build_base_grid(cells_dir, values, values, cell_extension=".jpg")
    if built is None:
        LOGGER.warning("No cells found under %s; skipping grid image.", cells_dir)
        return
    base_canvas, slots = built

    _save_grid_figure(
        base_canvas,
        sample_dir / "grid_clean.png",
        title=title,
        values_start=values,
        values_end=values,
        t_delta=t_delta,
    )

    def finite_scores(column: str):
        scores = {}
        for row in rows:
            value = row[column]
            if isinstance(value, (int, float)) and not (isinstance(value, float) and value != value):
                scores[(round(row["t_start"], 1), round(row["t_end"], 1))] = float(value)
        return scores

    for name, column, label in (
        ("grid_psnr.png", "psnr", "Whole PSNR"),
        ("grid_clip.png", "clip_similarity_target_image_edit_part", "CLIP-Edited"),
    ):
        scores = finite_scores(column)
        if not scores:
            continue
        vmin, vmax = min(scores.values()), max(scores.values())
        overlay = _apply_overlay(base_canvas, slots, scores, vmin=vmin, vmax=vmax)
        _save_grid_figure(
            overlay,
            sample_dir / name,
            title=title,
            values_start=values,
            values_end=values,
            t_delta=t_delta,
            colorbar_label=label,
            colorbar_vmin=vmin,
            colorbar_vmax=vmax,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Dataset root (PIE-Bench / UltraEdit style).")
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT, help="SD/SDXL component root.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Where cells + result.csv go.")
    parser.add_argument("--clip-model", default=CLIP_MODEL_ID, help="CLIP model id for the edit-part metric.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only process the first N samples (per shard).")
    parser.add_argument("--overwrite", action="store_true", help="Re-run samples already present in the result CSV.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split samples across this many GPU workers.")
    parser.add_argument("--shard", type=int, default=0, help="Which shard this process handles (0-based).")
    args = parser.parse_args()
    if not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, {args.num_shards}); got {args.shard}")
    return args


def strip_brackets(text: str) -> str:
    """PIE/UltraEdit mark edited words with [ ]; drop the markers for prompting/CLIP."""
    return text.replace("[", "").replace("]", "").strip()


def resolve_under(root: Path, rel: str) -> Path:
    """Resolve a mapping path that may be relative to root or root/annotation_images."""
    direct = root / rel
    if direct.exists():
        return direct
    return root / "annotation_images" / rel


def load_samples(mapping_path: Path, max_samples, shard: int = 0, num_shards: int = 1):
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    sample_ids = [sid for sid in sorted(mapping) if mapping[sid].get("image_path")]
    # Round-robin split so each GPU worker gets a balanced, disjoint slice.
    sample_ids = sample_ids[shard::num_shards]
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    return [(sid, mapping[sid]) for sid in sample_ids]


def load_mask(root: Path, meta: dict) -> np.ndarray:
    """Return an HxWx3 {0,1} edit mask from either a jpg mask image or a run-length mask."""
    from PIL import Image

    mask_rel = meta.get("mask_image_path")
    if mask_rel:
        with Image.open(resolve_under(root, mask_rel)) as mimg:
            gray = mimg.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        mask = (np.array(gray) > 127).astype(np.float32)
    elif meta.get("mask"):
        mask = mask_decode(meta["mask"], (IMAGE_SIZE, IMAGE_SIZE))
    else:
        mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return mask[:, :, np.newaxis].repeat(3, axis=2)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    args = parse_args()

    import torch
    from PIL import Image

    from pipeline_chord import ChordEditPipeline

    # Ampere TF32 + cuDNN autotune: speeds up the fp32 UNet forwards and VAE decode
    # (the dominant cost) with negligible numeric change. Input shapes per forward
    # type are constant across cells, so benchmark autotuning pays off.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root).expanduser().resolve()
    mapping_path = data_root / "mapping_file.json"
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_dir(output_root)
    # Per-shard CSV avoids concurrent-write races between GPU workers; merge later.
    csv_path = output_root / ("result.csv" if args.num_shards == 1 else f"result_shard{args.shard:02d}.csv")

    samples = load_samples(mapping_path, args.max_samples, args.shard, args.num_shards)
    values = grid_values("default")
    base_config = base_edit_config("default")
    require_factorizable_config(base_config)
    t_delta = 0.0

    LOGGER.info("Data root: %s (%d sample(s))", data_root, len(samples))
    if args.num_shards > 1:
        LOGGER.info("Shard %d/%d -> %s", args.shard, args.num_shards, csv_path.name)
    LOGGER.info("Grid: %dx%d t_start/t_end values, t_delta=%.2f", len(values), len(values), t_delta)
    LOGGER.info("Output: %s", output_root)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=resolve_component_paths(args.model_root, "sd"),
        model_type="sd",
        default_edit_config=base_config,
        device=device,
        torch_dtype=dtype_from_precision("fp32"),
        image_size=IMAGE_SIZE,
        use_center_crop=True,
        compute_dtype=torch.float32,
        use_attention_mask=False,
        use_safety_checker=False,
        chord_edit_mode="default",
    )

    LOGGER.info("Loading inlined PnPInversion metrics (PSNR + CLIP) on %s ...", device)
    metrics = InlineMetrics(device, args.clip_model)

    # Resume against every result CSV in the output dir (result.csv + all shard CSVs)
    # so a multi-GPU run skips samples already completed by any prior/other run and
    # only generates genuinely new folders.
    done_ids = set()
    if not args.overwrite:
        for existing in sorted(output_root.glob("result*.csv")):
            with existing.open("r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    sid = row.get("sample_id")
                    if sid:
                        done_ids.add(sid)
    write_header = not csv_path.exists()

    # Keep the CSV open for the whole run and flush after each sample so the file
    # on disk always reflects every image completed so far (never buffered to the end).
    csv_file = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    for index, (sample_id, meta) in enumerate(samples, start=1):
        if sample_id in done_ids:
            LOGGER.info("[%d/%d] %s already scored; skipping.", index, len(samples), sample_id)
            continue

        image_path = resolve_under(data_root, meta["image_path"])
        category = Path(meta["image_path"]).parent.name
        source_prompt = strip_brackets(meta.get("original_prompt", ""))
        target_prompt = strip_brackets(meta.get("editing_prompt", ""))
        LOGGER.info("[%d/%d] %s (%s): %r", index, len(samples), sample_id, category, target_prompt)

        with Image.open(image_path) as img:
            source_image = img.convert("RGB")
        src_for_metric = source_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        mask = load_mask(data_root, meta)

        record = LocalRecord(
            sample_name=category,
            image_path=image_path,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_prompt=strip_brackets(meta.get("editing_instruction", "")),
            edit_id=sample_id,
        )

        cells = run_factorized_grid(
            pipeline=pipeline,
            source_image=source_image,
            record=record,
            base_config=base_config,
            t_start_values=values,
            t_end_values=values,
            t_delta=t_delta,
            seed=args.seed,
        )

        cells_dir = output_root / sample_id / "cells"
        ensure_dir(cells_dir)

        # Collect cells in grid order, then score PSNR + CLIP in batched GPU ops.
        cell_specs = []
        generated_list = []
        for t_end in values:
            for t_start in values:
                generated = cells[(t_start, t_end)].convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                cell_path = cells_dir / f"{param_slug('t_start', t_start)}__{param_slug('t_end', t_end)}.jpg"
                generated.save(cell_path, quality=92)
                generated_list.append(generated)
                cell_specs.append((t_start, t_end, cell_path))

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
        grid_title = (
            f"{sample_id} ({category})\n"
            f'Source: "{source_prompt}"\n'
            f'Target: "{target_prompt}"'
        )
        save_sample_grids(cells_dir, output_root / sample_id, values, t_delta, grid_title, rows)

        LOGGER.info("[%d/%d] Done sample_id %s (%d cells scored)", index, len(samples), sample_id, len(rows))

    csv_file.close()
    LOGGER.info("Done. Results in %s", csv_path)


if __name__ == "__main__":
    main()
