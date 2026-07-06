"""Generate the t_start by t_end image grid for every source image in a dataset.

For each sample in <data-root>/mapping_file.json this generates the full
GRID_VALUES x GRID_VALUES set of cells (factorized fast path) and writes:

    <output-root>/<sample_id>/cells/t_start_<..>__t_end_<..>.jpg
    <output-root>/<sample_id>/grid_clean.png

No metrics are computed here (that is label_grid.py's job). Samples whose cells
already exist are skipped unless --overwrite is given, so runs are resumable and
shardable across GPUs.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import settings
from common import (
    LocalRecord,
    cell_filename,
    ensure_dir,
    load_pipeline,
    load_samples,
    strip_brackets,
    resolve_under,
    write_id_to_inputs,
    write_id_to_prompts,
)
from grid_render import save_clean_grid
from pipeline_ops import run_factorized_grid

LOGGER = logging.getLogger("generate_grid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=settings.DEFAULT_DATA_ROOT, help="Dataset root (PIE-Bench / UltraEdit style).")
    parser.add_argument("--model-root", default=settings.DEFAULT_MODEL_ROOT, help="SD component root.")
    parser.add_argument("--output-root", default=settings.DEFAULT_OUTPUT_ROOT, help="Where cells + grid_clean go.")
    parser.add_argument("--chord-edit-mode", choices=["default", "sym"], default=settings.CHORD_EDIT_MODE)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=settings.SEED)
    parser.add_argument("--max-samples", type=int, default=None, help="Only process the first N samples (per shard).")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate cells even if they already exist.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split samples across this many GPU workers.")
    parser.add_argument("--shard", type=int, default=0, help="Which shard this process handles (0-based).")
    args = parser.parse_args()
    if not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, {args.num_shards}); got {args.shard}")
    return args


def _cells_complete(cells_dir: Path, values) -> bool:
    return all((cells_dir / cell_filename(ts, te)).exists() for ts in values for te in values)


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

    # Ship every save folder with a sample_id -> prompts lookup table. Only shard 0
    # writes it (all shards would otherwise race on the same full-dataset file).
    if args.shard == 0:
        dest = write_id_to_prompts(output_root, data_root, mapping_path)
        LOGGER.info("Wrote %s", dest)
        dest = write_id_to_inputs(output_root, data_root, mapping_path)
        LOGGER.info("Wrote %s", dest)

    values = settings.grid_values(args.chord_edit_mode)
    base_config = settings.base_edit_config(args.chord_edit_mode)
    settings.require_factorizable_config(base_config)
    t_delta = settings.T_DELTA

    # Load the samples for this shard.
    samples = load_samples(mapping_path, args.max_samples, args.shard, args.num_shards)
    LOGGER.info("Data root: %s (%d sample(s))", data_root, len(samples))
    if args.num_shards > 1:
        LOGGER.info("Shard %d/%d", args.shard, args.num_shards)
    LOGGER.info("Grid: %dx%d, t_delta=%.2f -> %s", len(values), len(values), t_delta, output_root)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = load_pipeline(args.model_root, device, args.chord_edit_mode, base_config)

    # Generate the grid for each sample.
    for index, (sample_id, meta) in enumerate(samples, start=1):
        cells_dir = output_root / sample_id / "cells"
        if not args.overwrite and _cells_complete(cells_dir, values):
            LOGGER.info("[%d/%d] %s already generated; skipping.", index, len(samples), sample_id)
            continue

        # Extract the sample metadata.
        image_path = resolve_under(data_root, meta[settings.FIELD_IMAGE_PATH])
        category = Path(meta[settings.FIELD_IMAGE_PATH]).parent.name
        source_prompt = strip_brackets(meta.get(settings.FIELD_SOURCE_PROMPT, ""))
        target_prompt = strip_brackets(meta.get(settings.FIELD_TARGET_PROMPT, ""))
        LOGGER.info("[%d/%d] %s (%s): %r", index, len(samples), sample_id, category, target_prompt)

        with Image.open(image_path) as img:
            source_image = img.convert("RGB")

        # Create local record, will be used to generate the grid.
        record = LocalRecord(
            sample_name=category,
            image_path=image_path,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_prompt=strip_brackets(meta.get(settings.FIELD_EDIT_INSTRUCTION, "")),
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

        # Save the cells.
        ensure_dir(cells_dir)
        for t_end in values:
            for t_start in values:
                cell = cells[(t_start, t_end)].convert("RGB").resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
                cell.save(cells_dir / cell_filename(t_start, t_end), quality=settings.JPEG_QUALITY)

        # Save the clean grid.
        title = (f"{sample_id} ({category})\n" f'Source: "{source_prompt}"\n' f'Target: "{target_prompt}"')
        save_clean_grid(cells_dir, output_root / sample_id / "grid_clean.png", values, t_delta, title)
        LOGGER.info("[%d/%d] Generated %s (%d cells)", index, len(samples), sample_id, len(values) ** 2)

    LOGGER.info("Done. Images in %s", output_root)


if __name__ == "__main__":
    main()
