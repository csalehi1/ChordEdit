"""
Factorized ChordEdit t_start by t_end grid ablation.

Reuses shared encode/transport work across cells (requires n_steps=1).
See section comments below for the cost model vs run_grid_ablation.py.
"""

from __future__ import annotations

# Allow imports from scripts/ and the project root when invoked as a file path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))        # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

# Grid layout helpers, sweep constants, and shared CLI live in run_grid_ablation.py.
# I/O and pipeline wiring live in run_local_ablation.py. Symmetric defaults from run_pie_bench.py.
from daniel_create_grid_image import _build_base_grid, _save_grid_figure
from daniel_pipeline_chord import _cleanup_decode_row, _u_estimate
from run_grid_ablation import (
    GRID_VALUES,
    GRID_VALUES_SYM,
    build_argument_parser,
    value_slug,
)
from run_local_ablation import (
    LocalRecord,
    dtype_from_precision,
    ensure_dir,
    load_local_records,
    param_slug,
    resolve_component_paths,
    write_json,
)
from run_pie_bench import DEFAULT_EDIT_CONFIG_SYM

if TYPE_CHECKING:
    from PIL import Image
    from pipeline_chord import ChordEditPipeline


LOGGER = logging.getLogger("daniel_grid_ablation")

# Copied from run_local_ablation.py (app.py defaults).
DEFAULT_EDIT_CONFIG: Dict[str, Any] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.0,
    "step_scale": 1.0,
    "cleanup": True,
}

DEFAULT_OUTPUT_ROOT = "ablation_outputs/daniel_grid_t_start_t_end"
JPEG_EXTENSION = ".jpg"
JPEG_QUALITY = 75
GRID_BASENAME = "grid_clean"  # grid type 1 from daniel_create_grid_image.py


def _save_jpeg(image: Image.Image, destination: Path) -> None:
    ensure_dir(destination.parent)
    image.convert("RGB").save(destination, quality=JPEG_QUALITY)


def _cell_filename(t_start: float, t_end: float) -> str:
    return f"{param_slug('t_start', t_start)}__{param_slug('t_end', t_end)}{JPEG_EXTENSION}"


def parse_args():
    return build_argument_parser(
        description=(
            "Factorized ChordEdit t_start by t_end grid ablation. "
            "Reuses shared encode/transport work across cells when n_steps=1."
        ),
        output_root_default=DEFAULT_OUTPUT_ROOT,
    ).parse_args()


# run_grid_ablation.py always uses DEFAULT_EDIT_CONFIG and GRID_VALUES even when
# --chord-edit-mode sym is passed. Symmetric edits use a different u_estimate
# path and a smaller sweep range, so we switch both config and grid values here.
def base_edit_config(chord_edit_mode: str) -> Dict[str, Any]:
    if chord_edit_mode == "sym":
        return dict(DEFAULT_EDIT_CONFIG_SYM)
    return dict(DEFAULT_EDIT_CONFIG)


def grid_values(chord_edit_mode: str) -> List[float]:
    if chord_edit_mode == "sym":
        return list(GRID_VALUES_SYM)
    return list(GRID_VALUES)


def require_factorizable_config(base_config: Dict[str, Any]) -> None:
    n_steps = int(base_config.get("n_steps", 1))
    if n_steps != 1:
        raise ValueError(f"Factorized grid requires n_steps=1, got n_steps={n_steps}")


# run_grid_ablation.py calls pipeline.__call__ for every (t_start, t_end) cell.
# Each call repeats VAE encode, prompt encode, noise draw, and transport even
# when only t_end changes.
#
# For n_steps=1, pipeline._run_edit is equivalent to:
#   1. transport: x <- x_src + step_scale * u_estimate(x_src, t_start, t_delta)
#   2. cleanup:   x <- pred_x0(x, t_end)                     [if cleanup=True]
#   3. decode:    image <- VAE_decode(x)
#
# t_start only enters step 1; t_end only enters step 2. We therefore:
#   - run step 0 (encode / prompts / noise) once per image
#   - run step 1 once per unique t_start  (N times on a NxN grid)
#   - run steps 2-3 per cell              (N^2 times)
#
# Compared to run_grid_ablation.py on a 5x5 grid: 25 full pipelines become
# 1 encode + 5 transports + 25 cleanups + 25 decodes. Transport is the expensive
# part (each _u_estimate issues 4 batched UNet forwards).
#
# NOTE: t=0 shortcut (NOT applied): at t_start=0 or t_end=0 the diffusion noising term
# is identity (sigma->0, z_t ~= x), but u_estimate and pred_x0 still run the UNet
# and return non-zero updates. Verified on sd-turbo/images/001 with t_delta=0:
#   default: skip transport@t_start=0 is ~0.01 mean pixel error (not exact);
#            skip cleanup@t_end=0 is wrong (e.g. mean|diff|~1.3 at (0.9,0.0))
#   sym:     transport@t_start=0 changes latents by max|diff|~8.4; skip is very wrong
# Full transport + full cleanup matches pipeline.__call__ exactly for all tested cells.
def run_factorized_grid(
    *,
    pipeline: "ChordEditPipeline",
    source_image: "Image.Image",
    record: LocalRecord,
    base_config: Dict[str, Any],
    t_start_values: List[float],
    t_end_values: List[float],
    t_delta: float,
    seed: int,
) -> Dict[Tuple[float, float], "Image.Image"]:
    import torch

    with torch.no_grad():
        cfg = dict(base_config)
        cfg["t_delta"] = t_delta
        shared_params = pipeline._prepare_edit_params(cfg)

        # Shared setup (repeats every cell in run_grid_ablation.py)
        pixel_values = pipeline._prepare_image_tensor(source_image)
        latents = pipeline._encode_image_to_latent(pixel_values)
        src_embed = pipeline.encode_prompt([record.source_prompt])
        tgt_embed = pipeline.encode_prompt([record.target_prompt])
        noise_list = pipeline._prepare_noise_list(
            latents=latents,
            seed_value=seed,
            num_noises=shared_params["noise_samples"],
        )

        # Transport per unique t_start
        # Re-prepare params per t_start so t_delta clamping matches pipeline.__call__
        transport: Dict[float, torch.Tensor] = {}
        for t_start in t_start_values:
            cell_cfg = dict(base_config)
            cell_cfg["t_start"] = t_start
            cell_cfg["t_delta"] = t_delta
            params = pipeline._prepare_edit_params(cell_cfg)
            # _u_estimate picks the exact delta=0 fast path when applicable
            # (see daniel_pipeline_chord._u_estimate / _u_estimate_delta0).
            u_hat = _u_estimate(
                pipeline,
                latents,
                src_embed,
                tgt_embed,
                noise_list,
                params["t_start"],
                params["t_delta"],
            )
            transport[t_start] = (latents + params["step_scale"] * u_hat).detach()

        # Cleanup + decode per (t_start, t_end) cell.
        # For a fixed t_start only t_end (the cleanup timestep) changes, so we
        # batch the whole row into one _pred_x0 forward and one VAE decode
        # instead of grid single-item calls (see _cleanup_decode_row).
        results: Dict[Tuple[float, float], "Image.Image"] = {}
        for t_start in t_start_values:
            row_images = _cleanup_decode_row(
                pipeline,
                transport[t_start],
                tgt_embed,
                noise_list[0],
                t_end_values,
                bool(shared_params["cleanup"]),
            )
            for t_end, image in zip(t_end_values, row_images):
                results[(t_start, t_end)] = image

    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import torch
    from PIL import Image

    from pipeline_chord import ChordEditPipeline

    data_root = Path(args.data_root).expanduser().resolve()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser().resolve()
    component_paths = resolve_component_paths(args.model_root, args.model_type)
    base_config = base_edit_config(args.chord_edit_mode)
    require_factorizable_config(base_config)
    records = load_local_records(data_root, args.max_records)
    values = grid_values(args.chord_edit_mode)
    delta_values = [0.0]
    grid_size = len(values)
    cells_per_image = grid_size * grid_size

    LOGGER.info("Loaded %d record(s) from %s", len(records), data_root)
    LOGGER.info("Base edit config: %s", base_config)
    LOGGER.info("Grid values: %s (%dx%d = %d cells/image/t_delta)", values, grid_size, grid_size, cells_per_image)
    LOGGER.info("t_delta conditions: %s", delta_values)
    LOGGER.info("Chord edit mode: %s", args.chord_edit_mode)
    LOGGER.info(
        "Per image/t_delta: 1 encode, 1 prompt encode, 1 noise setup, "
        "%d transports, %d cleanups, %d decodes "
        "(run_grid_ablation.py would run %d full pipelines)",
        grid_size,
        cells_per_image,
        cells_per_image,
        cells_per_image,
    )
    LOGGER.info("Output root: %s", output_root)

    # Load the pipeline once; all cells for all records reuse this instance.
    torch_dtype = dtype_from_precision(args.precision)
    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        model_type=None if args.model_type == "auto" else args.model_type,
        default_edit_config=base_config,
        device=args.device,
        torch_dtype=torch_dtype,
        image_size=args.image_size,
        use_center_crop=args.center_crop,
        compute_dtype=torch.float32,
        use_attention_mask=args.use_attention_mask,
        use_safety_checker=args.safety_checker,
        chord_edit_mode=args.chord_edit_mode,
    )

    manifest: Dict[str, Any] = {
        "data_root": str(data_root),
        "model_root": str(Path(args.model_root).expanduser().resolve()),
        "run_timestamp": run_timestamp,
        "base_edit_config": base_config,
        "seed": args.seed,
        "chord_edit_mode": args.chord_edit_mode,
        "execution_mode": "factorized",
        "x_axis": {"name": "t_start", "values": values},
        "y_axis": {"name": "t_end", "values": values},
        "t_delta_values": delta_values,
        "records": [],
    }

    # Per record: sweep t_delta conditions, build NxN grids.
    for record_index, record in enumerate(records, start=1):
        record_key = f"{record.sample_name}_{record.edit_id}"
        sample_dir = output_root / record_key
        ensure_dir(sample_dir)

        LOGGER.info("Processing %d/%d: %s", record_index, len(records), record_key)
        with Image.open(record.image_path) as img:
            source_image = img.convert("RGB")
        source_path = sample_dir / f"source{JPEG_EXTENSION}"
        if not source_path.exists() or args.overwrite:
            _save_jpeg(source_image, source_path)

        record_payload = {
            "sample": record.sample_name,
            "edit_id": record.edit_id,
            "image_path": str(record.image_path),
            "source_prompt": record.source_prompt,
            "target_prompt": record.target_prompt,
            "edit_prompt": record.edit_prompt,
            "t_delta_results": [],
        }

        # Iterate over the t_delta conditions.
        for t_delta in delta_values:
            condition_name = f"t_delta_{value_slug(t_delta)}"
            condition_dir = sample_dir / condition_name
            cells_dir = condition_dir / "cells"
            ensure_dir(cells_dir)

            # Generate all missing cells for this t_delta in one shot.
            # If every cell JPEG already exists (and --overwrite is off), skip inference.
            factorized_images: Dict[Tuple[float, float], Image.Image] | None = None
            missing_cells = False
            for t_start in values:
                for t_end in values:
                    filename = _cell_filename(t_start, t_end)
                    if not (cells_dir / filename).exists() or args.overwrite:
                        missing_cells = True
                        break
                if missing_cells:
                    break
            if missing_cells:
                factorized_images = run_factorized_grid(
                    pipeline=pipeline,
                    source_image=source_image,
                    record=record,
                    base_config=base_config,
                    t_start_values=values,
                    t_end_values=values,
                    t_delta=t_delta,
                    seed=args.seed,
                )

            cell_outputs: List[Dict[str, Any]] = []
            for t_end in values:
                for t_start in values:
                    filename = _cell_filename(t_start, t_end)
                    out_path = cells_dir / filename
                    if out_path.exists() and not args.overwrite:
                        with Image.open(out_path) as cached:
                            generated = cached.convert("RGB")
                    else:
                        assert factorized_images is not None
                        generated = factorized_images[(t_start, t_end)]
                        _save_jpeg(generated, out_path)
                    
                    # Append the cell output for record payload.
                    cell_outputs.append(
                        {
                            "t_start": t_start,
                            "t_end": t_end,
                            "t_delta": t_delta,
                            "path": str(out_path),
                        }
                    )
                
            # grid_clean: labeled composite, no metric overlay (daniel_create_grid_image type 1).
            grid_path = condition_dir / f"{GRID_BASENAME}{JPEG_EXTENSION}"
            built = _build_base_grid(cells_dir, values, values, cell_extension=JPEG_EXTENSION)
            if built is not None:
                base_canvas, _ = built
                _save_grid_figure(
                    base_canvas,
                    grid_path,
                    title=(
                        f"{record_key}\n"
                        f'Source Prompt: "{record.source_prompt}"\n'
                        f'Target Prompt: "{record.target_prompt}"'
                    ),
                    values_start=values,
                    values_end=values,
                    t_delta=t_delta,
                )
            else:
                LOGGER.warning("No cell images found for %s; skipping %s", condition_dir, GRID_BASENAME)

            # Append the results to the record payload
            record_payload["t_delta_results"].append(
                {
                    "t_delta": t_delta,
                    "directory": str(condition_dir),
                    "grid": str(grid_path),
                    "cells": cell_outputs,
                }
            )

        write_json(sample_dir / "meta.json", record_payload)
        manifest["records"].append(record_payload)

    ensure_dir(output_root)
    write_json(output_root / "manifest.json", manifest)
    LOGGER.info("Finished grid ablations. Results saved in %s", output_root)


if __name__ == "__main__":
    main()
