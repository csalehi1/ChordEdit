"""
Generate a t_start by t_end cell grid for every sample in a dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict
import settings
from _helpers import (
    SampleRecord,
    cell_filename,
    get_output_dir_name,
    iter_cell_pairs,
    load_pipeline,
    load_samples,
    resolve_under,
    strip_brackets,
    write_id_to_inputs,
)

LOGGER = logging.getLogger("grid_generate")
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_ROOT = "/shared/ssd_30T/mirick/models/sd-turbo"
SEED = 42

# n_steps must stay 1 so transport (t_start) and cleanup (t_end) factorize.
DEFAULT_EDIT_CONFIG: Dict[str, Any] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.0,
    "step_scale": 1.0,
    "cleanup": True,
}

# SD component folders under --model-root.
SD_COMPONENT_SUBDIRS = {
    "unet_path": "unet",
    "scheduler_path": "scheduler",
    "text_encoder_path": "text_encoder",
    "tokenizer_path": "tokenizer",
    "vae_path": "vae",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Parent output directory; results are saved under <dataset-folder-name>",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grids", action="store_true", help="Also write grid_clean.png overviews")
    parser.add_argument(
        "--diagonal-optimization",
        action="store_true",
        help="Only generate cells where t_start > t_end",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="GPU ids; one shard per GPU")
    return parser.parse_args()


def run_shard(
    *,
    data_root: Path,
    output_root: Path,
    model_root: str,
    max_samples: int | None,
    write_grids: bool,
    diagonal_optimization: bool,
    shard: int,
    num_shards: int,
    gpu: int,
) -> None:
    """Generate one round-robin shard of samples on a single GPU."""
    # spawn starts a fresh interpreter: reconfigure logging here (parent config is not inherited).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)

    # Pin this process to one physical GPU BEFORE importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch
    from PIL import Image

    from _pipeline import bind_pipeline, run_factorized_grid
    from grid_render import save_clean_grid

    # Enable TF32 for faster GPU inference.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    mapping_path = data_root / "mapping_file.json"

    # Only shard 0 writes the full-dataset inputs CSV (avoids races).
    if shard == 0:
        dest = write_id_to_inputs(output_root, data_root, mapping_path)
        LOGGER.info("Wrote %s", dest)

    grid_values = list(settings.GRID_VALUES)
    base_config = dict(DEFAULT_EDIT_CONFIG)
    if int(base_config.get("n_steps", 1)) != 1:
        # Optimization only works with n_steps=1.
        raise ValueError(f"Factorized grid needs n_steps=1, got {base_config['n_steps']}")
    t_delta = settings.T_DELTA

    samples = load_samples(mapping_path, max_samples, shard, num_shards)
    LOGGER.info(
        "GPU %d: Shard %d/%d on physical GPU %d (cuda:0) | %d sample(s)",
        shard, shard + 1, num_shards, gpu, len(samples),
    )

    # After CUDA_VISIBLE_DEVICES pinning, the only visible device is cuda:0.
    bind_pipeline(load_pipeline(model_root, "cuda:0", base_config, SD_COMPONENT_SUBDIRS))

    # Generate cells for each sample.
    for index, (sample_id, meta) in enumerate(samples, start=1):
        cells_dir = output_root / sample_id / "cells"

        # Skip if every cell jpg is already on disk.
        expected = [
            cells_dir / cell_filename(ts, te)
            for ts, te in iter_cell_pairs(grid_values, diagonal_optimization=diagonal_optimization)
        ]
        if expected and all(path.exists() for path in expected):
            LOGGER.info("GPU %d: [%d/%d] %s already generated, skipping.", gpu, index, len(samples), sample_id)
            continue

        sample_start = time.perf_counter()

        image_path = resolve_under(data_root, meta[settings.FIELD_IMAGE_PATH])
        category = Path(meta[settings.FIELD_IMAGE_PATH]).parent.name
        # TODO: Bracket stripping may not be necessary. Investigate.
        source_prompt = strip_brackets(meta.get(settings.FIELD_SOURCE_PROMPT, ""))
        target_prompt = strip_brackets(meta.get(settings.FIELD_TARGET_PROMPT, ""))

        with Image.open(image_path) as img:
            source_image = img.convert("RGB")

        # Bundle this sample's metadata for the grid pipeline.
        record = SampleRecord(
            sample_name=category,
            image_path=image_path,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_prompt=strip_brackets(meta.get(settings.FIELD_EDIT_INSTRUCTION, "")),
            sample_id=sample_id,
        )

        cells = run_factorized_grid(
            source_image=source_image,
            record=record,
            base_config=base_config,
            t_start_values=grid_values,
            t_end_values=grid_values,
            t_delta=t_delta,
            seed=SEED,
            diagonal_optimization=diagonal_optimization,
        )

        cells_dir.mkdir(parents=True, exist_ok=True)
        for t_start, t_end in iter_cell_pairs(grid_values, diagonal_optimization=diagonal_optimization):
            # Resize the cell to the desired image size and save it.
            cell = cells[(t_start, t_end)].convert("RGB").resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            cell_path = cells_dir / cell_filename(t_start, t_end)
            if settings.JPEG_QUALITY is None:
                # Save as lossless .png
                cell.save(cell_path)
            else:
                # Save as .jpg at the specified quality or will save .png without compression.
                cell.save(cell_path, quality=settings.JPEG_QUALITY)

        if write_grids:
            # Write a grid overview image for each sample.
            title = f'{sample_id} ({category})\nSource: "{source_prompt}"\nTarget: "{target_prompt}"'
            save_clean_grid(cells_dir, output_root / sample_id / "grid_clean.png", grid_values, t_delta, title)

        cell_count = sum(1 for _ in iter_cell_pairs(grid_values, diagonal_optimization=diagonal_optimization))
        elapsed = time.perf_counter() - sample_start
        LOGGER.info(
            "GPU %d: [%d/%d] Generated %s (%d cells in %.2fs)",
            gpu, index, len(samples), sample_id, cell_count, elapsed,
        )

    LOGGER.info("GPU %d: Shard %d/%d done.", gpu, shard + 1, num_shards)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    gpus = args.gpus

    # Create the output root dir, <output-root>/<dataset-folder-name>[_n<max-samples>]/.
    output_root = (
        Path(args.output_root).expanduser().resolve() / get_output_dir_name(data_root.name, args.max_samples)
        if args.output_root is not None
        else SCRIPT_DIR / "generated" / get_output_dir_name(data_root.name, args.max_samples)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if args.output_root is None:
        generated_gitignore = output_root.parent / ".gitignore"
        if not generated_gitignore.exists():
            generated_gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

    kwargs = dict(
        data_root=data_root,
        output_root=output_root,
        model_root=args.model_root,
        max_samples=args.max_samples,
        write_grids=args.grids,
        diagonal_optimization=args.diagonal_optimization,
        num_shards=len(gpus),
    )

    # One process per GPU, each taking a round-robin slice of samples.
    # Multiprocessing uses "spawn" (not fork) so each child gets a fresh Python interpreter:
    # CUDA/PyTorch are not fork-safe, and run_shard sets CUDA_VISIBLE_DEVICES before import.
    if len(gpus) == 1:
        run_shard(shard=0, gpu=gpus[0], **kwargs)
    else:
        ctx = get_context("spawn")
        processes = [
            ctx.Process(target=run_shard, kwargs={**kwargs, "shard": shard, "gpu": gpu})
            for shard, gpu in enumerate(gpus)
        ]
        for process in processes:
            process.start()
            # Stagger by a second so shard 0 can write id_to_inputs first.
            time.sleep(1.0)
        failed = False
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failed = True
        if failed:
            raise SystemExit("One or more generate shards failed.")

    LOGGER.info("Done. Images in %s", output_root)


if __name__ == "__main__":
    main()
