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
    iter_cell_pairs,
    load_pipeline,
    load_samples,
    resolve_under,
    strip_brackets,
    validate_dataset_root,
    write_id_to_embeddings,
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
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--embeddings-root", default=None)
    parser.add_argument("--generated-root", default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--max-samples", type=int, default=None)
    # Optional flags: --add-plots, --cache-masks, --skip-embeddings, --skip-generated, --diagonal-optimization.
    parser.add_argument("--add-plots", action="store_true")
    parser.add_argument("--cache-masks", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-generated", action="store_true")
    parser.add_argument("--diagonal-optimization", action="store_true")
    return parser.parse_args()


def run_shard(
    *,
    data_root: Path,
    embeddings_root: Path,
    generated_root: Path,
    model_root: str,
    max_samples: int | None,
    write_plots: bool,
    diagonal_optimization: bool,
    cache_masks: bool,
    skip_embeddings: bool,
    skip_generated: bool,
    num_shards: int,
    shard: int,
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

    mapping_path = validate_dataset_root(data_root)

    # Only shard 0 writes the full-dataset embeddings/inputs CSVs (avoids races).
    if shard == 0:
        if not skip_embeddings:
            emb_dest = write_id_to_embeddings(embeddings_root, mapping_path, cache_masks=cache_masks)
            LOGGER.info("Wrote %s", emb_dest)
        if not skip_generated:
            dest = write_id_to_inputs(generated_root, data_root, mapping_path)
            LOGGER.info("Wrote %s", dest)

    grid_values = list(settings.GRID_VALUES)
    base_config = dict(DEFAULT_EDIT_CONFIG)
    if int(base_config.get("n_steps", 1)) != 1:
        # Optimization only works with n_steps=1.
        raise ValueError(f"Factorized grid needs n_steps=1, got {base_config['n_steps']}")
    t_delta = settings.T_DELTA

    samples = load_samples(mapping_path, max_samples, shard, num_shards)
    LOGGER.info(
        "GPU %d: Started shard %d/%d on GPU %d (cuda:0) | %d sample" + "s"*(len(samples) != 1),
        gpu, shard + 1, num_shards, gpu, len(samples),
    )

    # After CUDA_VISIBLE_DEVICES pinning, the only visible device is cuda:0.
    bind_pipeline(load_pipeline(model_root, "cuda:0", base_config, SD_COMPONENT_SUBDIRS))

    # Select (t_start, t_end) pairs, including filters for diagonal optimization
    # and t_start - t_delta >= 0 (invalid when the delta window would go negative).
    cell_pairs = list(
        iter_cell_pairs(grid_values, diagonal_optimization=diagonal_optimization, t_delta=t_delta)
    )

    # Generate embeddings and/or cells for each sample.
    for index, (sample_id, meta) in enumerate(samples, start=1):
        emb_dir = embeddings_root / settings.SAMPLES_DIRNAME / sample_id
        sample_dir = generated_root / settings.GRIDS_DIRNAME / sample_id
        cells_dir = sample_dir / settings.CELLS_DIRNAME
        mask_rel = meta.get(settings.FIELD_MASK_IMAGE_PATH, "") if cache_masks else ""

        # Determine if sample embeddings are already complete (i.e., partially generated).
        need_embeddings = False
        if not skip_embeddings:
            emb_paths = [emb_dir / name for name in settings.EMBEDDING_FILENAMES]
            # Masks are optional per sample; when caching them, an existing trio without mask.pt is incomplete.
            if mask_rel:
                emb_paths.append(emb_dir / settings.MASK_FILENAME)
            need_embeddings = not all(path.exists() for path in emb_paths)

        # Determine if sample cells are already complete (i.e., partially generated).
        need_cells = False
        if not skip_generated:
            cell_paths = [cells_dir / cell_filename(ts, te) for ts, te in cell_pairs]
            need_cells = not cell_paths or not all(path.exists() for path in cell_paths)

        if not need_embeddings and not need_cells:
            LOGGER.info("GPU %d: [%d/%d] Skipped %s, already generated.", gpu, index, len(samples), sample_id)
            continue

        sample_start = time.perf_counter()

        image_path = resolve_under(data_root, meta[settings.FIELD_IMAGE_PATH])
        category = Path(meta[settings.FIELD_IMAGE_PATH]).parent.name
        # TODO: Bracket stripping may not be necessary. Investigate.
        source_prompt = strip_brackets(meta.get(settings.FIELD_SOURCE_PROMPT, ""))
        target_prompt = strip_brackets(meta.get(settings.FIELD_TARGET_PROMPT, ""))
        edit_instruction = strip_brackets(meta.get(settings.FIELD_EDIT_INSTRUCTION, ""))
        mask_path = resolve_under(data_root, mask_rel) if (mask_rel and need_embeddings) else None

        with Image.open(image_path) as img:
            source_image = img.convert("RGB")
        mask_image = None
        if mask_path is not None:
            with Image.open(mask_path) as mask_img:
                # Masks are expected to be in RGB, not L, format for encoding.
                mask_image = mask_img.convert("RGB")

        # Bundle this sample's metadata for the grid pipeline.
        record = SampleRecord(
            sample_name=category,
            image_path=image_path,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_instruction=edit_instruction,
            sample_id=sample_id,
        )

        encode_only = skip_generated or not need_cells
        # Generate cells if they are needed, otherwise only encode the embeddings.
        cells = run_factorized_grid(
            source_image=source_image,
            record=record,
            base_config=base_config,
            cell_pairs=cell_pairs,
            t_delta=t_delta,
            seed=SEED,
            embeddings_dir=emb_dir if need_embeddings else None,
            mask_image=mask_image,
            skip_generated=encode_only,
        )

        if encode_only:
            elapsed = time.perf_counter() - sample_start
            LOGGER.info("GPU %d: [%d/%d] Saved embeddings for %s (%.2fs)", gpu, index, len(samples), sample_id, elapsed)
            continue

        cells_dir.mkdir(parents=True, exist_ok=True)
        for t_start, t_end in cell_pairs:
            # Resize the cell to the desired image size and save it.
            cell = cells[(t_start, t_end)].convert("RGB").resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            cell_path = cells_dir / cell_filename(t_start, t_end)
            if settings.JPEG_QUALITY is None:
                # Save as lossless .png
                cell.save(cell_path)
            else:
                # Save as .jpg at the specified quality or will save .png without compression.
                cell.save(cell_path, quality=settings.JPEG_QUALITY)

        if write_plots:
            # Write a grid overview image for each sample.
            title = f'{sample_id} ({category})\nSource: "{source_prompt}"\nTarget: "{target_prompt}"'
            save_clean_grid(cells_dir, sample_dir / "grid_clean.png", grid_values, t_delta, title)

        elapsed = time.perf_counter() - sample_start
        LOGGER.info(
            "GPU %d: [%d/%d] Generated %s (%d cells in %.2fs)",
            gpu, index, len(samples), sample_id, len(cell_pairs), elapsed,
        )

    LOGGER.info("GPU %d: Finished shard %d/%d.", gpu, shard + 1, num_shards)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    args = parse_args()
    if args.skip_generated and args.add_plots:
        raise SystemExit("--skip-generated cannot be combined with --add-plots")
    if args.skip_embeddings and args.skip_generated:
        raise SystemExit("--skip-embeddings cannot be combined with --skip-generated")
    if args.cache_masks and args.skip_embeddings:
        raise SystemExit("--cache-masks cannot be combined with --skip-embeddings")
    data_root = Path(args.data_root).expanduser().resolve()
    validate_dataset_root(data_root)
    gpus = args.gpus

    # Create embeddings and generated dirs under <root>/<dataset-folder-name>[_n<max-samples>]/.
    dataset_dir = data_root.name if args.max_samples is None else f"{data_root.name}_n{args.max_samples}"
    embeddings_root = (
        Path(args.embeddings_root).expanduser().resolve() / dataset_dir
        if args.embeddings_root is not None
        else SCRIPT_DIR / "embeddings" / dataset_dir
    )
    generated_root = (
        Path(args.generated_root).expanduser().resolve() / dataset_dir
        if args.generated_root is not None
        else SCRIPT_DIR / "generated" / dataset_dir
    )
    if not args.skip_embeddings:
        embeddings_root.mkdir(parents=True, exist_ok=True)
        if args.embeddings_root is None:
            embeddings_gitignore = embeddings_root.parent / ".gitignore"
            if not embeddings_gitignore.exists():
                embeddings_gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")
    if not args.skip_generated:
        generated_root.mkdir(parents=True, exist_ok=True)
        if args.generated_root is None:
            generated_gitignore = generated_root.parent / ".gitignore"
            if not generated_gitignore.exists():
                generated_gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

    kwargs = dict(
        data_root=data_root,
        embeddings_root=embeddings_root,
        generated_root=generated_root,
        model_root=args.model_root,
        max_samples=args.max_samples,
        write_plots=args.add_plots,
        diagonal_optimization=args.diagonal_optimization,
        cache_masks=args.cache_masks,
        skip_embeddings=args.skip_embeddings,
        skip_generated=args.skip_generated,
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
            # Stagger by a second so shard 0 can write id_to_embeddings first.
            time.sleep(1.0)
        failed = False
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failed = True
        if failed:
            raise SystemExit("One or more generate shards failed.")

    LOGGER.info("Done.")
    if not args.skip_embeddings:
        LOGGER.info("Embeddings in %s", embeddings_root)
    if not args.skip_generated:
        LOGGER.info("Images in %s", generated_root)


if __name__ == "__main__":
    main()
