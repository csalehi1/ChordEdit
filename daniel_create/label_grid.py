"""Score an already-generated cell grid with whole-image PSNR + masked CLIP.

Always reads/writes under daniel_create/generated/<Path(data_root).name>/
(same directory as generate_grid.py). Already-scored samples are skipped (resumable).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np

import settings
from common import (
    cell_filename,
    load_mask,
    load_samples,
    resolve_under,
    strip_brackets,
    write_id_to_metrics,
)
from render_grid import save_metric_grids

LOGGER = logging.getLogger("label_grid")
SCRIPT_DIR = Path(__file__).resolve().parent


class InlineMetrics:
    """PnPInversion PSNR + CLIP edit-part scores, inlined for the chordedit env."""

    def __init__(self, device: str, clip_model_id: str = settings.CLIP_MODEL_ID):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    def psnr_batch(self, source_image, edited_images) -> list[float]:
        """Whole-image PSNR (data_range=1.0) for many edits vs one source."""
        torch = self._torch
        source = torch.tensor(np.array(source_image).astype(np.float32) / 255).permute(2, 0, 1).to(self.device)
        preds = torch.stack(
            [torch.tensor(np.array(im).astype(np.float32) / 255).permute(2, 0, 1) for im in edited_images]
        ).to(self.device)
        mse = ((preds - source.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3))
        return (-10.0 * torch.log10(mse)).cpu().tolist()

    def clip_batch(self, images, prompt: str, mask=None, batch_size: int = 64) -> list[float]:
        """100 * cosine similarity of (optionally masked) edit region vs prompt."""
        from PIL import Image

        torch = self._torch
        masked_images = []
        for image in images:
            arr = np.array(image)
            if mask is not None:
                arr = np.uint8(arr * mask)
            masked_images.append(Image.fromarray(arr))

        scores: list[float] = []
        for start in range(0, len(masked_images), batch_size):
            chunk = masked_images[start : start + batch_size]
            inputs = self.clip_processor(
                text=[prompt], images=chunk, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.clip_model(**inputs)
            image_embeds = out.image_embeds / out.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = out.text_embeds / out.text_embeds.norm(p=2, dim=-1, keepdim=True)
            scores.extend((100 * (image_embeds * text_embeds).sum(dim=-1)).cpu().tolist())
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=settings.DEFAULT_DATA_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grids", action="store_true", help="Also write grid_psnr.png / grid_clip.png")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="GPU ids; one shard per GPU")
    return parser.parse_args()


def run_shard(
    *,
    data_root: Path,
    output_root: Path,
    max_samples: int | None,
    write_grids: bool,
    shard: int,
    num_shards: int,
    gpu: int,
) -> None:
    """Score one round-robin shard of samples on a single GPU."""
    import torch
    from PIL import Image

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    mapping_path = data_root / "mapping_file.json"
    grid_values = list(settings.GRID_VALUES)
    t_delta = settings.T_DELTA

    # Per-shard CSV when sharded; single result.csv otherwise.
    csv_name = settings.CSV_NAME if num_shards == 1 else f"result_shard{shard:02d}.csv"
    csv_path = output_root / csv_name
    samples = load_samples(mapping_path, max_samples, shard, num_shards)

    LOGGER.info(
        "Shard %d/%d on cuda:%d | %d sample(s) -> %s",
        shard, num_shards, gpu, len(samples), csv_path.name,
    )

    device = f"cuda:{gpu}"
    metrics = InlineMetrics(device)

    # Resume against every result*.csv already on disk.
    done_ids = set()
    for existing in sorted(output_root.glob("result*.csv")):
        with existing.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("sample_id"):
                    done_ids.add(row["sample_id"])

    write_header = not csv_path.exists()
    csv_file = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=settings.CSV_FIELDS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    for index, (sample_id, meta) in enumerate(samples, start=1):
        if sample_id in done_ids:
            LOGGER.info("[%d/%d] %s already scored; skipping.", index, len(samples), sample_id)
            continue

        cells_dir = output_root / sample_id / "cells"
        cell_specs = []
        edited_images = []
        incomplete = False
        for t_end in grid_values:
            for t_start in grid_values:
                cell_path = cells_dir / cell_filename(t_start, t_end)
                if not cell_path.exists():
                    incomplete = True
                    break
                with Image.open(cell_path) as img:
                    edited_images.append(
                        img.convert("RGB").resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
                    )
                cell_specs.append((t_start, t_end))
            if incomplete:
                break
        if incomplete:
            LOGGER.warning("[%d/%d] %s cells missing; run generate_grid.py first.", index, len(samples), sample_id)
            continue

        image_path = resolve_under(data_root, meta[settings.FIELD_IMAGE_PATH])
        category = Path(meta[settings.FIELD_IMAGE_PATH]).parent.name
        source_prompt = strip_brackets(meta.get(settings.FIELD_SOURCE_PROMPT, ""))
        target_prompt = strip_brackets(meta.get(settings.FIELD_TARGET_PROMPT, ""))
        LOGGER.info("[%d/%d] %s (%s): %r", index, len(samples), sample_id, category, target_prompt)

        with Image.open(image_path) as img:
            source_image = img.convert("RGB").resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        edit_mask = load_mask(data_root, meta)

        whole_psnr_list = metrics.psnr_batch(source_image, edited_images)
        if edit_mask.sum() == 0:
            clip_edited_list: list = ["nan"] * len(edited_images)
        else:
            clip_edited_list = metrics.clip_batch(edited_images, target_prompt, edit_mask)

        padded_id = str(sample_id).zfill(settings.SAMPLE_ID_WIDTH)
        rows = []
        for (t_start, t_end), whole_psnr, clip_edited in zip(
            cell_specs, whole_psnr_list, clip_edited_list
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "t_start": t_start,
                    "t_end": t_end,
                    "t_delta": t_delta,
                    "whole_psnr": whole_psnr,
                    "clip_edited": clip_edited,
                    "cell_path": f"{padded_id}/cells/{cell_filename(t_start, t_end)}",
                }
            )
        writer.writerows(rows)
        csv_file.flush()

        if write_grids:
            title = f'{sample_id} ({category})\nSource: "{source_prompt}"\nTarget: "{target_prompt}"'
            psnr_scores = {}
            clip_scores = {}
            for row in rows:
                key = (round(row["t_start"], 1), round(row["t_end"], 1))
                if isinstance(row["whole_psnr"], (int, float)):
                    psnr_scores[key] = float(row["whole_psnr"])
                if isinstance(row["clip_edited"], (int, float)):
                    clip_scores[key] = float(row["clip_edited"])
            save_metric_grids(
                cells_dir,
                output_root / sample_id,
                grid_values,
                t_delta,
                title,
                {
                    "grid_psnr.png": ("Whole PSNR", psnr_scores),
                    "grid_clip.png": ("CLIP-Edited", clip_scores),
                },
            )

        LOGGER.info("[%d/%d] Scored %s (%d cells)", index, len(samples), sample_id, len(rows))

    csv_file.close()
    LOGGER.info("Shard %d/%d done -> %s", shard, num_shards, csv_path)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    gpus = args.gpus

    # Same folder generate_grid.py writes to: daniel_create/generated/<name>/
    output_root = SCRIPT_DIR / "generated" / Path(args.data_root).name
    output_root.mkdir(parents=True, exist_ok=True)
    generated_gitignore = output_root.parent / ".gitignore"
    if not generated_gitignore.exists():
        generated_gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

    kwargs = dict(
        data_root=data_root,
        output_root=output_root,
        max_samples=args.max_samples,
        write_grids=args.grids,
        num_shards=len(gpus),
    )

    if len(gpus) == 1:
        run_shard(shard=0, gpu=gpus[0], **kwargs)
    else:
        processes = [
            Process(target=run_shard, kwargs={**kwargs, "shard": shard, "gpu": gpu})
            for shard, gpu in enumerate(gpus)
        ]
        for process in processes:
            process.start()
            time.sleep(0.5)
        failed = False
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failed = True
        if failed:
            raise SystemExit("One or more label shards failed.")

        # Merge per-shard CSVs into result.csv.
        merged = output_root / settings.CSV_NAME
        shard_csvs = sorted(output_root.glob("result_shard*.csv"))
        with merged.open("w", encoding="utf-8", newline="") as out_handle:
            for index, shard_csv in enumerate(shard_csvs):
                with shard_csv.open("r", encoding="utf-8") as in_handle:
                    if index == 0:
                        out_handle.write(in_handle.read())
                    else:
                        # Skip header on subsequent shards.
                        next(in_handle, None)
                        out_handle.write(in_handle.read())
        LOGGER.info("Merged %d shard CSV(s) -> %s", len(shard_csvs), merged)

    dest = write_id_to_metrics(output_root)
    if dest is not None:
        LOGGER.info("Wrote %s", dest)
    LOGGER.info("Done. Results under %s", output_root)


if __name__ == "__main__":
    main()
