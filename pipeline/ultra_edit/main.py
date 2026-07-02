#!/usr/bin/env python3
"""
Download the first N samples from the UltraEdit Region-Based 100k dataset and
lay them out like PIE-Bench_v1.

Source dataset:
  https://huggingface.co/datasets/BleachNick/UltraEdit_Region_Based_100k

Only these fields are used per sample:
  source_caption, target_caption, edit_prompt, source_image, mask_image

Data is read with the Hugging Face `datasets` library in streaming mode, which
pulls the underlying Parquet shards directly from the hub CDN. The images are
embedded in the Parquet rows, so there are no per-image HTTP requests and no
datasets-server rate limiting (HTTP 429). Only the shards needed for the first
N rows are downloaded.

Output layout (PIE-Bench_v1 style, flat). The default output folder is
<script_dir>/datasets/UltraEdit_Region_<n_samples>, i.e. downloaded folders are
saved inside datasets/ (override with --out-root). A single .gitignore is
written to datasets/ (not to each download folder) so every downloaded folder
is ignored by git:
  datasets/
    .gitignore              ignores everything in datasets/ (keeps itself)
    <out_root>/
      annotation_images/
        0000000000.jpg      source_image
        ...
      annotation_masks/
        0000000000.jpg      mask_image
        ...
      mapping_file.json

Each sample is keyed by a zero-padded 10-digit id that counts up
(0000000000, 0000000001, ...), and each mapping_file.json entry follows the
PIE-Bench_v1 style:
  original_prompt      the source_caption, with changed spans wrapped in [brackets]
  editing_prompt       the target_caption, with changed spans wrapped in [brackets]
  editing_instruction  the edit_prompt instruction, e.g. "Replace the cows with flamingos"
  image_path           source image path, e.g. annotation_images/0000000000.jpg
  mask_image_path      mask image path,   e.g. annotation_masks/0000000000.jpg

With --resume the script skips the rows it has already saved and continues from
where it left off, so no sample is ever downloaded twice.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


DATASET = "BleachNick/UltraEdit_Region_Based_100k"
CONFIG = "default"
SPLIT = "RegionBase"
# Downloaded folders are saved under ./datasets/.
DEFAULT_BASE_DIR = str(Path(__file__).resolve().parent / "datasets")
NEEDED_COLUMNS = ["source_image", "mask_image", "source_caption", "target_caption", "edit_prompt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", dest="n_samples", type=int, required=True, help="Number of samples to download (the first N rows of the split).")
    parser.add_argument("--out-root", default=None, help="Output folder to create (PIE-Bench-style layout). Defaults to <script_dir>/datasets/UltraEdit_Region_<n_samples> (inside pipeline_ultraedit/datasets/).")
    parser.add_argument("--dataset", default=DATASET, help="Hugging Face dataset id.")
    parser.add_argument("--config", default=CONFIG, help="Dataset config name.")
    parser.add_argument("--split", default=SPLIT, help="Dataset split name.")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for saved images.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing mapping_file.json and continue after the last id.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite image files that already exist on disk.")
    return parser.parse_args()


def _wrap_span(words: list[str]) -> str:
    """Bracket a changed span, keeping leading/trailing punctuation outside."""
    span = " ".join(words)
    match = re.match(r"^(\W*)(.*?)(\W*)$", span, flags=re.DOTALL)
    if not match:
        return f"[{span}]"
    lead, core, trail = match.groups()
    if not core:
        return span
    return f"{lead}[{core}]{trail}"


def bracket_diff(source: str, target: str) -> tuple[str, str]:
    """Wrap the differing word spans of each prompt in square brackets."""
    source_words = source.split()
    target_words = target.split()
    matcher = difflib.SequenceMatcher(a=source_words, b=target_words, autojunk=False)
    
    # Initialize lists to store the output words.
    source_out: list[str] = []
    target_out: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            source_out.extend(source_words[i1:i2])
            target_out.extend(target_words[j1:j2])
            continue
        if i2 > i1:
            source_out.append(_wrap_span(source_words[i1:i2]))
        if j2 > j1:
            target_out.append(_wrap_span(target_words[j1:j2]))

    return " ".join(source_out), " ".join(target_out)


def next_start_index(mapping: dict[str, dict[str, Any]]) -> int:
    max_index = -1
    for key in mapping:
        try:
            max_index = max(max_index, int(key))
        except ValueError:
            continue
    return max_index + 1


def load_existing_mapping(mapping_path: Path, resume: bool) -> dict[str, dict[str, Any]]:
    if resume and mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    return {}


def write_mapping(mapping_path: Path, mapping: dict[str, dict[str, Any]]) -> None:
    ordered = {key: mapping[key] for key in sorted(mapping)}
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(ordered, handle, indent=4, ensure_ascii=False)


def write_gitignore(datasets_dir: Path) -> None:
    """Drop a single .gitignore in the datasets/ folder.

    Uses the "ignore everything but keep the folder" pattern so every
    downloaded dataset folder (images, masks, mapping_file.json) is never
    committed, while this .gitignore itself stays tracked. Individual
    download folders inside datasets/ do not get their own .gitignore.
    """
    datasets_dir.mkdir(parents=True, exist_ok=True)
    gitignore_path = datasets_dir / ".gitignore"
    gitignore_path.write_text("*\n!.gitignore\n", encoding="utf-8")


def save_jpeg(image: Any, dst: Path, quality: int, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, format="JPEG", quality=quality)


def load_stream(dataset: str, config: str, split: str, token: str | None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with "
            "`pip install datasets pillow`."
        ) from exc

    stream = load_dataset(
        dataset, name=config, split=split, streaming=True, token=token
    )
    try:
        stream = stream.select_columns(NEEDED_COLUMNS)
    except Exception:  # noqa: BLE001 - select_columns is an optimization only
        pass
    return stream


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")

    if args.out_root:
        out_root = Path(args.out_root).expanduser()
    else:
        out_root = Path(DEFAULT_BASE_DIR) / f"UltraEdit_Region_{args.n_samples}"
    image_dir = out_root / "annotation_images"
    mask_dir = out_root / "annotation_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    write_gitignore(out_root.parent)
    mapping_path = out_root / "mapping_file.json"

    mapping = load_existing_mapping(mapping_path, args.resume)
    start_index = next_start_index(mapping)

    stream = load_stream(args.dataset, args.config, args.split, token)
    if start_index > 0:
        print(f"Resuming: skipping the first {start_index} already-saved rows.")
        stream = stream.skip(start_index)
    stream = stream.take(args.n_samples)

    print(
        f"Streaming {args.n_samples} samples "
        f"(rows {start_index}..{start_index + args.n_samples - 1}) from "
        f"{args.dataset} [{args.split}]."
    )

    saved = 0
    failed = 0
    for offset, example in enumerate(stream):
        sample_id = f"{start_index + offset:0{8}d}"
        image_name = f"{sample_id}.jpg"
        try:
            # Save the source image.
            save_jpeg(
                example["source_image"],
                image_dir / image_name,
                args.jpeg_quality,
                args.overwrite,
            )
            # Save the mask image.
            save_jpeg(
                example["mask_image"],
                mask_dir / image_name,
                args.jpeg_quality,
                args.overwrite,
            )
            source_caption = str(example.get("source_caption", "") or "")
            target_caption = str(example.get("target_caption", "") or "")
            editing_instruction = str(example.get("edit_prompt", "") or "")
            original_prompt, editing_prompt = bracket_diff(source_caption, target_caption)
            mapping[sample_id] = {
                "image_path": f"annotation_images/{image_name}",
                "mask_image_path": f"annotation_masks/{image_name}",
                "original_prompt": original_prompt,
                "editing_prompt": editing_prompt,
                "editing_instruction": editing_instruction,
            }
            saved += 1
            print(f"[{saved}/{args.n_samples}] saved {sample_id}")
        except Exception as exc:  # noqa: BLE001 - keep going on per-row failures
            failed += 1
            print(f"[{offset + 1}] FAILED {sample_id}: {exc}")

        if saved % 200 == 0 and saved > 0:
            write_mapping(mapping_path, mapping)

    write_mapping(mapping_path, mapping)
    print(f"Saved {saved} samples, {failed} failed.")
    print(f"Wrote mapping for {len(mapping)} total samples to {mapping_path}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    # Some C extensions (torch/pyarrow) can crash during interpreter shutdown
    # after all work is done. Everything is already flushed to disk, so exit
    # immediately to avoid that noisy teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
