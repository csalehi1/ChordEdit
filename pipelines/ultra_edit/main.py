"""
Download the first N rows from UltraEdit Region-Based 100k and write a
PIE-Bench_v1-style folder (annotation_images/, annotation_masks/, mapping_file.json).

Dataset: https://huggingface.co/datasets/BleachNick/UltraEdit_Region_Based_100k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import bracket_diff, save_image

DATASET = "BleachNick/UltraEdit_Region_Based_100k"
SPLIT = "RegionBase"
BASE_COLUMNS = ["source_image", "source_caption", "target_caption", "edit_prompt"]
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# JPEG quality for saved images to save space. 
# Set to None to write lossless PNG at native resolution.
JPEG_QUALITY: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--include-edits", action="store_true")
    parser.add_argument("--include-masks", action="store_true")
    return parser.parse_args()


def load_stream(token: str | None, include_masks: bool, include_edits: bool) -> Any:
    try:
        from datasets import load_dataset
        stream = load_dataset(DATASET, name="default", split=SPLIT, streaming=True, token=token)
        columns = list(BASE_COLUMNS)
        if include_masks:
            columns.append("mask_image")
        if include_edits:
            columns.append("edited_image")
        return stream.select_columns(columns)
    except ImportError as exc:
        raise SystemExit("Install dependencies: pip install datasets pillow") from exc


def main() -> int:
    args = parse_args()
    parent = args.output_root or DATASETS_DIR
    out_root = parent / f"UltraEdit_Region_{args.n_samples}"
    image_dir = out_root / "annotation_images"
    mask_dir = out_root / "annotation_masks"
    edit_dir = out_root / "annotation_edits"
    mapping_path = out_root / "mapping_file.json"

    # Check if the output folder already exists and is not empty.
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output folder already exists: {out_root}")
    image_dir.mkdir(parents=True)
    if args.include_masks:
        mask_dir.mkdir(parents=True)
    if args.include_edits:
        edit_dir.mkdir(parents=True)

    # Keep downloaded folders out of git when writing under ./datasets/.
    if out_root.parent.name == "datasets":
        gitignore = out_root.parent / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

    stream = load_stream(
        os.environ.get("HF_TOKEN"), args.include_masks, args.include_edits
    ).take(args.n_samples)
    print(f"Streaming {args.n_samples} samples from {DATASET}")

    image_ext = ".jpg" if JPEG_QUALITY is not None else ".png"
    mapping: dict[str, dict[str, str]] = {}
    failed = 0
    for index, example in enumerate(stream):
        sample_start = time.perf_counter()
        sample_id, image_name = f"{index:08d}", f"{index:08d}{image_ext}"
        try:
            save_image(example["source_image"], image_dir / image_name, JPEG_QUALITY)
            original_prompt, editing_prompt = bracket_diff(str(example.get("source_caption")), str(example.get("target_caption")))
            entry: dict[str, str] = {}
            entry["image_path"] = f"annotation_images/{image_name}"
            if args.include_masks:
                save_image(example["mask_image"], mask_dir / image_name, JPEG_QUALITY)
                entry["mask_image_path"] = f"annotation_masks/{image_name}"
            if args.include_edits:
                save_image(example["edited_image"], edit_dir / image_name, JPEG_QUALITY)
                entry["edited_image_path"] = f"annotation_edits/{image_name}"
            entry["original_prompt"] = original_prompt
            entry["editing_prompt"] = editing_prompt
            entry["editing_instruction"] = str(example.get("edit_prompt"))
            mapping[sample_id] = entry
            elapsed = time.perf_counter() - sample_start
            print(f"[{len(mapping)}/{args.n_samples}] saved {sample_id} ({elapsed:.2f}s)")
        except Exception as exc:
            # One bad row should not stop the run, but should be logged.
            failed += 1
            print(f"[{index + 1}] FAILED {sample_id}: {exc}")

    # Write the PIE-Bench-style mapping file.
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=4, ensure_ascii=False)

    print(f"Done: {len(mapping)} saved, {failed} failed")
    print(f"Saved to: {out_root}")
    return 0


if __name__ == "__main__":
    code = main()
    # Avoid noisy C-extension teardown after pyarrow/torch.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
