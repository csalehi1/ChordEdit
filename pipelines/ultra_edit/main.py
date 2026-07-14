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
import time
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import bracket_diff, save_image, save_raw_bytes

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

    # List of columns to load from the dataset.
    columns = list(BASE_COLUMNS)
    if args.include_masks:
        columns.append("mask_image")
    if args.include_edits:
        columns.append("edited_image")
    stream = load_dataset(
        DATASET, name="default", split=SPLIT,
        streaming=True, token=os.environ.get("HF_TOKEN"), columns=columns,
    )

    # When JPEG_QUALITY is None, we don't need to decode the images.
    # The raw bytes are saved to the image directory directly.
    if JPEG_QUALITY is None:
        stream = stream.decode(False)
    else:
        stream = stream.decode(num_threads=min(32, (os.cpu_count() or 1) + 4))
    stream = stream.take(args.n_samples)
    print(f"Streaming {args.n_samples} samples from {DATASET}")

    raw_mode = JPEG_QUALITY is None
    image_ext = ".jpg" if JPEG_QUALITY is not None else ".png"
    mapping: dict[str, dict[str, str]] = {}
    failed = 0
    for index, example in enumerate(stream):
        sample_start = time.perf_counter()
        sample_id = f"{index:08d}"
        base_name = f"{sample_id}{image_ext}"
        try:
            # List of columns to save to the image directory.
            image_cols = [("source_image", "image_path", image_dir)]
            if args.include_masks:
                image_cols.append(("mask_image", "mask_image_path", mask_dir))
            if args.include_edits:
                image_cols.append(("edited_image", "edited_image_path", edit_dir))

            # Save the images to the image directory.
            entry: dict[str, str] = {}
            for col, key, dest in image_cols:
                if raw_mode:
                    saved = save_raw_bytes(example[col]["bytes"], dest / base_name)
                else:
                    saved = dest / base_name
                    save_image(example[col], saved, JPEG_QUALITY)
                entry[key] = f"{dest.name}/{saved.name}"

            original_prompt, editing_prompt = bracket_diff(str(example.get("source_caption")), str(example.get("target_caption")))
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
