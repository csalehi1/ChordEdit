#!/usr/bin/env python3
"""
Download the first N rows from UltraEdit Region-Based 100k and write a
PIE-Bench_v1-style folder (annotation_images/, annotation_masks/, mapping_file.json).

Dataset: https://huggingface.co/datasets/BleachNick/UltraEdit_Region_Based_100k
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
SPLIT = "RegionBase"
COLUMNS = ["source_image", "mask_image", "source_caption", "target_caption", "edit_prompt"]
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# JPEG quality for saved images to save space. 
# Set to None to write lossless PNG at native resolution.
JPEG_QUALITY: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, required=True)
    return parser.parse_args()


def bracket_diff(source: str, target: str) -> tuple[str, str]:
    """Wrap differing word spans in [brackets] for PIE-Bench-style prompts."""

    def wrap(words: list[str]) -> str:
        span = " ".join(words)
        # Split the span into leading/trailing non-word characters and the core content.
        match = re.match(r"^(\W*)(.*?)(\W*)$", span, flags=re.DOTALL)
        if not match or not match.group(2):
            return span
        lead, core, trail = match.groups()
        # Wrap only the core content in [brackets].
        return f"{lead}[{core}]{trail}"

    source_words, target_words = source.split(), target.split()
    # Use Python's built-in difflib to find the longest common subsequence.
    matcher = difflib.SequenceMatcher(a=source_words, b=target_words, autojunk=False)
    source_out, target_out = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            source_out.extend(source_words[i1:i2])
            target_out.extend(target_words[j1:j2])
        else:
            if i2 > i1:
                source_out.append(wrap(source_words[i1:i2]))
            if j2 > j1:
                target_out.append(wrap(target_words[j1:j2]))
    return " ".join(source_out), " ".join(target_out)


def save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if JPEG_QUALITY is None:
        # No re-encoding, keep the image's native mode and full resolution.
        image.save(path)
        return
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(path, format="JPEG", quality=JPEG_QUALITY)


def load_stream(token: str | None) -> Any:
    try:
        # Streaming reads Parquet shards from the hub.
        from datasets import load_dataset
        stream = load_dataset(DATASET, name="default", split=SPLIT, streaming=True, token=token)
        return stream.select_columns(COLUMNS)
    except ImportError as exc:
        raise SystemExit("Install dependencies: pip install datasets pillow") from exc



def main() -> int:
    args = parse_args()
    out_root = DATASETS_DIR / f"UltraEdit_Region_{args.n_samples}"
    image_dir = out_root / "annotation_images"
    mask_dir = out_root / "annotation_masks"
    mapping_path = out_root / "mapping_file.json"

    # Check if the output folder already exists and is not empty.
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output folder already exists: {out_root}")
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Keep downloaded folders out of git when writing under ./datasets/.
    if out_root.parent.name == "datasets":
        gitignore = out_root.parent / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

    stream = load_stream(os.environ.get("HF_TOKEN")).take(args.n_samples)
    print(f"Streaming {args.n_samples} samples from {DATASET} [{SPLIT}] -> {out_root}")

    image_ext = ".jpg" if JPEG_QUALITY is not None else ".png"
    mapping: dict[str, dict[str, str]] = {}
    failed = 0
    for index, example in enumerate(stream):
        sample_id, image_name = f"{index:08d}", f"{index:08d}{image_ext}"
        try:
            save_image(example["source_image"], image_dir / image_name)
            save_image(example["mask_image"], mask_dir / image_name)
            original_prompt, editing_prompt = bracket_diff(str(example.get("source_caption")), str(example.get("target_caption")))
            mapping[sample_id] = {
                "image_path": f"annotation_images/{image_name}",
                "mask_image_path": f"annotation_masks/{image_name}",
                "original_prompt": original_prompt,
                "editing_prompt": editing_prompt,
                "editing_instruction": str(example.get("edit_prompt")),
            }
            print(f"[{len(mapping)}/{args.n_samples}] saved {sample_id}")
        except Exception as exc:
            # One bad row should not stop the run, but should be logged.
            failed += 1
            print(f"[{index + 1}] FAILED {sample_id}: {exc}")

    # Write the PIE-Bench-style mapping file.
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=4, ensure_ascii=False)

    print(f"Done: {len(mapping)} saved, {failed} failed -> {mapping_path}")
    return 0


if __name__ == "__main__":
    code = main()
    # Avoid noisy C-extension teardown after pyarrow/torch.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
