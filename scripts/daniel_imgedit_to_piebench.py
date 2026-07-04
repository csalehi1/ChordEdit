#!/usr/bin/env python3
"""Transform the ImgEdit dataset into a flat, PIE-Bench-style layout.

Output layout (mirrors PIE-Bench_v1, minus the category subfolders)::

    ImgEdit<N>/
        annotation_images/
            0000000000000000.jpg
            0000000000000001.jpg
            ...
        mapping_file.json

``<N>`` is the number of images that were actually written. Every image is
copied under ``annotation_images/`` with a zero-padded 16-digit id (0-indexed),
and ``mapping_file.json`` holds one entry per id using PIE-Bench field names.

Only rows whose *source* image can be resolved on disk are included (PIE-Bench
``annotation_images`` stores the original/source image to be edited). A ``mask``
entry is written only when a mask file is actually present for that row.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("imgedit_to_pie")

DEFAULT_IMGEDIT_ROOT = Path("/shared/ssd_30T/mirick/datasets/ImgEdit")
DEFAULT_CSV = "imgedit_pie_style_1000_qwen_labeled.csv"
DEFAULT_IMAGE_DIRS = ["part1"]
DEFAULT_ID_WIDTH = 16
ANNOTATION_SUBDIR = "annotation_images"
MAPPING_FILENAME = "mapping_file.json"

# Stable ImgEdit edit-type -> PIE-style numeric id (kept as strings like PIE-Bench).
EDIT_TYPE_IDS: Dict[str, str] = {
    "action": "0",
    "add": "1",
    "adjust": "2",
    "adjust_canny": "2",
    "background": "3",
    "content_memory": "4",
    "content_understanding": "5",
    "hybrid": "6",
    "reference_extract": "7",
    "reference_replace": "8",
    "remove": "9",
    "replace": "10",
    "style_transfer": "11",
    "version_backtracking": "12",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert ImgEdit into a flat PIE-Bench-style dataset (ImgEdit<N>).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--imgedit-root", type=Path, default=DEFAULT_IMGEDIT_ROOT,
                   help="Root of the ImgEdit dataset.")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV,
                   help="Metadata CSV (absolute, or relative to --imgedit-root).")
    p.add_argument("--image-dirs", nargs="+", default=DEFAULT_IMAGE_DIRS,
                   help="Directories (absolute, or relative to --imgedit-root) that are "
                        "recursively indexed by basename to resolve source/mask images.")
    p.add_argument("--mask-dirs", nargs="+", default=None,
                   help="Directories to resolve mask files. Defaults to --image-dirs.")
    p.add_argument("--output-parent", type=Path, default=None,
                   help="Where the ImgEdit<N> folder is created. Defaults to the parent "
                        "directory of --imgedit-root (sibling of the source dataset).")
    p.add_argument("--folder-prefix", type=str, default="ImgEdit",
                   help="Prefix for the output folder; the image count is appended.")
    p.add_argument("--id-width", type=int, default=DEFAULT_ID_WIDTH,
                   help="Zero-padding width for the numeric image ids.")
    p.add_argument("--start-index", type=int, default=0,
                   help="First numeric id (0-indexed like PIE-Bench).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N resolvable rows.")
    p.add_argument("--require-target", action="store_true",
                   help="Also require the target image to exist locally (default: only "
                        "the source image is required, matching PIE-Bench).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite the output folder if it already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written without touching disk.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def resolve_under_root(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (root / candidate)


def build_basename_index(dirs: List[Path]) -> Dict[str, Path]:
    """Map image basename -> path, walking each directory recursively.

    First occurrence wins so earlier directories take precedence.
    """
    index: Dict[str, Path] = {}
    for d in dirs:
        if not d.exists():
            LOGGER.warning("Image directory does not exist, skipping: %s", d)
            continue
        for cur, _, files in os.walk(d):
            for fn in files:
                index.setdefault(fn, Path(cur) / fn)
    return index


def resolve_image(name: str, index: Dict[str, Path], search_dirs: List[Path]) -> Optional[Path]:
    """Resolve an image reference to an existing path.

    Handles absolute paths, paths relative to the search dirs, and bare basenames
    (looked up in the prebuilt index). Windows-style paths in the CSV fall back to
    their basename.
    """
    if not name:
        return None
    raw = Path(name.replace("\\", "/"))
    if raw.is_absolute() and raw.exists():
        return raw
    for d in search_dirs:
        cand = d / raw
        if cand.exists():
            return cand
    return index.get(raw.name)


def editing_type_id(row: Dict[str, str]) -> str:
    for key in ("qwen_edit_type", "edit_type"):
        val = (row.get(key) or "").strip().lower()
        if val and val in EDIT_TYPE_IDS:
            return EDIT_TYPE_IDS[val]
    return "0"


def blended_word(row: Dict[str, str]) -> str:
    fg = (row.get("foreground") or "").strip()
    fgt = (row.get("foreground_target") or "").strip()
    return " ".join(part for part in (fg, fgt) if part)


def build_entry(row: Dict[str, str], image_filename: str) -> Dict[str, object]:
    """Construct a PIE-Bench-style mapping entry (mask added separately)."""
    return {
        "image_path": image_filename,
        "original_prompt": row.get("source_prompt", "") or "",
        "editing_prompt": row.get("target_prompt", "") or "",
        "editing_instruction": row.get("editing_instruction", "") or "",
        "editing_type_id": editing_type_id(row),
        "blended_word": blended_word(row),
        # Extra provenance fields (ignored by PIE consumers, useful for tracing).
        "edit_type": row.get("edit_type", "") or "",
        "imgedit_id": row.get("id", "") or "",
        "source_image": row.get("source_image", "") or "",
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    imgedit_root = args.imgedit_root.expanduser().resolve()
    csv_path = resolve_under_root(imgedit_root, args.csv).resolve()
    if not csv_path.exists():
        LOGGER.error("CSV not found: %s", csv_path)
        return 1

    image_dirs = [resolve_under_root(imgedit_root, d).resolve() for d in args.image_dirs]
    mask_dir_args = args.mask_dirs if args.mask_dirs is not None else args.image_dirs
    mask_dirs = [resolve_under_root(imgedit_root, d).resolve() for d in mask_dir_args]

    output_parent = (args.output_parent.expanduser().resolve()
                     if args.output_parent is not None else imgedit_root.parent)

    LOGGER.info("ImgEdit root : %s", imgedit_root)
    LOGGER.info("CSV          : %s", csv_path)
    LOGGER.info("Image dirs   : %s", ", ".join(str(d) for d in image_dirs))

    LOGGER.info("Indexing image directories ...")
    image_index = build_basename_index(image_dirs)
    mask_index = image_index if mask_dirs == image_dirs else build_basename_index(mask_dirs)
    LOGGER.info("Indexed %d image basenames.", len(image_index))

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    LOGGER.info("Loaded %d CSV rows.", len(rows))

    # Pass 1: select resolvable rows (source image required; target optional).
    selected: List[Tuple[Dict[str, str], Path, Optional[Path]]] = []
    missing_source = missing_target = 0
    for row in rows:
        src = resolve_image(row.get("source_image", ""), image_index, image_dirs)
        if src is None:
            missing_source += 1
            continue
        tgt = resolve_image(row.get("target_image", ""), image_index, image_dirs)
        if args.require_target and tgt is None:
            missing_target += 1
            continue
        selected.append((row, src, tgt))
        if args.limit is not None and len(selected) >= args.limit:
            break

    n = len(selected)
    LOGGER.info("Resolvable rows: %d (missing source=%d, skipped for missing target=%d)",
                n, missing_source, missing_target)
    if n == 0:
        LOGGER.error("No rows had a resolvable source image; nothing to write.")
        return 1

    out_dir = output_parent / f"{args.folder_prefix}{n}"
    annotations_dir = out_dir / ANNOTATION_SUBDIR
    mapping_path = out_dir / MAPPING_FILENAME
    LOGGER.info("Output folder: %s", out_dir)

    if out_dir.exists() and not args.overwrite and not args.dry_run:
        LOGGER.error("Output folder already exists (use --overwrite): %s", out_dir)
        return 1

    mapping: Dict[str, Dict[str, object]] = {}
    masks_written = 0
    dup_basenames = 0
    seen_sources: Dict[str, str] = {}

    if not args.dry_run:
        annotations_dir.mkdir(parents=True, exist_ok=True)

    for offset, (row, src_path, _tgt) in enumerate(selected):
        sample_id = f"{args.start_index + offset:0{args.id_width}d}"
        ext = src_path.suffix or ".jpg"
        image_filename = f"{sample_id}{ext}"
        dest = annotations_dir / image_filename

        if src_path.name in seen_sources:
            dup_basenames += 1
            LOGGER.debug("Duplicate source image %s (ids %s and %s)",
                         src_path.name, seen_sources[src_path.name], sample_id)
        else:
            seen_sources[src_path.name] = sample_id

        entry = build_entry(row, image_filename)

        # Mask: only include when an actual mask file is present.
        mask_ref = (row.get("mask_path") or "").strip()
        if mask_ref:
            mask_path = resolve_image(mask_ref, mask_index, mask_dirs)
            if mask_path is not None:
                mask_ext = mask_path.suffix or ".png"
                mask_filename = f"{sample_id}_mask{mask_ext}"
                entry["mask"] = mask_filename
                if not args.dry_run:
                    shutil.copy2(mask_path, annotations_dir / mask_filename)
                masks_written += 1
            else:
                LOGGER.warning("Row %s references mask '%s' but it was not found; "
                               "omitting mask.", sample_id, mask_ref)

        if not args.dry_run:
            shutil.copy2(src_path, dest)

        mapping[sample_id] = entry

    if args.dry_run:
        LOGGER.info("[dry-run] Would write %d images and mapping (%d masks) to %s",
                    n, masks_written, out_dir)
    else:
        with mapping_path.open("w", encoding="utf-8") as fh:
            json.dump(mapping, fh, indent=2, ensure_ascii=False)
        LOGGER.info("Wrote %d images (%d masks) + %s", n, masks_written, mapping_path)

    if dup_basenames:
        LOGGER.warning("%d rows reused a source image already assigned to another id.",
                       dup_basenames)

    LOGGER.info("Done. Dataset: %s (%d entries)", out_dir, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
