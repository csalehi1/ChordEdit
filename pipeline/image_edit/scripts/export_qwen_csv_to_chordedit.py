"""
Export a Qwen-labeled ImgEdit PIE-style CSV into ChordEdit's mapping layout.

Output layout:
  out_root/
    annotation_images/
    target_images/          optional, for reference only
    annotation_masks/       optional full-image masks, for compatibility
    mapping_file.json
    exported_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Qwen-labeled ImgEdit CSV into ChordEdit mapping_file.json format."
    )
    parser.add_argument("--input-csv", required=True, help="Qwen-labeled CSV path.")
    parser.add_argument("--out-root", required=True, help="ChordEdit export folder to create.")
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        help="Root folder where ImgEdit images live. Can be passed more than once.",
    )
    parser.add_argument(
        "--recursive-image-search",
        action="store_true",
        help="Index images below --image-root so CSV basenames can be resolved.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum exported rows.")
    parser.add_argument(
        "--min-confidence",
        choices=["low", "medium", "high"],
        default="low",
        help="Minimum Qwen confidence to export.",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Do not skip rows with a non-empty vlm_error field.",
    )
    parser.add_argument(
        "--copy-targets",
        action="store_true",
        help="Copy target images into out_root/target_images for reference.",
    )
    parser.add_argument(
        "--make-full-masks",
        action="store_true",
        help="Create full-white masks in out_root/annotation_masks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def safe_sample_id(value: str, fallback: str) -> str:
    raw = value.strip() or fallback
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return safe.strip("._") or fallback


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_image_index(roots: Iterable[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            normalized = normalize_path(str(path))
            for key in {normalized, path.name, path.stem}:
                index.setdefault(key, path)
    return index


def resolve_image(value: str, roots: list[Path], image_index: dict[str, Path]) -> Path | None:
    normalized = normalize_path(value)
    if not normalized:
        return None

    raw = Path(normalized)
    if raw.is_absolute() and raw.exists():
        return raw

    for root in roots:
        candidates = [
            root / normalized,
            root / normalized.lstrip("/"),
            root / raw.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    for key in (normalized, raw.name, raw.stem):
        if key in image_index:
            return image_index[key]
    return None


def save_as_png(src: Path, dst: Path, overwrite: bool) -> tuple[int, int]:
    from PIL import Image

    if dst.exists() and not overwrite:
        with Image.open(dst) as existing:
            return existing.size

    with Image.open(src) as image:
        rgb = image.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dst)
        return rgb.size


def copy_reference_image(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not dst.exists():
        shutil.copy2(src, dst)


def save_full_mask(path: Path, size: tuple[int, int], overwrite: bool) -> None:
    from PIL import Image

    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 255).save(path)


def row_is_exportable(row: dict[str, str], min_confidence: str, include_errors: bool) -> bool:
    if not row.get("source_prompt", "").strip() or not row.get("target_prompt", "").strip():
        return False
    if not include_errors and row.get("vlm_error", "").strip():
        return False

    confidence = row.get("vlm_confidence", "low").strip().lower() or "low"
    return CONFIDENCE_RANK.get(confidence, 0) >= CONFIDENCE_RANK[min_confidence]


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    image_roots = [Path(root).expanduser().resolve() for root in args.image_root]

    rows = read_csv(input_csv)
    image_index = build_image_index(image_roots) if args.recursive_image_search else {}

    image_dir = out_root / "annotation_images"
    target_dir = out_root / "target_images"
    mask_dir = out_root / "annotation_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_targets:
        target_dir.mkdir(parents=True, exist_ok=True)
    if args.make_full_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, dict[str, str]] = {}
    exported_rows: list[dict[str, str]] = []
    skipped = 0
    missing_images = 0

    for row in rows:
        if args.limit is not None and len(mapping) >= args.limit:
            break
        if not row_is_exportable(row, args.min_confidence, args.include_errors):
            skipped += 1
            continue

        source_path = resolve_image(row.get("source_image", ""), image_roots, image_index)
        if source_path is None:
            missing_images += 1
            continue

        fallback_id = f"imgedit_{len(mapping):06d}"
        sample_id = safe_sample_id(row.get("id", ""), fallback_id)
        base_sample_id = sample_id
        suffix = 1
        while sample_id in mapping:
            sample_id = f"{base_sample_id}_{suffix}"
            suffix += 1

        image_name = f"{sample_id}.png"
        image_size = save_as_png(source_path, image_dir / image_name, args.overwrite)

        entry: dict[str, str] = {
            "dataset_name": row.get("dataset_name", "imgedit") or "imgedit",
            "image_path": image_name,
            "source_prompt": row.get("source_prompt", "").strip(),
            "target_prompt": row.get("target_prompt", "").strip(),
            "editing_instruction": row.get("editing_instruction", "").strip(),
            "foreground": row.get("foreground", "").strip(),
            "foreground_target": row.get("foreground_target", "").strip(),
            "edit_type": row.get("qwen_edit_type", "").strip()
            or row.get("edit_type", "").strip(),
            "clip_score": row.get("clip_score", "").strip(),
            "clip_score_category": row.get("clip_score_category", "").strip(),
            "vlm_confidence": row.get("vlm_confidence", "").strip(),
            "vlm_notes": row.get("vlm_notes", "").strip(),
            "original_source_image": row.get("source_image", "").strip(),
            "original_target_image": row.get("target_image", "").strip(),
        }

        if args.make_full_masks:
            mask_name = f"{sample_id}.png"
            save_full_mask(mask_dir / mask_name, image_size, args.overwrite)
            entry["mask_path"] = mask_name

        if args.copy_targets:
            target_path = resolve_image(row.get("target_image", ""), image_roots, image_index)
            if target_path is not None:
                target_name = f"{sample_id}{target_path.suffix.lower()}"
                copy_reference_image(target_path, target_dir / target_name, args.overwrite)
                entry["target_image_path"] = f"target_images/{target_name}"

        mapping[sample_id] = entry
        exported = dict(row)
        exported["chordedit_sample_id"] = sample_id
        exported["chordedit_image_path"] = image_name
        exported_rows.append(exported)

    with (out_root / "mapping_file.json").open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)

    if exported_rows:
        fieldnames = list(exported_rows[0].keys())
        with (out_root / "exported_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(exported_rows)

    print(f"Exported {len(mapping)} ChordEdit rows to {out_root}")
    print(f"Skipped {skipped} rows because prompts/errors/confidence did not pass filters")
    print(f"Skipped {missing_images} rows because source images were missing")


if __name__ == "__main__":
    main()
