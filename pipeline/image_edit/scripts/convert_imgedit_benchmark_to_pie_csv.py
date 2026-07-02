#!/usr/bin/env python3
"""
Convert ImgEdit-Bench files from Benchmark.tar into a ChordEdit/PIE-style CSV.

ImgEdit-Bench usually provides a source image and an edit instruction, not a
finished target image. That is still enough for ChordEdit after a VLM generates:
  source_prompt: caption of the source image
  target_prompt: caption of the intended edited image
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


CSV_COLUMNS = [
    "id",
    "dataset_name",
    "benchmark_split",
    "edit_type",
    "source_image",
    "target_image",
    "mask_path",
    "source_prompt",
    "target_prompt",
    "editing_instruction",
    "clip_score",
    "clip_score_category",
    "source_metadata",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert extracted ImgEdit Benchmark.tar into PIE-style CSV."
    )
    parser.add_argument(
        "--benchmark-root",
        required=True,
        help=(
            "Extracted benchmark root, e.g. "
            "C:\\Datasets\\ImgEdit\\Benchmark\\Benchmark"
        ),
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--include-hard",
        action="store_true",
        help="Also include Benchmark/hard/annotation.jsonl rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max rows for a smoke test.",
    )
    return parser.parse_args()


def image_exists(root: Path, relative_path: str) -> bool:
    return (root / relative_path).exists()


def iter_singleturn(root: Path) -> Iterable[dict[str, Any]]:
    metadata_path = root / "singleturn" / "singleturn.json"
    if not metadata_path.exists():
        return

    data = json.load(metadata_path.open("r", encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {metadata_path}")

    for key, item in sorted(data.items(), key=lambda pair: str(pair[0])):
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("id", "")).strip()
        if not image_id:
            continue
        relative_image = f"singleturn/{image_id}".replace("\\", "/")
        if not image_exists(root, relative_image):
            continue

        yield {
            "id": f"imgedit_bench_singleturn_{key}",
            "dataset_name": "imgedit_benchmark",
            "benchmark_split": "singleturn",
            "edit_type": str(item.get("edit_type", "")).strip(),
            "source_image": relative_image,
            "target_image": "",
            "mask_path": "",
            "source_prompt": "",
            "target_prompt": "",
            "editing_instruction": str(item.get("prompt", "")).strip(),
            "clip_score": "",
            "clip_score_category": "missing",
            "source_metadata": metadata_path.as_posix(),
        }


def iter_hard(root: Path) -> Iterable[dict[str, Any]]:
    metadata_path = root / "hard" / "annotation.jsonl"
    if not metadata_path.exists():
        return

    with metadata_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_id = str(item.get("id", "")).strip()
            if not image_id:
                continue
            relative_image = f"hard/{image_id}".replace("\\", "/")
            if not image_exists(root, relative_image):
                continue

            yield {
                "id": f"imgedit_bench_hard_{index:04d}",
                "dataset_name": "imgedit_benchmark",
                "benchmark_split": "hard",
                "edit_type": str(item.get("edit_type", "hard")).strip(),
                "source_image": relative_image,
                "target_image": "",
                "mask_path": "",
                "source_prompt": "",
                "target_prompt": "",
                "editing_instruction": str(item.get("prompt", "")).strip(),
                "clip_score": "",
                "clip_score_category": "missing",
                "source_metadata": metadata_path.as_posix(),
            }


def main() -> int:
    args = parse_args()
    root = Path(args.benchmark_root)
    if not root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {root}")

    rows = list(iter_singleturn(root))
    if args.include_hard:
        rows.extend(iter_hard(root))
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} ImgEdit benchmark rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
