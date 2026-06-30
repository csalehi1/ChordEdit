#!/usr/bin/env python3
"""
Convert ImgEdit parquet metadata into a PIE-Bench-style CSV.

This is the first conversion step: it flattens ImgEdit's final parquet records
into one CSV row per edit. It can also join an optional CLIP-score/preprocess
file and add a high/medium/low category for each row.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PIE_COLUMNS = [
    "id",
    "dataset_name",
    "edit_type",
    "source_image",
    "target_image",
    "mask_path",
    "source_prompt",
    "target_prompt",
    "editing_instruction",
    "clip_score",
    "clip_score_category",
    "turn_index",
    "source_parquet",
    "sample_key",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten ImgEdit parquet metadata into a PIE-style CSV."
    )
    parser.add_argument(
        "--parquet-root",
        required=True,
        help="Path to one ImgEdit .parquet file or to the ImgEdit/Parquet directory.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path, e.g. imgedit_pie_style.csv.",
    )
    parser.add_argument(
        "--clip-score-file",
        default=None,
        help=(
            "Optional JSON/JSONL/CSV score or preprocessing file. The converter "
            "tries to map scores by image path, basename, or ImgEdit sample key."
        ),
    )
    parser.add_argument(
        "--clip-aggregate",
        choices=["mean", "max", "min", "first"],
        default="mean",
        help=(
            "How to collapse multiple segmentation clip_score values for one "
            "source image when using preprocessing JSON."
        ),
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.75,
        help="Minimum CLIP score for the 'medium' category.",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.90,
        help="Minimum CLIP score for the 'high' category.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Rows to process per parquet batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CSV rows to write for a smoke test.",
    )
    return parser.parse_args()


def find_parquet_files(parquet_root: Path) -> list[Path]:
    if parquet_root.is_file():
        if parquet_root.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a .parquet file, got: {parquet_root}")
        return [parquet_root]

    if not parquet_root.is_dir():
        raise FileNotFoundError(f"Parquet path does not exist: {parquet_root}")

    files = sorted(parquet_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under: {parquet_root}")
    return files


def edit_type_from_file(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^(?P<task>.+?)_part\d+$", stem)
    return match.group("task") if match else stem


def first_item(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return first_item(value[0]) if value else ""
    try:
        if hasattr(value, "tolist"):
            return first_item(value.tolist())
    except Exception:
        pass
    return str(value)


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def sample_key_from_path(path_value: str) -> str:
    normalized = normalize_path(path_value)
    if not normalized:
        return ""

    path = Path(normalized)
    parent = path.parent.name
    if parent:
        return parent
    return path.stem


def candidate_keys(path_value: str) -> set[str]:
    normalized = normalize_path(path_value)
    if not normalized:
        return set()

    path = Path(normalized)
    stem = path.stem
    parent = path.parent.name
    without_ext = str(path.with_suffix("")).replace("\\", "/")

    keys = {
        normalized,
        without_ext,
        path.name,
        stem,
        parent,
        normalized.replace("/", "_"),
        without_ext.replace("/", "_"),
    }
    return {key for key in keys if key and key != "."}


def category_for_clip_score(
    score: Any, medium_threshold: float, high_threshold: float
) -> str:
    if score in ("", None):
        return "missing"
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return "missing"
    if numeric_score >= high_threshold:
        return "high"
    if numeric_score >= medium_threshold:
        return "medium"
    return "low"


def collect_clip_scores(value: Any) -> list[float]:
    scores: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key.lower() in {"clip_score", "clipscore", "clip"}:
                    try:
                        scores.append(float(child))
                    except (TypeError, ValueError):
                        pass
                else:
                    walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return scores


def aggregate_scores(scores: list[float], mode: str) -> str:
    if not scores:
        return ""
    if mode == "mean":
        return f"{mean(scores):.6f}"
    if mode == "max":
        return f"{max(scores):.6f}"
    if mode == "min":
        return f"{min(scores):.6f}"
    return f"{scores[0]:.6f}"


def score_from_record(record: dict[str, Any], aggregate: str) -> str:
    direct_fields = [
        "clip_score",
        "clipscore",
        "clip",
        "CLIPScore",
        "image_clip_score",
    ]
    for field in direct_fields:
        if field in record and record[field] not in ("", None):
            try:
                return f"{float(record[field]):.6f}"
            except (TypeError, ValueError):
                return str(record[field])

    return aggregate_scores(collect_clip_scores(record), aggregate)


def record_path_values(record: dict[str, Any]) -> list[str]:
    path_fields = [
        "path",
        "image",
        "image_path",
        "source_image",
        "input_image",
        "original_image",
        "file",
        "filename",
    ]
    values: list[str] = []
    for field in path_fields:
        if field in record:
            values.append(first_item(record[field]))

    for list_field in ("input_images", "source_images", "images"):
        if list_field in record:
            values.append(first_item(record[list_field]))

    return [value for value in values if value]


def add_score_mapping(
    mapping: dict[str, str], record: dict[str, Any], aggregate: str
) -> None:
    score = score_from_record(record, aggregate)
    if score == "":
        return

    for path_value in record_path_values(record):
        for key in candidate_keys(path_value):
            mapping.setdefault(key, score)


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first_char = handle.read(1)
        handle.seek(0)

        if first_char == "[":
            data = json.load(handle)
            for item in data:
                if isinstance(item, dict):
                    yield item
            return

        if first_char == "{":
            data = json.load(handle)
            if isinstance(data, dict):
                if all(isinstance(value, dict) for value in data.values()):
                    for key, item in data.items():
                        item = dict(item)
                        item.setdefault("path", key)
                        yield item
                else:
                    yield data
            return

        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                yield item


def load_clip_score_map(path_value: str | None, aggregate: str) -> dict[str, str]:
    if not path_value:
        return {}

    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"CLIP score file does not exist: {path}")

    mapping: dict[str, str] = {}
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                add_score_mapping(mapping, dict(record), aggregate)
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        for record in iter_json_records(path):
            add_score_mapping(mapping, record, aggregate)
    else:
        raise ValueError(
            "Unsupported score file type. Use .json, .jsonl, .ndjson, or .csv."
        )

    return mapping


def lookup_clip_score(score_map: dict[str, str], source_image: str) -> str:
    if not score_map:
        return ""
    for key in candidate_keys(source_image):
        if key in score_map:
            return score_map[key]
    return ""


def flatten_imgedit_record(
    record: dict[str, Any],
    source_parquet: Path,
    edit_type: str,
    row_index: int,
    score_map: dict[str, str],
    medium_threshold: float,
    high_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(record.get("data"), list):
        turns = record["data"]
    else:
        turns = [record]

    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue

        source_image = first_item(turn.get("input_images") or turn.get("source_image"))
        target_image = first_item(turn.get("output_images") or turn.get("target_image"))
        editing_instruction = str(turn.get("prompt") or turn.get("instruction") or "")
        clip_score = lookup_clip_score(score_map, source_image)

        rows.append(
            {
                "id": f"imgedit_{row_index:08d}_{turn_index:02d}",
                "dataset_name": "imgedit",
                "edit_type": edit_type,
                "source_image": source_image,
                "target_image": target_image,
                "mask_path": "",
                "source_prompt": "",
                "target_prompt": "",
                "editing_instruction": editing_instruction,
                "clip_score": clip_score,
                "clip_score_category": category_for_clip_score(
                    clip_score, medium_threshold, high_threshold
                ),
                "turn_index": turn_index,
                "source_parquet": source_parquet.as_posix(),
                "sample_key": sample_key_from_path(source_image),
            }
        )

    return rows


def iter_parquet_records(path: Path, batch_size: int) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in your ChordEdit/ImgEdit "
            "environment with `pip install pyarrow`, then rerun this script."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for record in batch.to_pylist():
            yield record


def convert(args: argparse.Namespace) -> int:
    parquet_files = find_parquet_files(Path(args.parquet_root))
    score_map = load_clip_score_map(args.clip_score_file, args.clip_aggregate)

    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    source_record_index = 0

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIE_COLUMNS)
        writer.writeheader()

        for parquet_path in parquet_files:
            edit_type = edit_type_from_file(parquet_path)
            for record in iter_parquet_records(parquet_path, args.batch_size):
                rows = flatten_imgedit_record(
                    record=record,
                    source_parquet=parquet_path,
                    edit_type=edit_type,
                    row_index=source_record_index,
                    score_map=score_map,
                    medium_threshold=args.medium_threshold,
                    high_threshold=args.high_threshold,
                )
                source_record_index += 1

                for row in rows:
                    writer.writerow(row)
                    written += 1
                    if args.limit is not None and written >= args.limit:
                        return written

    return written


def main() -> int:
    args = parse_args()
    written = convert(args)
    print(f"Wrote {written:,} PIE-style rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
