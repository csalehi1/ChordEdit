#!/usr/bin/env python3
"""
Generate PIE-style source/target prompts for ImgEdit rows with a vision model.

Input:
  A PIE-style CSV from convert_imgedit_to_pie_csv.py.

Output:
  A new CSV with source_prompt and target_prompt filled from source/target
  images, plus review fields that account for CLIP-score quality.

This script is intentionally conservative: low or missing CLIP scores are
flagged for review rather than silently treated as clean labels.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


EXTRA_COLUMNS = [
    "prompt_generation_model",
    "prompt_confidence",
    "prompt_review_status",
    "edit_type_guess",
    "edit_summary",
    "prompt_notes",
    "vlm_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich 1000 ImgEdit demo rows with VLM-generated prompts."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="PIE-style CSV, e.g. C:\\Datasets\\ImgEdit\\imgedit_pie_style.csv.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV with generated prompts.",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        help=(
            "Root folder used to resolve relative image paths. Can be passed "
            "multiple times, e.g. --image-root C:\\Datasets\\ImgEdit"
        ),
    )
    parser.add_argument(
        "--recursive-image-search",
        action="store_true",
        help=(
            "Build a filename index under each image root. Useful after "
            "extracting ImgEdit tar files, because files may be nested."
        ),
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help=(
            "Use only source_image plus editing_instruction to generate both "
            "source_prompt and target_prompt. Use this for ImgEdit-Bench."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of rows to enrich.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based row offset in the input CSV.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="Vision-capable model to use.",
    )
    parser.add_argument(
        "--detail",
        choices=["low", "high", "original", "auto"],
        default="low",
        help="Image detail level. Use low for a cheap pilot, high/original for QA.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per row on transient API errors.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output CSV exists, skip rows whose id already appears there.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve files and write status rows without calling the API.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row.get("id", "") for row in csv.DictReader(handle)}


def output_columns(input_columns: list[str]) -> list[str]:
    columns = list(input_columns)
    for required in ("source_prompt", "target_prompt"):
        if required not in columns:
            columns.append(required)
    for column in EXTRA_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def normalize_relative_path(value: str) -> str:
    return value.strip().strip('"').replace("\\", "/")


def build_image_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
            try:
                relative = path.relative_to(root).as_posix()
                index.setdefault(relative, path)
            except ValueError:
                pass
    return index


def resolve_image_path(
    value: str, roots: list[Path], image_index: dict[str, Path] | None = None
) -> Path | None:
    if not value:
        return None

    normalized = normalize_relative_path(value)
    raw_path = Path(normalized)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    candidates = []
    for root in roots:
        candidates.append(root / normalized)
        candidates.append(root / normalized.lstrip("/"))
        candidates.append(root / Path(normalized).name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if image_index:
        path_name = Path(normalized).name
        path_stem = Path(normalized).stem
        for key in (normalized, path_name, path_stem):
            if key in image_index:
                return image_index[key]
    return None


def image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def clip_status(row: dict[str, str]) -> str:
    category = (row.get("clip_score_category") or "").strip().lower()
    score_text = (row.get("clip_score") or "").strip()
    if category and category != "missing":
        return category
    try:
        score = float(score_text)
    except (TypeError, ValueError):
        return "missing"
    if score >= 0.90:
        return "high"
    if score >= 0.75:
        return "medium"
    return "low"


def build_prompt(row: dict[str, str], source_only: bool = False) -> str:
    score = row.get("clip_score", "")
    category = clip_status(row)
    instruction = row.get("editing_instruction", "")
    edit_type = row.get("edit_type", "")

    image_description = (
        "You will receive one SOURCE image before editing."
        if source_only
        else "You will receive two images:\n1. SOURCE image before editing.\n2. TARGET image after editing."
    )

    target_rule = (
        "target_prompt should be one concise natural sentence describing the intended edited image after applying the editing instruction."
        if source_only
        else "target_prompt should be one concise natural sentence describing the TARGET image."
    )

    return f"""
You are converting an image-editing dataset row into PIE-Bench-style prompts.

{image_description}

Dataset edit_type: {edit_type}
Editing instruction: {instruction}
CLIP score: {score}
CLIP score category: {category}

Return strict JSON only, with exactly these keys:
{{
  "source_prompt": "...",
  "target_prompt": "...",
  "edit_summary": "...",
  "edit_type_guess": "...",
  "prompt_confidence": "high|medium|low",
  "prompt_review_status": "auto_accept|review_recommended|manual_review",
  "notes": "..."
}}

Rules:
- source_prompt should be one concise natural sentence describing the SOURCE image.
- {target_rule}
- The prompts should preserve shared scene context and differ mainly around the edit.
- Do not say "source image", "target image", "before", "after", or "edited image" in the prompts.
- If the edit instruction conflicts with the images, trust the images and explain briefly in notes.
- If CLIP score category is high, use auto_accept unless the images look mismatched.
- If CLIP score category is medium, use review_recommended unless the pair is obviously clean.
- If CLIP score category is low or missing, use manual_review unless the pair is unusually clear.
- Keep notes short.
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON: {text[:300]}")
    return json.loads(match.group(0))


def get_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install it with `pip install openai`."
        ) from exc
    return OpenAI()


def call_vision_model(
    client: Any,
    model: str,
    detail: str,
    row: dict[str, str],
    source_path: Path,
    target_path: Path | None,
) -> dict[str, Any]:
    content = [
        {"type": "input_text", "text": build_prompt(row, source_only=target_path is None)},
        {
            "type": "input_image",
            "image_url": image_to_data_url(source_path),
            "detail": detail,
        },
    ]
    if target_path is not None:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(target_path),
                "detail": detail,
            }
        )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_output_tokens=500,
    )
    return extract_json(response.output_text)


def enrich_row(
    row: dict[str, str],
    client: Any,
    args: argparse.Namespace,
    roots: list[Path],
    image_index: dict[str, Path] | None,
) -> dict[str, str]:
    enriched = dict(row)
    for column in EXTRA_COLUMNS:
        enriched.setdefault(column, "")

    source_path = resolve_image_path(row.get("source_image", ""), roots, image_index)
    target_path = resolve_image_path(row.get("target_image", ""), roots, image_index)

    if source_path is None or (target_path is None and not args.source_only):
        missing = []
        if source_path is None:
            missing.append("source_image")
        if target_path is None and not args.source_only:
            missing.append("target_image")
        enriched["prompt_review_status"] = "manual_review"
        enriched["prompt_confidence"] = "low"
        enriched["vlm_error"] = "missing local " + " and ".join(missing)
        return enriched

    if args.dry_run:
        enriched["prompt_review_status"] = "dry_run"
        enriched["prompt_confidence"] = "missing"
        enriched["prompt_notes"] = (
            f"Resolved source={source_path}; target={target_path}"
        )
        return enriched

    for attempt in range(1, args.max_retries + 1):
        try:
            result = call_vision_model(
                client=client,
                model=args.model,
                detail=args.detail,
                row=row,
                source_path=source_path,
                target_path=target_path if not args.source_only else None,
            )
            enriched["source_prompt"] = str(result.get("source_prompt", "")).strip()
            enriched["target_prompt"] = str(result.get("target_prompt", "")).strip()
            enriched["edit_summary"] = str(result.get("edit_summary", "")).strip()
            enriched["edit_type_guess"] = str(result.get("edit_type_guess", "")).strip()
            enriched["prompt_confidence"] = str(
                result.get("prompt_confidence", "")
            ).strip()
            enriched["prompt_review_status"] = str(
                result.get("prompt_review_status", "")
            ).strip()
            enriched["prompt_notes"] = str(result.get("notes", "")).strip()
            enriched["prompt_generation_model"] = args.model
            enriched["vlm_error"] = ""
            return enriched
        except Exception as exc:
            if attempt >= args.max_retries:
                enriched["prompt_review_status"] = "manual_review"
                enriched["prompt_confidence"] = "low"
                enriched["vlm_error"] = f"{type(exc).__name__}: {exc}"
                return enriched
            time.sleep(min(2**attempt, 30))

    return enriched


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    roots = [Path(root) for root in args.image_root]

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Set it first, or run with --dry-run."
        )

    rows = read_csv_rows(input_path)
    selected_rows = rows[args.start_index : args.start_index + args.limit]
    if not selected_rows:
        print("No rows selected.")
        return 0

    columns = output_columns(list(selected_rows[0].keys()))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = existing_ids(output_path) if args.resume else set()
    mode = "a" if args.resume and output_path.exists() else "w"
    client = None if args.dry_run else get_client()
    image_index = build_image_index(roots) if args.recursive_image_search else None
    if image_index is not None:
        print(f"Indexed {len(image_index):,} image lookup keys under image roots.")

    processed = 0
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        for row in selected_rows:
            row_id = row.get("id", "")
            if row_id in done_ids:
                continue

            enriched = enrich_row(row, client, args, roots, image_index)
            writer.writerow(enriched)
            handle.flush()
            processed += 1

            status = enriched.get("prompt_review_status", "")
            error = enriched.get("vlm_error", "")
            print(f"[{processed}/{len(selected_rows)}] {row_id} {status} {error}")

            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

    print(f"Wrote {processed:,} enriched rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
