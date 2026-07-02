#!/usr/bin/env python3
"""
Label ImgEdit rows with a local Qwen VL model for ChordEdit.

Input:
  A PIE-style CSV containing source_image, target_image, and editing_instruction.

Output:
  A CSV with ChordEdit-ready source_prompt and target_prompt, plus extra
  foreground/edit fields for filtering and later CLIP scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import tempfile
from pathlib import Path
from typing import Any


LABEL_COLUMNS = [
    "source_prompt",
    "target_prompt",
    "foreground",
    "foreground_target",
    "qwen_edit_type",
    "vlm_confidence",
    "vlm_notes",
    "vlm_model",
    "vlm_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use local Qwen VL to label ImgEdit rows for ChordEdit."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        help="Root folder for resolving source/target image paths.",
    )
    parser.add_argument(
        "--recursive-image-search",
        action="store_true",
        help="Index images recursively under image roots.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help=(
            "Use auto for GPU/device_map inference, or cpu to avoid CUDA kernel "
            "compatibility errors."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype. Use float32 with --device cpu.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa"],
        default="eager",
        help="Use eager for maximum compatibility; sdpa may be faster on newer GPUs.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=768,
        help="Resize each image so its longest side is at most this many pixels.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help=(
            "Use only source_image plus editing_instruction. Do not require "
            "target_image. Useful for ImgEdit-Bench rows."
        ),
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
    for column in LABEL_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def normalize_path(value: str) -> str:
    return value.strip().strip('"').replace("\\", "/")


def build_image_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
            try:
                index.setdefault(path.relative_to(root).as_posix(), path)
            except ValueError:
                pass
    return index


def resolve_image_path(
    value: str, roots: list[Path], image_index: dict[str, Path] | None
) -> Path | None:
    if not value:
        return None
    normalized = normalize_path(value)
    raw = Path(normalized)
    if raw.is_absolute() and raw.exists():
        return raw

    for root in roots:
        candidates = [
            root / normalized,
            root / normalized.lstrip("/"),
            root / Path(normalized).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    if image_index:
        path = Path(normalized)
        for key in (normalized, path.name, path.stem):
            if key in image_index:
                return image_index[key]
    return None


def build_label_prompt(row: dict[str, str], source_only: bool) -> str:
    instruction = row.get("editing_instruction", "")
    edit_type = row.get("edit_type", "")

    target_context = (
        "You will receive one source image and must infer the intended target "
        "prompt from the editing instruction."
        if source_only
        else "You will receive a source image before editing and a target image after editing."
    )

    return f"""
You are labeling an image-editing dataset for ChordEdit.

{target_context}

ImgEdit edit_type: {edit_type}
ImgEdit editing instruction: {instruction}

Return strict JSON only, with exactly these keys:
{{
  "source_prompt": "...",
  "target_prompt": "...",
  "foreground": "...",
  "foreground_target": "...",
  "edit_type": "add|remove|replace|action|style|background|adjust|other",
  "vlm_confidence": "high|medium|low",
  "notes": "..."
}}

Requirements:
- source_prompt must be one natural sentence describing the original scene.
- target_prompt must be one natural sentence describing the edited scene.
- The prompts must be usable directly as ChordEdit source/target prompts.
- Keep shared scene context consistent between source_prompt and target_prompt.
- Make the actual edit difference clear.
- Do not use the phrases "source image", "target image", "before", or "after".
- foreground is the object, person, attribute, action, or region being edited.
- foreground_target is what it becomes, or the new action/state.
- If the images and instruction disagree, trust the images and set vlm_confidence to low.
- Keep both prompts concise but specific.
""".strip()


def prepare_image_for_qwen(path: Path, max_image_side: int) -> str:
    if max_image_side <= 0:
        return str(path)

    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_image_side:
        return str(path)

    scale = max_image_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    tmp_dir = Path(tempfile.gettempdir()) / "imgedit_qwen_resized"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_dir / f"{path.stem}_{max_image_side}.jpg"
    image.save(output_path, format="JPEG", quality=90)
    return str(output_path)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))

        if text.startswith("{"):
            salvage: dict[str, Any] = {}
            for key in (
                "source_prompt",
                "target_prompt",
                "foreground",
                "foreground_target",
                "edit_type",
                "vlm_confidence",
                "notes",
            ):
                key_match = re.search(
                    rf'"{re.escape(key)}"\s*:\s*"(?P<value>[^"]*)',
                    text,
                    flags=re.DOTALL,
                )
                if key_match:
                    salvage[key] = key_match.group("value").replace("\n", " ").strip()
            if salvage.get("source_prompt") and salvage.get("target_prompt"):
                salvage.setdefault("vlm_confidence", "medium")
                salvage.setdefault("notes", "Recovered from truncated JSON.")
                return salvage

        raise ValueError(f"No JSON object found in model output: {text[:500]}")


def dtype_from_arg(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def load_qwen(
    model_id: str,
    device: str,
    dtype_name: str,
    attn_implementation: str,
) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing Qwen dependencies. Install transformers, torch, accelerate, "
            "pillow, and qwen-vl-utils first."
        ) from exc

    dtype = dtype_from_arg(dtype_name)
    common_kwargs = {
        "dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if device == "auto":
        common_kwargs["device_map"] = "auto"

    try:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            **common_kwargs,
        )
    except ImportError:
        from transformers import AutoModelForMultimodalLM

        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            **common_kwargs,
        )
    if device == "cpu":
        model = model.to("cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def model_input_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    return "cpu"


def run_qwen(
    model: Any,
    processor: Any,
    row: dict[str, str],
    source_path: Path,
    target_path: Path | None,
    max_new_tokens: int,
    source_only: bool,
    max_image_side: int,
) -> dict[str, Any]:
    import torch

    content: list[dict[str, Any]] = [
        {"type": "image", "image": prepare_image_for_qwen(source_path, max_image_side)},
    ]
    if target_path is not None and not source_only:
        content.append(
            {"type": "image", "image": prepare_image_for_qwen(target_path, max_image_side)}
        )
    content.append({"type": "text", "text": build_label_prompt(row, source_only)})

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    try:
        inputs = inputs.to(model.device)
    except AttributeError:
        inputs = inputs.to(model_input_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return extract_json(output_text)


def label_row(
    row: dict[str, str],
    model: Any,
    processor: Any,
    roots: list[Path],
    image_index: dict[str, Path] | None,
    args: argparse.Namespace,
) -> dict[str, str]:
    labeled = dict(row)
    for column in LABEL_COLUMNS:
        labeled.setdefault(column, "")

    source_path = resolve_image_path(row.get("source_image", ""), roots, image_index)
    target_path = resolve_image_path(row.get("target_image", ""), roots, image_index)

    if source_path is None or (target_path is None and not args.source_only):
        missing = []
        if source_path is None:
            missing.append("source_image")
        if target_path is None and not args.source_only:
            missing.append("target_image")
        labeled["vlm_error"] = "missing local " + " and ".join(missing)
        labeled["vlm_confidence"] = "low"
        return labeled

    try:
        result = run_qwen(
            model=model,
            processor=processor,
            row=row,
            source_path=source_path,
            target_path=target_path,
            max_new_tokens=args.max_new_tokens,
            source_only=args.source_only,
            max_image_side=args.max_image_side,
        )
        labeled["source_prompt"] = str(result.get("source_prompt", "")).strip()
        labeled["target_prompt"] = str(result.get("target_prompt", "")).strip()
        labeled["foreground"] = str(result.get("foreground", "")).strip()
        labeled["foreground_target"] = str(result.get("foreground_target", "")).strip()
        labeled["qwen_edit_type"] = str(result.get("edit_type", "")).strip()
        labeled["vlm_confidence"] = str(result.get("vlm_confidence", "")).strip()
        labeled["vlm_notes"] = str(result.get("notes", "")).strip()
        labeled["vlm_model"] = args.model
        labeled["vlm_error"] = ""
    except Exception as exc:
        labeled["vlm_error"] = f"{type(exc).__name__}: {exc}"
        labeled["vlm_confidence"] = "low"
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return labeled


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    roots = [Path(root) for root in args.image_root]

    rows = read_csv_rows(input_path)
    selected_rows = rows[args.start_index : args.start_index + args.limit]
    if not selected_rows:
        print("No rows selected.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = existing_ids(output_path) if args.resume else set()
    mode = "a" if args.resume and output_path.exists() else "w"
    columns = output_columns(list(selected_rows[0].keys()))

    image_index = build_image_index(roots) if args.recursive_image_search else None
    if image_index is not None:
        print(f"Indexed {len(image_index):,} image lookup keys under image roots.")

    print(
        f"Loading {args.model} on device={args.device}, "
        f"dtype={args.dtype}, attn={args.attn_implementation}..."
    )
    model, processor = load_qwen(
        args.model,
        device=args.device,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    print("Model loaded.")

    processed = 0
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        for row in selected_rows:
            row_id = row.get("id", "")
            if row_id in done_ids:
                continue
            print(f"Starting {row_id}...")
            started = time.time()
            labeled = label_row(row, model, processor, roots, image_index, args)
            writer.writerow(labeled)
            handle.flush()
            processed += 1
            elapsed = time.time() - started
            status = labeled.get("vlm_confidence", "")
            error = labeled.get("vlm_error", "")
            print(f"[{processed}/{len(selected_rows)}] {row_id} {status} {elapsed:.1f}s {error}")

    print(f"Wrote {processed:,} labeled rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
