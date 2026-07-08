"""Shared bits used by generate_grid.py and label_grid.py."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Repo root so pipeline_chord imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import settings


@dataclass(frozen=True)
class LocalRecord:
    """One edit request handed to the factorized grid."""

    sample_name: str
    image_path: Path
    source_prompt: str
    target_prompt: str
    edit_prompt: str
    edit_id: str


def cell_filename(t_start: float, t_end: float) -> str:
    """e.g. t_start_0p9__t_end_0p3.jpg (or .png when JPEG_QUALITY is None)."""
    start = f"t_start_{t_start:.1f}".replace(".", "p")
    end = f"t_end_{t_end:.1f}".replace(".", "p")
    return f"{start}__{end}{settings.CELL_EXTENSION}"


def strip_brackets(text: str) -> str:
    """Drop PIE/UltraEdit [bracket] markers from prompts."""
    return text.replace("[", "").replace("]", "").strip()


def resolve_under(root: Path, relative: str) -> Path:
    """Resolve a mapping path under root, or under root/annotation_images/."""
    direct = root / relative
    return direct if direct.exists() else root / "annotation_images" / relative


def load_samples(
    mapping_path: Path,
    max_samples: Optional[int] = None,
    shard: int = 0,
    num_shards: int = 1,
) -> List[Tuple[str, dict]]:
    """[(sample_id, meta), ...] for this shard's round-robin slice."""
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    sample_ids = [sid for sid in sorted(mapping) if mapping[sid].get(settings.FIELD_IMAGE_PATH)]
    sample_ids = sample_ids[shard::num_shards]
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    return [(sid, mapping[sid]) for sid in sample_ids]


def write_id_to_inputs(output_root: Path, data_root: Path, mapping_path: Path) -> Path:
    """Write id_to_inputs_<suffix>.csv for the whole mapping (all samples)."""
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    suffix = output_root.name.lower().replace("_", "")
    dest = output_root / f"id_to_inputs_{suffix}.csv"

    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=settings.ID_TO_INPUTS_FIELDS)
        writer.writeheader()
        for sample_id in sorted(mapping):
            meta = mapping[sample_id]
            image_rel = meta.get(settings.FIELD_IMAGE_PATH)
            if not image_rel:
                continue
            mask_rel = meta.get(settings.FIELD_MASK_IMAGE_PATH, "")
            writer.writerow(
                {
                    "sample_id": str(sample_id).zfill(settings.SAMPLE_ID_WIDTH),
                    "source_prompt": strip_brackets(meta.get(settings.FIELD_SOURCE_PROMPT, "")),
                    "target_prompt": strip_brackets(meta.get(settings.FIELD_TARGET_PROMPT, "")),
                    "image_path": str(resolve_under(data_root, image_rel)),
                    "mask_image_path": str(resolve_under(data_root, mask_rel)) if mask_rel else "",
                }
            )
    return dest


def write_id_to_metrics(output_root: Path) -> Optional[Path]:
    """Publish id_to_metrics_<suffix>.csv from result.csv or result_shard*.csv."""
    merged = output_root / settings.CSV_NAME
    sources = [merged] if merged.exists() else sorted(output_root.glob("result_shard*.csv"))
    rows: List[dict] = []
    for path in sources:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        return None

    published = []
    for row in rows:
        sample_id = str(row["sample_id"]).zfill(settings.SAMPLE_ID_WIDTH)
        t_start = float(row["t_start"])
        t_end = float(row["t_end"])
        # Accept either published names or older intermediate names.
        whole_psnr = row.get("whole_psnr", row.get("psnr"))
        clip_edited = row.get("clip_edited", row.get("clip_similarity_target_image_edit_part"))
        published.append(
            {
                "sample_id": sample_id,
                "t_start": t_start,
                "t_end": t_end,
                "t_delta": row["t_delta"],
                "whole_psnr": whole_psnr,
                "clip_edited": clip_edited,
                # Relative to the CSV (same folder as the sample dirs).
                "cell_path": f"{sample_id}/cells/{cell_filename(t_start, t_end)}",
            }
        )
    published.sort(key=lambda r: (r["sample_id"], float(r["t_start"]), float(r["t_end"])))

    suffix = output_root.name.lower().replace("_", "")
    dest = output_root / f"id_to_metrics_{suffix}.csv"
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=settings.ID_TO_METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(published)
    return dest


def load_mask(root: Path, meta: dict, image_size: int = settings.IMAGE_SIZE) -> np.ndarray:
    """HxWx3 {0,1} edit mask from a mask image, RLE mask, or all-ones fallback."""
    from PIL import Image

    mask_rel = meta.get(settings.FIELD_MASK_IMAGE_PATH)
    if mask_rel:
        with Image.open(resolve_under(root, mask_rel)) as mask_image:
            gray = mask_image.convert("L").resize((image_size, image_size))
        mask_2d = (np.array(gray) > 127).astype(np.float32)
    elif meta.get(settings.FIELD_MASK):
        # PIE-Bench run-length encoding -> flat then reshape.
        encoded = meta[settings.FIELD_MASK]
        length = image_size * image_size
        flat = np.zeros((length,), dtype=np.float32)
        for i in range(0, len(encoded), 2):
            start = encoded[i]
            run = min(encoded[i + 1], length - start)
            flat[start : start + run] = 1.0
        mask_2d = flat.reshape((image_size, image_size))
        mask_2d[0, :] = mask_2d[-1, :] = mask_2d[:, 0] = mask_2d[:, -1] = 1
    else:
        mask_2d = np.ones((image_size, image_size), dtype=np.float32)
    return mask_2d[:, :, np.newaxis].repeat(3, axis=2)


def load_pipeline(
    model_root: str,
    device: str,
    base_config: Dict[str, Any],
    component_subdirs: Dict[str, str],
) -> Any:
    """Load fp32 SD ChordEditPipeline for grid generation (default edit mode)."""
    import torch
    from pipeline_chord import ChordEditPipeline

    model_path = Path(model_root).expanduser().resolve()
    component_paths = {
        key: str((model_path / sub).resolve()) for key, sub in component_subdirs.items()
    }
    return ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        model_type="sd",
        default_edit_config=base_config,
        device=device,
        torch_dtype=torch.float32,
        image_size=settings.IMAGE_SIZE,
        use_center_crop=True,
        compute_dtype=torch.float32,
        use_attention_mask=False,
        use_safety_checker=False,
        chord_edit_mode="default",
    )

