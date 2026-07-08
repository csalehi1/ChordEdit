"""
Shared helpers for generate_grid.py and label_grid.py.

Covers filesystem helpers, dataset (mapping_file.json) loading, mask decoding,
and building the ChordEditPipeline. Copied/condensed from scripts/run_local_ablation.py
and scripts/daniel_run_pie_grid_pnp_metrics.py so daniel_create/ is self-contained.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

# ChordEditPipeline + pipeline_chord live at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import settings

if TYPE_CHECKING:
    import torch
    from PIL import Image

    from pipeline_chord import ChordEditPipeline


@dataclass(frozen=True)
class LocalRecord:
    """One edit request: source image + source/target prompts."""

    sample_name: str
    image_path: Path
    source_prompt: str
    target_prompt: str
    edit_prompt: str
    edit_id: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def param_slug(param_name: str, value: float) -> str:
    """Cell filename component, e.g. param_slug('t_start', 0.9) -> 't_start_0p9'."""
    return f"{param_name}_{value:.1f}".replace(".", "p")


def cell_filename(t_start: float, t_end: float) -> str:
    return f"{param_slug('t_start', t_start)}__{param_slug('t_end', t_end)}{settings.CELL_EXTENSION}"


def format_sample_id(sample_id: str, width: int = settings.SAMPLE_ID_WIDTH) -> str:
    return str(sample_id).zfill(width)


def relative_cell_path(sample_id: str, t_start: float, t_end: float) -> str:
    """Cell path relative to the metrics CSV (same directory as sample folders)."""
    sid = format_sample_id(sample_id)
    return f"{sid}/cells/{cell_filename(t_start, t_end)}"


def dtype_from_precision(value: Optional[str]) -> "torch.dtype":
    import torch

    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    precision = (value or "fp32").lower()
    if precision not in mapping:
        raise ValueError(f"Unsupported precision '{value}'. Choose from {list(mapping)}.")
    return mapping[precision]


def resolve_component_paths(model_root: str | Path, model_type: str = "auto") -> Dict[str, str]:
    root = Path(model_root).expanduser().resolve()
    component_paths = {key: str((root / sub).resolve()) for key, sub in settings.SD_COMPONENT_SUBDIRS.items()}
    sdxl_paths = {key: (root / sub).resolve() for key, sub in settings.SDXL_COMPONENT_SUBDIRS.items()}
    if model_type == "sdxl" or (model_type == "auto" and all(p.exists() for p in sdxl_paths.values())):
        component_paths.update({key: str(p) for key, p in sdxl_paths.items()})
    return component_paths


def strip_brackets(text: str) -> str:
    """PIE / UltraEdit mark edited words with [ ]; drop the markers for prompting."""
    return text.replace("[", "").replace("]", "").strip()


def resolve_under(root: Path, rel: str) -> Path:
    """A mapping path may be relative to root or root/annotation_images."""
    direct = root / rel
    return direct if direct.exists() else root / "annotation_images" / rel


def load_samples(
    mapping_path: Path,
    max_samples: Optional[int] = None,
    shard: int = 0,
    num_shards: int = 1,
) -> List[Tuple[str, dict]]:
    """Return [(sample_id, meta), ...] for this shard's round-robin slice."""
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    sample_ids = [sid for sid in sorted(mapping) if mapping[sid].get(settings.FIELD_IMAGE_PATH)]
    sample_ids = sample_ids[shard::num_shards]
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    return [(sid, mapping[sid]) for sid in sample_ids]


def _id_to_csv_name(kind: str, output_root: Path) -> str:
    return f"id_to_{kind}_{settings.output_root_suffix(output_root)}.csv"


def write_id_to_inputs(output_root: Path, data_root: Path, mapping_path: Path) -> Path:
    """Write sample_id -> prompts + dataset image/mask paths (under data_root)."""
    import csv

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    dest = output_root / _id_to_csv_name("inputs", output_root)
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
                    "sample_id": format_sample_id(sample_id),
                    "source_prompt": strip_brackets(meta.get(settings.FIELD_SOURCE_PROMPT, "")),
                    "target_prompt": strip_brackets(meta.get(settings.FIELD_TARGET_PROMPT, "")),
                    "image_path": str(resolve_under(data_root, image_rel)),
                    "mask_image_path": str(resolve_under(data_root, mask_rel)) if mask_rel else "",
                }
            )
    return dest


def _read_result_rows(output_root: Path) -> List[dict]:
    """Load metric rows from result.csv or, if absent, all result_shard*.csv files."""
    import csv

    merged = output_root / settings.CSV_NAME
    sources = [merged] if merged.exists() else sorted(output_root.glob("result_shard*.csv"))
    rows: List[dict] = []
    for path in sources:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_id_to_metrics(output_root: Path) -> Optional[Path]:
    """
    Write id_to_metrics_<suffix>.csv from result.csv (or shard CSVs).

    Columns: sample_id, t_start, t_end, t_delta, whole_psnr, clip_edited, cell_path.
    Sorted by sample_id, t_start, t_end; cell_path is relative to the CSV.
    """
    import csv

    rows = _read_result_rows(output_root)
    if not rows:
        return None

    out_rows = []
    for row in rows:
        sample_id = format_sample_id(row["sample_id"])
        t_start = float(row["t_start"])
        t_end = float(row["t_end"])
        # Accept either the published column names or legacy result.csv names.
        whole_psnr = row.get("whole_psnr", row.get("psnr"))
        clip_edited = row.get("clip_edited", row.get("clip_similarity_target_image_edit_part"))
        out_rows.append(
            {
                "sample_id": sample_id,
                "t_start": t_start,
                "t_end": t_end,
                "t_delta": row["t_delta"],
                "whole_psnr": whole_psnr,
                "clip_edited": clip_edited,
                "cell_path": relative_cell_path(sample_id, t_start, t_end),
            }
        )

    out_rows.sort(key=lambda row: (row["sample_id"], float(row["t_start"]), float(row["t_end"])))

    dest = output_root / _id_to_csv_name("metrics", output_root)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=settings.ID_TO_METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    return dest


def mask_decode(encoded_mask, image_shape: Tuple[int, int]) -> np.ndarray:
    """PIE-Bench run-length mask -> HxW {0,1} (border forced to 1)."""
    length = image_shape[0] * image_shape[1]
    mask = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded_mask), 2):
        start = encoded_mask[i]
        run = min(encoded_mask[i + 1], length - start)
        mask[start : start + run] = 1.0
    mask = mask.reshape(image_shape)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 1
    return mask


def load_mask(root: Path, meta: dict, image_size: int = settings.IMAGE_SIZE) -> np.ndarray:
    """HxWx3 {0,1} edit mask from a jpg mask image or a run-length mask."""
    from PIL import Image

    mask_rel = meta.get(settings.FIELD_MASK_IMAGE_PATH)
    if mask_rel:
        with Image.open(resolve_under(root, mask_rel)) as mimg:
            gray = mimg.convert("L").resize((image_size, image_size))
        mask = (np.array(gray) > 127).astype(np.float32)
    elif meta.get(settings.FIELD_MASK):
        mask = mask_decode(meta[settings.FIELD_MASK], (image_size, image_size))
    else:
        mask = np.ones((image_size, image_size), dtype=np.float32)
    return mask[:, :, np.newaxis].repeat(3, axis=2)


def load_pipeline(
    model_root: str,
    device: str,
    chord_edit_mode: str = settings.CHORD_EDIT_MODE,
    base_config: Optional[Dict[str, Any]] = None,
    image_size: int = settings.IMAGE_SIZE,
) -> "ChordEditPipeline":
    """Load the fp32 SD ChordEditPipeline used for grid generation."""
    import torch

    from pipeline_chord import ChordEditPipeline

    if base_config is None:
        base_config = settings.base_edit_config(chord_edit_mode)

    return ChordEditPipeline.from_local_weights(
        component_paths=resolve_component_paths(model_root, "sd"),
        model_type="sd",
        default_edit_config=base_config,
        device=device,
        torch_dtype=dtype_from_precision("fp32"),
        image_size=image_size,
        use_center_crop=True,
        compute_dtype=torch.float32,
        use_attention_mask=False,
        use_safety_checker=False,
        chord_edit_mode=chord_edit_mode,
    )
