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


def write_id_to_prompts(output_root: Path, data_root: Path, mapping_path: Path) -> Path:
    """
    Write the sample_id -> prompts lookup table for the whole dataset.

    Columns: sample_id, source_image_path, source_prompt, target_prompt. Written
    whenever the output folder is (re)created so every save folder ships with a
    prompt reference. Covers every sample in mapping_file.json (not just a shard).
    """
    import csv

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    dest = output_root / settings.ID_TO_PROMPTS_NAME
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=settings.ID_TO_PROMPTS_FIELDS)
        writer.writeheader()
        for sample_id in sorted(mapping):
            meta = mapping[sample_id]
            image_rel = meta.get(settings.FIELD_IMAGE_PATH)
            if not image_rel:
                continue
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "source_image_path": str(resolve_under(data_root, image_rel)),
                    "source_prompt": strip_brackets(meta.get(settings.FIELD_SOURCE_PROMPT, "")),
                    "target_prompt": strip_brackets(meta.get(settings.FIELD_TARGET_PROMPT, "")),
                }
            )
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
