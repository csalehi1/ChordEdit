"""
_helpers.py

Shared helpers for daniel_create.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Repo root so pipeline_chord imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import settings


@dataclass(frozen=True)
class SampleRecord:
    """
    One edit request handed to the factorized grid.
    
    Inspired by paper's utils.py:LocalEditDataset.
    """

    sample_name: str
    image_path: Path
    source_prompt: str
    target_prompt: str
    edit_prompt: str
    sample_id: str


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
    """
    Return [(sample_id, meta), ...] for this shard's round-robin slice.

    When max_samples is set, the global list is capped first, then split across shards.
    """
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    sample_ids = [sid for sid in sorted(mapping) if mapping[sid].get(settings.FIELD_IMAGE_PATH)]
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    sample_ids = sample_ids[shard::num_shards]
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
                    "source_prompt": meta.get(settings.FIELD_SOURCE_PROMPT, ""),
                    "target_prompt": meta.get(settings.FIELD_TARGET_PROMPT, ""),
                    "image_path": str(resolve_under(data_root, image_rel)),
                    "mask_image_path": str(resolve_under(data_root, mask_rel)) if mask_rel else "",
                }
            )
    return dest


def get_output_dir_name(dataset_name: str, max_samples: int | None) -> str:
    """Dataset folder name under the output root; appends _n<max_samples> when capped."""
    if max_samples is not None:
        return f"{dataset_name}_n{max_samples}"
    return dataset_name


def iter_cell_pairs(
    grid_values: list[float],
    *,
    diagonal_optimization: bool,
) -> Iterable[Tuple[float, float]]:
    """Yield (t_start, t_end) pairs to generate."""
    for t_start in grid_values:
        for t_end in grid_values:
            # Support for diagonal optimization, exclude t_start <= t_end.
            if diagonal_optimization and t_start <= t_end:
                continue
            yield t_start, t_end


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
