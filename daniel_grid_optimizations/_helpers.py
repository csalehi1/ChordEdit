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
    edit_instruction: str
    sample_id: str


def cell_filename(t_start: float, t_end: float) -> str:
    """e.g. t_start_0p9__t_end_0p3.jpg (or .png when JPEG_QUALITY is None)."""
    start = f"t_start_{t_start:.1f}".replace(".", "p")
    end = f"t_end_{t_end:.1f}".replace(".", "p")
    return f"{start}__{end}{settings.CELL_EXTENSION}"


def metrics_fieldnames(metrics: List[str]) -> List[str]:
    """Header for id_to_metrics CSV (metric columns may be a subset)."""
    return ["sample_id", "t_start", "t_end", "t_delta", *metrics, "cell_path"]


def strip_brackets(text: str) -> str:
    """Drop [bracket] markers from prompts."""
    return text.replace("[", "").replace("]", "").strip()


def resolve_under(root: Path, relative: str) -> Path:
    """Resolve a mapping path under root (absolute paths pass through)."""
    path = Path(relative)
    if path.is_absolute():
        return path
    direct = root / relative
    return direct if direct.exists() else root / "annotation_images" / relative


def validate_dataset_root(data_root: Path) -> Path:
    """
    Require mapping_file.json, annotation_images/, annotation_masks/.
    Optional when present: annotation_masks_downloaded/, annotation_edits/.
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"data-root is not a directory: {data_root}")
    mapping_path = data_root / settings.MAPPING_FILENAME
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing {settings.MAPPING_FILENAME} under {data_root}")
    missing = [name for name in settings.DATASET_REQUIRED_SUBDIRS if not (data_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"data-root missing required folders {missing}: {data_root}"
        )
    return mapping_path


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


def write_id_to_embeddings(embeddings_root: Path, mapping_path: Path) -> Path:
    """Write id_to_embeddings_<suffix>.csv with absolute paths to per-sample .pt files."""
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    suffix = embeddings_root.name.lower().replace("_", "").replace("-", "")
    dest = embeddings_root / f"id_to_embeddings_{suffix}.csv"
    embeddings_root.mkdir(parents=True, exist_ok=True)

    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=settings.ID_TO_EMBEDDINGS_FIELDS)
        writer.writeheader()
        for sample_id in sorted(mapping):
            meta = mapping[sample_id]
            if not meta.get(settings.FIELD_IMAGE_PATH):
                continue
            sid = str(sample_id).zfill(settings.SAMPLE_ID_WIDTH)
            sample_dir = embeddings_root / settings.SAMPLES_DIRNAME / sid
            writer.writerow(
                {
                    "sample_id": sid,
                    "source_embedding": str(sample_dir / "source.pt"),
                    "target_embedding": str(sample_dir / "target.pt"),
                    "image_embedding": str(sample_dir / "image.pt"),
                    "mask_embedding": str(sample_dir / "mask.pt"),
                }
            )
    return dest


def write_id_to_inputs(generated_root: Path, data_root: Path, mapping_path: Path) -> Path:
    """
    Write id_to_inputs_<suffix>.csv with absolute image/mask paths.
    downloaded_mask_image_path is filled only when that optional folder/field is present.
    """
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    suffix = generated_root.name.lower().replace("_", "").replace("-", "")
    dest = generated_root / f"id_to_inputs_{suffix}.csv"
    has_downloaded_masks = (data_root / "annotation_masks_downloaded").is_dir()

    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=settings.ID_TO_INPUTS_FIELDS)
        writer.writeheader()
        for sample_id in sorted(mapping):
            meta = mapping[sample_id]
            image_rel = meta.get(settings.FIELD_IMAGE_PATH)
            if not image_rel:
                continue
            mask_rel = meta.get(settings.FIELD_MASK_IMAGE_PATH, "")
            downloaded_mask_rel = (
                meta.get(settings.FIELD_DOWNLOADED_MASK_IMAGE_PATH, "")
                if has_downloaded_masks
                else ""
            )
            writer.writerow(
                {
                    "sample_id": str(sample_id).zfill(settings.SAMPLE_ID_WIDTH),
                    "source_prompt": meta.get(settings.FIELD_SOURCE_PROMPT, ""),
                    "target_prompt": meta.get(settings.FIELD_TARGET_PROMPT, ""),
                    "image_path": str(resolve_under(data_root, image_rel)),
                    "mask_image_path": str(resolve_under(data_root, mask_rel)) if mask_rel else "",
                    "downloaded_mask_image_path": (
                        str(resolve_under(data_root, downloaded_mask_rel)) if downloaded_mask_rel else ""
                    ),
                }
            )
    return dest


def iter_cell_pairs(
    t_start_values: list[float],
    t_end_values: list[float] | None = None,
    *,
    diagonal_optimization: bool,
    t_delta: float = 0.0,
) -> Iterable[Tuple[float, float]]:
    """Yield (t_start, t_end) pairs to generate. Defaults to a square grid."""
    ends = t_start_values if t_end_values is None else t_end_values
    for t_start in t_start_values:
        if t_start - t_delta < 0:
            continue
        for t_end in ends:
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
