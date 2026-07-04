"""Central configuration for the daniel_create grid generation + labeling tools.

Everything tunable (paths, grid resolution, edit configs, metric model, plot
constants) lives here so generate_grid.py and label_grid.py stay thin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


"""
Filesystem roots.
"""

# daniel_create/ -> repo root is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_ROOT = "/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_1000"
DEFAULT_MODEL_ROOT = "/shared/ssd_30T/mirick/models/sd-turbo"
DEFAULT_OUTPUT_ROOT = str(REPO_ROOT / "outputs" / "UltraEdit_Region_1000")


"""
Dataset field mapping: which mapping_file.json keys hold each field we need.
Override these to point at a different dataset's naming; the defaults below match
UltraEdit (source prompt = original_prompt, target prompt = editing_prompt).
"""

FIELD_IMAGE_PATH = "image_path"
FIELD_SOURCE_PROMPT = "original_prompt"
FIELD_TARGET_PROMPT = "editing_prompt"
FIELD_EDIT_INSTRUCTION = "editing_instruction"
FIELD_MASK_IMAGE_PATH = "mask_image_path"
FIELD_MASK = "mask"

# Per-output prompt lookup table written whenever the output folder is created.
ID_TO_PROMPTS_NAME = "id_to_prompts.csv"
ID_TO_PROMPTS_FIELDS: List[str] = [
    "sample_id",
    "source_image_path",
    "source_prompt",
    "target_prompt",
]


"""
Generation settings.
"""

CHORD_EDIT_MODE = "default"  # "default" or "sym"
SEED = 42
IMAGE_SIZE = 512
T_DELTA = 0.0

# t_start / t_end sweep values (grid is GRID_VALUES x GRID_VALUES).
GRID_VALUES: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
GRID_VALUES_SYM: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# app.py edit defaults; n_steps must stay 1 for the factorized fast path.
DEFAULT_EDIT_CONFIG: Dict[str, Any] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.0,
    "step_scale": 1.0,
    "cleanup": True,
}
DEFAULT_EDIT_CONFIG_SYM: Dict[str, Any] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.40,
    "t_end": 0.20,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "cleanup": True,
}


"""
Model component layout (SD / SDXL subfolders under --model-root).
"""

SD_COMPONENT_SUBDIRS: Dict[str, str] = {
    "unet_path": "unet",
    "scheduler_path": "scheduler",
    "text_encoder_path": "text_encoder",
    "tokenizer_path": "tokenizer",
    "vae_path": "vae",
}
SDXL_COMPONENT_SUBDIRS: Dict[str, str] = {
    "text_encoder_2_path": "text_encoder_2",
    "tokenizer_2_path": "tokenizer_2",
}


"""
Cell images on disk.
"""

CELL_EXTENSION = ".jpg"
JPEG_QUALITY = 92


"""
Metrics used for labeling.
"""

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CSV_NAME = "result.csv"
CSV_FIELDS: List[str] = [
    "sample_id",
    "category",
    "t_start",
    "t_end",
    "t_delta",
    "psnr",
    "clip_similarity_target_image_edit_part",
    "cell_path",
]


"""
Grid plot appearance.
"""

THUMB_PX = 128
GAP = 3
FIG_DPI = 100
OVERLAY_ALPHA = 0.5
CMAP = "viridis"


def grid_values(chord_edit_mode: str = CHORD_EDIT_MODE) -> List[float]:
    """t_start/t_end sweep values for the requested edit mode."""
    return list(GRID_VALUES_SYM if chord_edit_mode == "sym" else GRID_VALUES)


def base_edit_config(chord_edit_mode: str = CHORD_EDIT_MODE) -> Dict[str, Any]:
    """A fresh copy of the base edit config for the requested edit mode."""
    return dict(DEFAULT_EDIT_CONFIG_SYM if chord_edit_mode == "sym" else DEFAULT_EDIT_CONFIG)


def require_factorizable_config(base_config: Dict[str, Any]) -> None:
    """The factorized grid only reuses work correctly when n_steps == 1."""
    n_steps = int(base_config.get("n_steps", 1))
    if n_steps != 1:
        raise ValueError(f"Factorized grid requires n_steps=1, got n_steps={n_steps}")
