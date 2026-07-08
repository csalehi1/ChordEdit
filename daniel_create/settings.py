"""Shared knobs for daniel_create (dataset fields, grid, CSVs, plot style).

Generate-only constants (model root, seed, edit configs, SD component layout)
live at the top of generate_grid.py. Cell encode format is settings.JPEG_QUALITY
(None → .png, else JPEG at that quality).
"""

from __future__ import annotations

from typing import List

# Dataset default. Output always goes to daniel_create/generated/<Path(data_root).name>/.
DEFAULT_DATA_ROOT = "/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_1000"

# mapping_file.json field names (UltraEdit-style defaults).
FIELD_IMAGE_PATH = "image_path"
FIELD_SOURCE_PROMPT = "original_prompt"
FIELD_TARGET_PROMPT = "editing_prompt"
FIELD_EDIT_INSTRUCTION = "editing_instruction"
FIELD_MASK_IMAGE_PATH = "mask_image_path"
FIELD_MASK = "mask"

# CSV schemas. Filename suffix = output folder stem, lowercased, underscores stripped
# (UltraEdit_Region_1000 -> ultraeditregion1000).
SAMPLE_ID_WIDTH = 8
ID_TO_INPUTS_FIELDS = ["sample_id", "source_prompt", "target_prompt", "image_path", "mask_image_path"]
ID_TO_METRICS_FIELDS = ["sample_id", "t_start", "t_end", "t_delta", "whole_psnr", "clip_edited", "cell_path"]

# Shared generation/labeling sweep (default ChordEdit mode only).
IMAGE_SIZE = 512
T_DELTA = 0.0
GRID_VALUES: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Cells + metrics.
# None → lossless PNG cells; otherwise JPEG at this quality.
JPEG_QUALITY: int | None = 90
CELL_EXTENSION = ".jpg" if JPEG_QUALITY is not None else ".png"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
# Intermediate resume CSV; published artifact is id_to_metrics_<suffix>.csv.
CSV_NAME = "result.csv"
CSV_FIELDS = list(ID_TO_METRICS_FIELDS)

# Overview-grid appearance.
THUMB_PX = 128
GAP = 3
FIG_DPI = 100
OVERLAY_ALPHA = 0.5
CMAP = "viridis"
