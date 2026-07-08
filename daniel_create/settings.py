"""
Shared settings for daniel_create.
"""

from __future__ import annotations

from typing import List

# Dataset default. Output always goes to daniel_create/generated/<Path(data_root).name>/.
DEFAULT_DATA_ROOT = "/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_1000"

# mapping_file.json field names.
FIELD_IMAGE_PATH = "image_path"
FIELD_SOURCE_PROMPT = "original_prompt"
FIELD_TARGET_PROMPT = "editing_prompt"
FIELD_EDIT_INSTRUCTION = "editing_instruction"
FIELD_MASK_IMAGE_PATH = "mask_image_path"

# CSV schema.
SAMPLE_ID_WIDTH = 8
ID_TO_INPUTS_FIELDS = ["sample_id", "source_prompt", "target_prompt", "image_path", "mask_image_path"]

# Generation sweep (default ChordEdit mode only).
IMAGE_SIZE = 512
T_DELTA = 0.0
GRID_VALUES: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# JPEG quality for generated images to save space. 
# Set to None to write lossless PNG at native resolution.
JPEG_QUALITY: int | None = 90
CELL_EXTENSION = ".jpg" if JPEG_QUALITY is not None else ".png"

# Grid appearance settings.
THUMB_PX = 128
GAP = 3
FIG_DPI = 100
