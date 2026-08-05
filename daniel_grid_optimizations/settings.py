"""
Shared settings for daniel_create.

Expected dataset layout under --data-root:

  <data-root>/
    mapping_file.json
    annotation_images/{id}.*
    annotation_masks/{id}.*
    annotation_masks_downloaded/{id}.*   # optional
    annotation_edits/{id}.*              # optional

Expected embeddings layout under --embeddings-root/<dataset-name>/:

  id_to_embeddings_<suffix>.csv
  annotation_embeddings/{id}/source.pt   # pooled CLIP vector, (1024,) float32
  annotation_embeddings/{id}/target.pt   # pooled CLIP vector, (1024,) float32
  annotation_embeddings/{id}/image.pt    # flat VAE latent, (16384,) float32
  annotation_embeddings/{id}/mask.pt     # flat VAE latent of the RGB mask, only with --cache-masks

All embedding files are packing-ready: one flat float32 vector per file,
derived from the same pipeline tensors that condition generation.

Expected generated layout under --generated-root/<dataset-name>/:

  id_to_inputs_<suffix>.csv
  id_to_metrics_<suffix>.csv
  grids/{id}/cells/t_start_{X}p{Y}__t_end_{A}p{B}.jpg
"""

from __future__ import annotations
from typing import List, Tuple


# mapping_file.json field names.
FIELD_IMAGE_PATH = "image_path"
FIELD_EDITED_IMAGE_PATH = "edited_image_path"
FIELD_SOURCE_PROMPT = "original_prompt"
FIELD_TARGET_PROMPT = "editing_prompt"
FIELD_EDIT_INSTRUCTION = "editing_instruction"
FIELD_MASK_IMAGE_PATH = "mask_image_path"
FIELD_DOWNLOADED_MASK_IMAGE_PATH = "downloaded_mask_image_path"

# Subdirs under --data-root.
DATASET_REQUIRED_SUBDIRS: Tuple[str, ...] = (
    "annotation_images",
    "annotation_masks",
)
DATASET_OPTIONAL_SUBDIRS: Tuple[str, ...] = (
    "annotation_masks_downloaded",
    "annotation_edits",
)
MAPPING_FILENAME = "mapping_file.json"

# Per-sample generated layout under the output dataset folder.
GRIDS_DIRNAME = "grids"
CELLS_DIRNAME = "cells"

# Per-sample embedding layout under --embeddings-root/<dataset-name>/.
SAMPLES_DIRNAME = "annotation_embeddings"
EMBEDDING_FILENAMES: Tuple[str, ...] = ("source.pt", "target.pt", "image.pt")
# Written per sample only with --cache-masks (and only when the sample has a mask).
MASK_FILENAME = "mask.pt"

# CSV schemas.
SAMPLE_ID_WIDTH = 8
ID_TO_EMBEDDINGS_FIELDS = [
    "sample_id",
    "source_embedding",
    "target_embedding",
    "image_embedding",
]
ID_TO_INPUTS_FIELDS = [
    "sample_id",
    "source_prompt",
    "target_prompt",
    "image_path",
    "mask_image_path",
    "downloaded_mask_image_path",
]
ID_TO_METRICS_FIELDS = [
    "sample_id",
    "t_start",
    "t_end",
    "t_delta",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "clip_similarity_target_image_edit_part",
    "cell_path",
]

# Generation sweep (default ChordEdit mode only).
IMAGE_SIZE = 512
T_DELTA = 0.15
GRID_VALUES: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# JPEG quality for generated images to save space.
# Set to None to write lossless PNG at native resolution.
JPEG_QUALITY: int | None = 90
CELL_EXTENSION = ".jpg" if JPEG_QUALITY is not None else ".png"

# Grid appearance settings.
THUMB_PX = 128
GAP = 3
FIG_DPI = 100
