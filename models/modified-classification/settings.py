"""
Configuration for the metric-predictor model.

Surrogate for editing metrics: given the source image, the prompt pair, and a
candidate (t_start, t_end), it predicts the resulting PSNR and CLIP scores.

    M(img_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)
"""

from pathlib import Path

import numpy as np

_PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = _PARENT_DIR / "data"

# UltraEdit Region 1000 grid ablation. Each row is one (sample_id, t_start,
# t_end) cell with measured psnr/clip and a path to the rendered cell image.
ULTRA_EDIT_DATA_ROOT = Path("/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_1000")
ULTRA_EDIT_GENERATED_ROOT = Path("/shared/ssd_30T/mirick/generated/ultra_edit/UltraEdit_Region_1000")
METRICS_CSV = ULTRA_EDIT_GENERATED_ROOT / "id_to_metrics_ultraeditregion1000.csv"
STRINGS_CSV = ULTRA_EDIT_GENERATED_ROOT / "id_to_inputs_ultraeditregion1000.csv"

OUTPUTS_SUBDIR = "ultra_edit_region_1000"
OUTPUTS_DIR = _PARENT_DIR / "outputs" / OUTPUTS_SUBDIR
if not OUTPUTS_DIR.exists():
    OUTPUTS_DIR.mkdir(parents=True)


"""
Data columns. These are the computed metric columns in the METRICS_CSV file,
as well as the shared image path column.
"""

PSNR_COL = "psnr"
CLIP_COL = "clip_similarity_target_image_edit_part"
IMAGE_PATH_COL = "cell_path"

# Targets the model regresses, in order. Output tensor columns follow this list.
TARGET_COLS = ("psnr", "clip")
TARGET_LABELS = {"psnr": "Whole PSNR", "clip": "CLIP-Edited"}
TARGET_METRIC_COL = "-".join(c for c in TARGET_COLS)

# Source images: id_to_inputs.image_path relative to ULTRA_EDIT_DATA_ROOT.
SOURCE_IMAGE_ROOT = ULTRA_EDIT_DATA_ROOT
SOURCE_IMAGE_PATH_COL = "image_path"
CELL_PATH_ROOT = ULTRA_EDIT_GENERATED_ROOT

# Merge key in STRINGS_CSV (sample_id for UltraEdit, id for PIE-Bench exports).
STRINGS_ID_COL = "sample_id"


"""
Data selection. The model signature takes only (t_start, t_end), so a single
t_delta slice is used. Set to None to train across every t_delta in the data.
"""

TARGET_T_DELTA = 0.0

# Baseline timestep bounds from the original ChordEdit paper (used in eval).
DEFAULT_T_START = 0.9
DEFAULT_T_END = 0.3

# Discrete grid axes for T (timestep selector).
GRID_T_START = tuple(round(float(v), 1) for v in np.arange(0, 1.01, 0.1))
GRID_T_END = tuple(round(float(v), 1) for v in np.arange(0, 1.01, 0.1))

# Scalarization weights for collapsing (PSNR, CLIP) -> m in T.
W_PSNR = 0.5
W_CLIP = 0.5
NOISE_FLOOR_M = 0.0


"""
Encoders. A ChordEditPipeline is loaded from SD_TURBO_ROOT and reused for
text and VAE image encoding (same paths and preprocessing as inference).
"""

SD_TURBO_ROOT = Path("/shared/ssd_30T/mirick/models/sd-turbo")
IMAGE_SIZE = 512
USE_CENTER_CROP = True
FREEZE_ENCODERS = True


"""
Model architecture (regressor MLP body widths).

Image/text bottlenecks keep timestep features from being drowned by the 16k VAE
latent. Fourier timestep encoding + FiLM on the CLIP tower sharpen grid surfaces.
"""

IMG_PROJ_DIM = 512
TEXT_PROJ_DIM = 256
MLP_WIDE = 256
MLP_HIDDEN = 128
MLP_INNER = 64
MLP_DROPOUT = 0.2
MLP_CLIP_DROPOUT = 0.0
T_FOURIER_FREQS = 8
T_PROJ_DIM = 128


"""
Training hyperparameters. Targets are standardized (z-scored) using train-split
statistics so PSNR (~9-36) and CLIP (~0.06-0.35) contribute comparably to the
MSE objective; predictions are de-standardized before metrics are reported.
"""

SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.01
NORMALIZE_TARGETS = True

# Sample-level split ratios (by sample_id, not individual grid rows).
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# Within-sample ranking loss (aligns M with T argmax objective).
RANKING_LOSS_WEIGHT = 0.3
