# settings.py

from pathlib import Path

from scores import compute_weighted_combined_score


"""
Dataset settings.
"""

# NOTE: Set this to the directory containing the generated metrics and strings CSV files.
GENERATED_DIR = Path("/shared/ssd_30T/mirick/generated/ultra_edit/UltraEdit_Region_100")
DATASET_DIR = Path("/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_100")

INPUTS_CSV = GENERATED_DIR / "id_to_inputs_ultraeditregion100.csv"
METRICS_CSV = GENERATED_DIR / "id_to_metrics_ultraeditregion100.csv"

# NOTE: Set this to the directory where the model outputs will be saved.
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs/UltraEdit_Region_100"
if not OUTPUTS_DIR.exists():
    OUTPUTS_DIR.mkdir(parents=True)

# Column names in the id_to_inputs_*.csv file.
SAMPLE_ID_COL = "sample_id"
SOURCE_PROMPT_COL = "source_prompt"
TARGET_PROMPT_COL = "target_prompt"
IMAGE_PATH_COL = "image_path"
MASK_PATH_COL = "mask_image_path"

# Column names in the id_to_metrics_*.csv file.
SAMPLE_ID_COL = "sample_id"
CATEGORY_COL = "category"
T_START_COL = "t_start"
T_END_COL = "t_end"
T_DELTA_COL = "t_delta"
PSNR_COL = "psnr"
CLIP_COL = "clip_similarity_target_image_edit_part"
CELL_PATH_COL = "cell_path"


"""
Shared model settings.
"""

# Values of t_delta column to train models on. Set to `None` to use every t_delta.
TARGET_T_DELTA = 0.0

# Sample-level split ratios (by sample_id, not individual grid rows).
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

SEED = 42


"""
M model settings.

    Model architecture:
    M(img_emb, mask_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)
"""

# Regression targets in the loaded dataframe (after PSNR_COL/CLIP_COL rename).
M_TARGET_COLS = (PSNR_COL, CLIP_COL)
M_TARGET_LABELS = {PSNR_COL: "Whole PSNR", CLIP_COL: "CLIP-Edited"}

# ChordEdit encoders loaded from SD-TURBO_ROOT for image/text embedding.
SD_TURBO_ROOT = Path("/shared/ssd_30T/mirick/models/sd-turbo")
IMAGE_SIZE = 512
USE_CENTER_CROP = True
FREEZE_ENCODERS = True

# Regressor MLP architecture.
IMG_PROJ_DIM = 512
TEXT_PROJ_DIM = 256
MLP_WIDE = 256
MLP_HIDDEN = 128
MLP_INNER = 64
MLP_DROPOUT = 0.2
MLP_CLIP_DROPOUT = 0.0
T_FOURIER_FREQS = 8
T_PROJ_DIM = 128

# M training hyperparameters.
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.01
NORMALIZE_TARGETS = True
RANKING_LOSS_WEIGHT = 0.3


"""
T model settings.

    Model architecture:
    T(img, src_prompt, tar_prompt) -> (t_start, t_end)
"""

# Scalar objective for timestep selection and M ranking loss.
T_TARGET_FUNC = lambda df: compute_weighted_combined_score(df, PSNR_COL, CLIP_COL)
T_TARGET_COL = "combined_score"
T_TARGET_LABEL = "Combined Score"

# Baseline timestep bounds from the original ChordEdit paper (used in eval).
DEFAULT_T_START = 0.9
DEFAULT_T_END = 0.3

# Discrete grid axes for T (timestep selector).
GRID_T_START = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID_T_END = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Deviate-or-default gate: minimum predicted gain to leave baseline timesteps.
NOISE_FLOOR_M = 0.0
