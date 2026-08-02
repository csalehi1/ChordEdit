from functools import partial
from pathlib import Path

import numpy as np

from models.classification.scores import linex_score, score_df


"""
Data settings.
"""

# NOTE: Set this to the directory containing the generated metrics and inputs CSV files.
DIR_NAME = "UltraEdit_Region_10"
GENERATED_DIR = Path(f"/shared/ssd_30T/mirick/generated/ultra_edit/{DIR_NAME}")
DATASET_DIR = Path(f"/shared/ssd_30T/mirick/datasets/ultra_edit/{DIR_NAME}")
# Precomputed embeddings (used when FREEZE_ENCODERS is True). Set to None to force on-the-fly encode.
# EMBEDDINGS_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/ultraedit/annotation_embeddings/{DIR_NAME}")

INPUTS_CSV = GENERATED_DIR / f"id_to_inputs_{DIR_NAME.replace('_', '').lower()}.csv"
METRICS_CSV = GENERATED_DIR / f"id_to_metrics_{DIR_NAME.replace('_', '').lower()}.csv"

# NOTE: Set this to the directory where the model outputs will be saved.
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs" / DIR_NAME
if not OUTPUTS_DIR.exists():
    OUTPUTS_DIR.mkdir(parents=True)

# Shared column name for sample IDs.
SAMPLE_ID_COL = "sample_id"

# Column names in the id_to_inputs_*.csv file.
SOURCE_PROMPT_COL = "source_prompt"
TARGET_PROMPT_COL = "target_prompt"
IMAGE_PATH_COL = "image_path"
MASK_PATH_COL = "mask_image_path"

# Column names in the id_to_metrics_*.csv file.
CATEGORY_COL = "category"
T_START_COL = "t_start"
T_END_COL = "t_end"
T_DELTA_COL = "t_delta"
PSNR_COL = "psnr_unedit_part"
LPIPS_COL = "lpips_unedit_part"
CLIP_COL = "clip_similarity_target_image_edit_part"
CELL_PATH_COL = "cell_path"

C_TARGET_COLS = (PSNR_COL, CLIP_COL)
C_TARGET_LABELS = {PSNR_COL: "Whole PSNR", CLIP_COL: "CLIP-Edited"}

# Baseline timestep bounds from the original ChordEdit paper.
PAPER_T_START = 0.9
PAPER_T_END = 0.3
PAPER_T_DELTA = 0.15

# Account for the paper's finding by using the nearest grid value.
DEFAULT_T_START = 0.8
DEFAULT_T_END = PAPER_T_END

# Value in `t_delta` column to select data from.
TARGET_T_DELTA = 0.0

GRID_VALUES = np.linspace(0.0, 1.0, 11)
GRID_T_START = GRID_VALUES
GRID_T_END = GRID_VALUES

# Scalar objective used to pick one row per sample_id.
# Scores consume per-sample normalized deltas Δ; see scores.calc_normalized_deltas.
C_TARGET_ALPHA = 2.0
_C_SCORE_KW = dict(alpha=C_TARGET_ALPHA)
C_TARGET_SCORE = partial(linex_score, **_C_SCORE_KW)                          # torch φ(Δ)
C_TARGET_SCORE_DF = partial(score_df, score_fn=linex_score, **_C_SCORE_KW)    # (df, *cols) -> Series


def linex_score_df(df):
    """C_TARGET_FUNC with C_TARGET_COLS bound: metrics DataFrame -> pd.Series."""
    return C_TARGET_SCORE_DF(df, *C_TARGET_COLS)


C_TARGET_FUNC = linex_score_df
C_TARGET_COL = "linex_score"
C_TARGET_LABEL = f"LINEX Score ($\\alpha={C_TARGET_ALPHA:g}$)"


"""
Model architecture.
"""

# Name of HuggingFace checkpoint for text-encoder
# TODO: Use "sentence-transformers/Qwen3-VL-Embedding-2B"
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Prevent encoder weights from updating during training
FREEZE_ENCODER = True
# NOTE: Select head type to use for last step of model, 
# may be "CORAL", "MSE", or "CE". Choose one.
HEAD_TYPE = "CE"
# NOTE: Only used when HEAD_TYPE = "CE". May be 
# "cost_sensitive_ce_loss" or "one_hot_ce_loss". Choose one.
CE_LOSS_TYPE = "one_hot_ce_loss"
# Counteract label imbalance in the training split.
USE_CLASS_WEIGHTS = True
# Softens overconfident majority-class collapse in CE training.
LABEL_SMOOTHING = 0.15

"""
Training hyperparameters.
"""

SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
MLP_LR = 1e-4

# Only used when FREEZE_ENCODER is False
ENCODER_LR = 2e-5

# Body of the model
MLP_WIDE = 256
MLP_HIDDEN = 128
MLP_INNER = 64

MLP_DROPOUT = 0.2
WEIGHT_DECAY = 0.05
