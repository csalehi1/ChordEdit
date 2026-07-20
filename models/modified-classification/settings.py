# settings.py

from functools import partial
from pathlib import Path

from scores import weighted_combined_score


"""
Dataset settings.
"""

# NOTE: Set this to the directory containing the generated metrics and inputs CSV files.
DIR_NAME = "UltraEdit_Region_1000"
GENERATED_DIR = Path(f"/shared/ssd_30T/mirick/generated/ultra_edit/{DIR_NAME}")
DATASET_DIR = Path(f"/shared/ssd_30T/mirick/datasets/ultra_edit/{DIR_NAME}")
# Precomputed embeddings (used when FREEZE_ENCODERS is True). 
# Set to None for on-the-fly encode. Will pack into cache dir.
EMBEDDINGS_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/ultraedit/{DIR_NAME}")
EMBEDDINGS_SAMPLES_DIRNAME = "annotation_embeddings"

INPUTS_CSV = GENERATED_DIR / f"id_to_inputs_{DIR_NAME.replace('_', '').lower()}.csv"
METRICS_CSV = GENERATED_DIR / f"id_to_metrics_{DIR_NAME.replace('_', '').lower()}.csv"
EMBEDDINGS_CSV = EMBEDDINGS_DIR / f"id_to_embeddings_{DIR_NAME.replace('_', '').lower()}.csv"

# NOTE: Set this to the directory where the model outputs will be saved.
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs" / DIR_NAME
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
_outputs_gitignore = OUTPUTS_DIR.parent / ".gitignore"
if not _outputs_gitignore.exists():
    _outputs_gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

# Training-ready pooled/flattened embedding tables (built on first load from annotation_embeddings/).
M_EMBEDDINGS_PATH = OUTPUTS_DIR.parent / ".cache" / "embeddings" / f"m_{DIR_NAME.replace('_', '').lower()}.pt"

# Shared column name for sample IDs.
SAMPLE_ID_COL = "sample_id"

# Column names in the id_to_inputs_*.csv file.
SOURCE_PROMPT_COL = "source_prompt"
TARGET_PROMPT_COL = "target_prompt"
IMAGE_PATH_COL = "image_path"
MASK_PATH_COL = "mask_image_path"

# Column names in the id_to_embeddings_*.csv file.
SOURCE_EMB_COL = "source_embedding"
TARGET_EMB_COL = "target_embedding"
IMAGE_EMB_COL = "image_embedding"
MASK_EMB_COL = "mask_embedding"

# Column names in the id_to_metrics_*.csv file.
CATEGORY_COL = "category"
T_START_COL = "t_start"
T_END_COL = "t_end"
T_DELTA_COL = "t_delta"
PSNR_COL = "psnr_unedit_part"
LPIPS_COL = "lpips_unedit_part"
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
M_TARGET_LABELS = {PSNR_COL: "PSNR-Unedited", CLIP_COL: "CLIP-Edited"}

# ChordEdit encoders loaded from SD-TURBO_ROOT for image/text embedding.
# When FREEZE_ENCODERS is True and EMBEDDINGS_CSV is set, embeddings are loaded from disk.
# Set FREEZE_ENCODERS=False or EMBEDDINGS_CSV=None to encode on the fly instead.
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
MLP_CLIP_DROPOUT = 0.2
T_FOURIER_FREQS = 8
T_PROJ_DIM = 128

# M training hyperparameters.
EPOCHS = 10
BATCH_SIZE = 64
# chunk size for batched VAE/text embedding precompute (peak VRAM vs throughput).
EMBED_BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 0.05
NORMALIZE_TARGETS = True
RANKING_LOSS_WEIGHT = 0.3


"""
T model settings.

    Model architecture:
    T(img, src_prompt, tar_prompt) -> (t_start, t_end)
"""

# Scalar objective for timestep selection and M ranking loss.
# Bind hyperparameters with partial, e.g. partial(softplus_score, alpha=1.0, beta=2.0).
T_TARGET_SCORE = partial(weighted_combined_score, weights=None, normalize=True)
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
