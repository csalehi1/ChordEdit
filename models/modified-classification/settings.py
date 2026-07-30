# settings.py

from functools import partial
from pathlib import Path

from scores import linex_score, score_df


"""
Dataset settings.
"""

# NOTE: Set this to the directory containing the generated metrics and inputs CSV files.
DIR_NAME_DEFAULT = "UltraEdit_Region_10"
DIR_NAME = input(f"Dataset directory [{DIR_NAME_DEFAULT}]: ") or DIR_NAME_DEFAULT

# ChordEdit backbone used for encoders / embedding caches.
# Disk layout uses embeddings/<CHORD_EDIT_MODEL>/{DIR_NAME}/...
CHORD_EDIT_MODEL = "sd_turbo"  # "sd_turbo" | "sdxl_turbo" | "flux"

CHORD_EDIT_MODEL_CONFIGS = {
    "sd_turbo": {
        "root": Path("/shared/ssd_30T/mirick/models/sd-turbo"),
        "image_size": 512,
        "pipeline_type": "sd",
    },
    "sdxl_turbo": {
        "root": Path("/shared/ssd_30T/zarageddes/models/sdxl-turbo"),
        "image_size": 1024,
        "pipeline_type": "sdxl",
    },
    "flux": {
        "root": Path("/shared/ssd_30T/zarageddes/models/flux1-schnell"),
        "image_size": 1024,
        "pipeline_type": "flux",
    },
}
if CHORD_EDIT_MODEL not in CHORD_EDIT_MODEL_CONFIGS:
    raise ValueError(
        f"Unknown CHORD_EDIT_MODEL={CHORD_EDIT_MODEL!r}; "
        f"expected one of {sorted(CHORD_EDIT_MODEL_CONFIGS)}"
    )
_CHORD_CFG = CHORD_EDIT_MODEL_CONFIGS[CHORD_EDIT_MODEL]
CHORD_EDIT_MODEL_ROOT = _CHORD_CFG["root"]
IMAGE_SIZE = int(_CHORD_CFG["image_size"])
CHORD_EDIT_PIPELINE_TYPE = str(_CHORD_CFG["pipeline_type"])

GENERATED_DIR = Path(f"/shared/ssd_30T/mirick/generated/ultra_edit/{DIR_NAME}")
DATASET_DIR = Path(f"/shared/ssd_30T/mirick/datasets/ultra_edit/{DIR_NAME}")
# Scattered per-sample embeddings (used when FREEZE_ENCODERS is True).
EMBEDDINGS_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/{CHORD_EDIT_MODEL}/{DIR_NAME}")
EMBEDDINGS_SAMPLES_DIRNAME = "annotation_embeddings"

_SLUG = DIR_NAME.replace("_", "").lower()
INPUTS_CSV = GENERATED_DIR / f"id_to_inputs_{_SLUG}.csv"
METRICS_CSV = GENERATED_DIR / f"id_to_metrics_{_SLUG}.csv"
EMBEDDINGS_CSV = EMBEDDINGS_DIR / f"id_to_embeddings_{_SLUG}.csv"

# Package root: live settings.py sits next to model_m.py; run copies live under
# outputs/<DIR_NAME>/<timestamp>/settings.py (parents[2] == package dir).
_HERE = Path(__file__).resolve().parent
_PACKAGE_DIR = _HERE if (_HERE / "model_m.py").exists() else _HERE.parents[2]

# NOTE: Set this to the directory where the model outputs will be saved.
OUTPUTS_DIR = _PACKAGE_DIR / "outputs" / DIR_NAME
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
_outputs_gitignore = OUTPUTS_DIR.parent / ".gitignore"
if not _outputs_gitignore.exists():
    _outputs_gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

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
# t_delta is the paper's transport-estimator parameter delta; paper results use delta = 0.
TARGET_T_DELTA = 0.0

# Sample-level split ratios (by sample_id, not individual grid rows).
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

SEED = 42


"""
M model settings.

Paper: surrogate model M_hat(x_src, c_src, c_tar, t*, t**) -> s = (s_1, s_2),
predicting s_1 = PSNR-Unedited and s_2 = CLIP-Edited. Code's t_start/t_end are
the paper's (t*, t**); mask is the edit mask m_obj.

    Model architecture:
    M(img_emb, mask_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)
"""

# Regression targets in the loaded dataframe (after PSNR_COL/CLIP_COL rename).
M_TARGET_COLS = (PSNR_COL, CLIP_COL)
M_TARGET_LABELS = {PSNR_COL: "PSNR-Unedited", CLIP_COL: "CLIP-Edited"}

# ChordEdit encoders loaded from CHORD_EDIT_MODEL_ROOT for image/text embedding.
# When FREEZE_ENCODERS is True and EMBEDDINGS_CSV is set, embeddings are loaded from disk.
# Set FREEZE_ENCODERS=False or EMBEDDINGS_CSV=None to encode on the fly instead.
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

Paper: selector model T(s in S) -> (t*, t**), which evaluates M_hat over all
(t*, t**) in the quantized grid T and picks the argmax of phi.

    Model architecture:
    T(img, src_prompt, tar_prompt) -> (t_start, t_end)
"""

# Scalar objective phi for timestep selection and M_hat ranking loss.
# T_TARGET_PHI takes per-sample normalized deltas Delta; see scores.calc_normalized_deltas.
_T_SCORE_KW = dict(alpha=2.0)
T_TARGET_PHI = partial(linex_score, **_T_SCORE_KW)  # Torch phi(Delta)
T_TARGET_PHI_DF = partial(score_df, score_fn=linex_score, **_T_SCORE_KW)  # DataFrame
T_TARGET_COL = "linex_score"
T_TARGET_LABEL = "LINEX Score"

# Baseline timestep bounds from the original ChordEdit paper (used in eval).
DEFAULT_T_START = 0.9
DEFAULT_T_END = 0.3

# Discrete grid axes for T (timestep selector).
GRID_T_START = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID_T_END = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Deviate-or-default gate: minimum predicted phi gain to leave baseline timesteps.
NOISE_FLOOR_PHI = 0.0
