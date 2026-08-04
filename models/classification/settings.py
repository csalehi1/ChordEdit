# settings.py

import json as _json
import os as _os
from functools import partial
from inspect import signature as _signature
from pathlib import Path

import numpy as np

from scores import cara_score, linex_score, naive_score, score_df


"""
Configuration.

Every tunable lives in settings.json; this module reads that file and derives
the rest (paths, column names, the score partials) from it. Nothing here reads
per-setting environment variables - to run a different configuration, edit
settings.json or point CE_SETTINGS_JSON at another copy of it:

    CE_SETTINGS_JSON=/tmp/my_config.json python train.py

Each run saves the exact config it used to <run_dir>/settings.json, and
_helpers.load_run_settings replays a run by pointing CE_SETTINGS_JSON at it,
so evaluating a run reproduces its training config.
"""

_SETTINGS_DIR = Path(__file__).resolve().parent
SETTINGS_JSON = Path(_os.environ.get("CE_SETTINGS_JSON") or _SETTINGS_DIR / "settings.json")
if not SETTINGS_JSON.exists():
    raise FileNotFoundError(f"Missing settings file: {SETTINGS_JSON}")
CONFIG: dict = {k: v for k, v in _json.loads(SETTINGS_JSON.read_text()).items() if not k.startswith("_")}


def _cfg(name):
    """One value from settings.json, with a readable error when it is absent."""
    if name not in CONFIG:
        raise KeyError(f"{name!r} is missing from {SETTINGS_JSON}")
    return CONFIG[name]


"""
Dataset settings.
"""

DIR_NAME = str(_cfg("DIR_NAME"))  # UltraEdit_Background_1000_v2 | UltraEdit_Region_<N> | UltraEdit_Style_1000_v2
CHORD_EDIT_MODEL = str(_cfg("CHORD_EDIT_MODEL"))  # "sd_turbo" | "sdxl_turbo" | "flux"

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
CHORD_EDIT_IMAGE_SIZE = int(_CHORD_CFG["image_size"])
CHORD_EDIT_PIPELINE_TYPE = str(_CHORD_CFG["pipeline_type"])

GENERATED_DIR = Path(f"/shared/ssd_30T/mirick/generated/{CHORD_EDIT_MODEL}/0p0/{DIR_NAME}")
DATASET_DIR = Path(f"/shared/ssd_30T/mirick/datasets/ultra_edit/{DIR_NAME}")
# Scattered per-sample embeddings written by the annotation pipeline.
EMBEDDINGS_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/{CHORD_EDIT_MODEL}/{DIR_NAME}")
EMBEDDINGS_SAMPLES_DIRNAME = "annotation_embeddings"

_SLUG = DIR_NAME.replace("_", "").lower()
INPUTS_CSV = GENERATED_DIR / f"id_to_inputs_{_SLUG}.csv"
METRICS_CSV = GENERATED_DIR / f"id_to_metrics_{_SLUG}.csv"
EMBEDDINGS_CSV = EMBEDDINGS_DIR / f"id_to_embeddings_{_SLUG}.csv"

# Package root: settings.py sits next to model.py, except for the copy saved
# under outputs/<DIR_NAME>/<run>/ (parents[2] == package dir).
_HERE = Path(__file__).resolve().parent
_PACKAGE_DIR = _HERE if (_HERE / "model.py").exists() else _HERE.parents[2]

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

# Column names in the id_to_predictions_*.csv file written after selection.
PRED_T_START_COL = "pred_t_start"
PRED_T_END_COL = "pred_t_end"


"""
Shared model settings.
"""

# Values of t_delta column to train models on. Set to `null` to use every t_delta.
# t_delta is the paper's transport-estimator parameter delta; paper results use delta = 0.
TARGET_T_DELTA = _cfg("TARGET_T_DELTA")

# Which grid cells are candidates: "all" (121) or "lower" (t_end < t_start, 55).
# "lower" reproduces the strict lower triangle that was the only labeled region
# before the annotation pass filled the metrics CSV out to all 11 x 11 cells.
# phi is normalized per sample over the candidate cells, so this changes the
# labels and the objective's scale, not just the argmax domain: runs either side
# of a change are not comparable.
CELL_SUBSET = str(_cfg("CELL_SUBSET"))
if CELL_SUBSET not in ("all", "lower"):
    raise ValueError(f"Unknown {CELL_SUBSET=}; expected 'all' or 'lower'")

# Sample-level split ratios (by sample_id).
TRAIN_FRAC = float(_cfg("TRAIN_FRAC"))
VAL_FRAC = float(_cfg("VAL_FRAC"))
TEST_FRAC = float(_cfg("TEST_FRAC"))

# Seeds model init and batch order. Held apart from SPLIT_SEED so repeats over
# SEED measure init noise on one fixed test set rather than resampling it.
SEED = int(_cfg("SEED"))

# Seeds the sample-level train/val/test split. Changing it moves samples
# between splits, so runs either side of a change are not comparable.
SPLIT_SEED = int(_cfg("SPLIT_SEED"))


"""
C model settings.

Classifier C(img_emb, mask_emb, src_emb, tar_emb) -> (t_start, t_end),
predicting the grid cell with the best C_TARGET_COL score for each sample.
Code's t_start/t_end are the paper's (t*, t**); mask is the edit mask m_obj.
"""

# Metric columns scalarized into the per-sample selection target.
C_TARGET_COLS = (PSNR_COL, CLIP_COL)
C_TARGET_LABELS = {PSNR_COL: "Whole PSNR", CLIP_COL: "CLIP-Edited"}

# How the flattened VAE latents are projected: "linear" (one Linear over the
# 16k flat vector) or "conv" (fold back to (C, S, S) and downsample). The
# latent's spatial layout carries the mask's size and position, which a flat
# Linear cannot see.
IMG_ENCODER = str(_cfg("IMG_ENCODER"))

# Classifier MLP architecture.
IMG_PROJ_DIM = int(_cfg("IMG_PROJ_DIM"))
TEXT_PROJ_DIM = int(_cfg("TEXT_PROJ_DIM"))
MLP_WIDE = int(_cfg("MLP_WIDE"))
MLP_HIDDEN = int(_cfg("MLP_HIDDEN"))
MLP_INNER = int(_cfg("MLP_INNER"))
MLP_DROPOUT = float(_cfg("MLP_DROPOUT"))

# Head type for the last step of the model: "CORAL" | "MSE" | "CE".
HEAD_TYPE = str(_cfg("HEAD_TYPE"))
if HEAD_TYPE not in ("CORAL", "MSE", "CE"):
    raise ValueError(f"Unknown {HEAD_TYPE=}; expected 'CORAL', 'MSE', or 'CE'")

# Only used when HEAD_TYPE is "CE": "cost_sensitive_ce_loss" | "one_hot_ce_loss".
CE_LOSS_TYPE = str(_cfg("CE_LOSS_TYPE"))
if CE_LOSS_TYPE not in ("cost_sensitive_ce_loss", "one_hot_ce_loss"):
    raise ValueError(f"Unknown {CE_LOSS_TYPE=}")

# Counteract label imbalance in the training split.
USE_CLASS_WEIGHTS = bool(_cfg("USE_CLASS_WEIGHTS"))
# Softens overconfident majority-class collapse in CE training.
LABEL_SMOOTHING = float(_cfg("LABEL_SMOOTHING"))

# C training hyperparameters.
EPOCHS = int(_cfg("EPOCHS"))
BATCH_SIZE = int(_cfg("BATCH_SIZE"))
LR = float(_cfg("LR"))
WEIGHT_DECAY = float(_cfg("WEIGHT_DECAY"))

# Per-epoch learning-rate decay: "none" | "cosine" | "linear" | "step" | "plateau".
# LR_MIN_FACTOR is the floor as a fraction of LR (ignored by "step").
LR_SCHEDULER = str(_cfg("LR_SCHEDULER"))
LR_MIN_FACTOR = float(_cfg("LR_MIN_FACTOR"))

# Validation metric that selects the checkpoint; see train.CKPT_METRICS.
# "bal_acc_t_start" is the classifier's native label metric, "regret_median"
# the selection metric the pipeline actually consumes.
CKPT_METRIC = str(_cfg("CKPT_METRIC"))

# Run directory name under OUTPUTS_DIR; empty string means use a timestamp.
RUN_NAME = str(_cfg("RUN_NAME"))


"""
Selection target settings.

Scalar objective phi used to pick one row (grid cell) per sample_id as the
classification label. Scores consume per-sample normalized deltas Delta; see
scores.calc_normalized_deltas.

  "naive" - weighted sum of the deltas; indifferent to how gains are split
            between the metrics
  "cara"  - deltas through an exponential utility; concave, so it penalizes
            regressions superlinearly and biases toward balance
  "linex" - the average of the two: CARA's regression penalty without its
            reward cap

C_TARGET_ALPHA sets the curvature and is ignored by scores that do not take
it. Changing either name or alpha changes the labels themselves, so runs
either side of a change are not comparable.
"""

C_TARGET_FNS = {
    "naive": (naive_score, "Naive Score"),
    "cara": (cara_score, "CARA Score"),
    "linex": (linex_score, "LINEX Score"),
}
C_TARGET_FN = str(_cfg("C_TARGET_FN"))
if C_TARGET_FN not in C_TARGET_FNS:
    raise ValueError(f"Unknown {C_TARGET_FN=}")
_C_SCORE_FN, _C_TARGET_NAME = C_TARGET_FNS[C_TARGET_FN]

C_TARGET_ALPHA = float(_cfg("C_TARGET_ALPHA"))
_C_SCORE_KW = {"alpha": C_TARGET_ALPHA} if "alpha" in _signature(_C_SCORE_FN).parameters else {}
C_TARGET_SCORE = partial(_C_SCORE_FN, **_C_SCORE_KW)  # Torch phi(Delta)
C_TARGET_SCORE_DF = partial(score_df, score_fn=_C_SCORE_FN, **_C_SCORE_KW)  # (df, *cols) -> Series
C_TARGET_COL = f"{C_TARGET_FN}_score"
C_TARGET_LABEL = (
    f"{_C_TARGET_NAME} ($\\alpha={C_TARGET_ALPHA:g}$)" if _C_SCORE_KW else _C_TARGET_NAME
)


def c_target_score_df(df):
    """C_TARGET_FUNC with C_TARGET_COLS bound: metrics DataFrame -> pd.Series."""
    return C_TARGET_SCORE_DF(df, *C_TARGET_COLS)


C_TARGET_FUNC = c_target_score_df

# Baseline timestep bounds from the original ChordEdit paper.
PAPER_T_START = 0.9
PAPER_T_END = 0.3
PAPER_T_DELTA = 0.15

# Baseline cell used for the delta scores, accounting for the paper's finding
# with the nearest grid value. phi is measured relative to this cell, so
# moving it redefines the objective.
DEFAULT_T_START = float(_cfg("DEFAULT_T_START"))
DEFAULT_T_END = float(_cfg("DEFAULT_T_END"))

# Discrete grid axes for the classifier's output buckets.
GRID_VALUES = np.linspace(0.0, 1.0, 11)
GRID_T_START = GRID_VALUES
GRID_T_END = GRID_VALUES
