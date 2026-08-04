# settings.py

import json as _json
import sys as _sys
from functools import partial
from inspect import signature as _signature
from pathlib import Path

from scores import cara_score, linex_score, naive_score, score_df


"""
Configuration.

Every tunable lives in settings.json; this module reads that file and derives
the rest (paths, column names, the phi partials) from it. Nothing here reads
the environment - to run a different configuration, edit settings.json or pass
--settings-path pointing at another copy of it:

    python train_m.py --settings-path /tmp/my_config.json

Each run saves the exact config it used to <run_dir>/settings.json, and
_helpers.load_run_settings replays a run from that snapshot, so evaluating a
run reproduces its training config.
"""

_SETTINGS_DIR = Path(__file__).resolve().parent
_SETTINGS_ARG = "--settings-path"


def _settings_path_from_argv() -> str | None:
    """Value of --settings-path on the command line, or None if absent.

    This module binds every constant at import time, which happens before any
    entry point's argparse runs, so the path has to be read straight off argv.
    The entry points still declare the flag so that it appears in --help and a
    typo is an error rather than a silent fallback to the package default.
    """
    argv = _sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == _SETTINGS_ARG:
            if i + 1 >= len(argv):
                raise ValueError(f"{_SETTINGS_ARG} expects a path")
            return argv[i + 1]
        if arg.startswith(f"{_SETTINGS_ARG}="):
            return arg.split("=", 1)[1]
    return None


# _helpers.load_run_settings sets SETTINGS_PATH_OVERRIDE on this module before
# executing it, so replaying a run's snapshot never depends on the argv of
# whatever script is doing the replaying.
SETTINGS_JSON = Path(
    globals().get("SETTINGS_PATH_OVERRIDE")
    or _settings_path_from_argv()
    or _SETTINGS_DIR / "settings.json"
)
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

# Package root: settings.py sits next to model_m.py, except for the copy saved
# under outputs/<DIR_NAME>/<run>/ (parents[2] == package dir).
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

# Column names in the id_to_predictions_*.csv file written after selection.
PRED_T_START_COL = "pred_t_start"
PRED_T_END_COL = "pred_t_end"


"""
Shared model settings.
"""

# Values of t_delta column to train models on. Set to `null` to use every t_delta.
# t_delta is the paper's transport-estimator parameter delta; paper results use delta = 0.
TARGET_T_DELTA = _cfg("TARGET_T_DELTA")

# Sample-level split ratios (by sample_id, not individual grid rows).
TRAIN_FRAC = float(_cfg("TRAIN_FRAC"))
VAL_FRAC = float(_cfg("VAL_FRAC"))
TEST_FRAC = float(_cfg("TEST_FRAC"))

# Seeds model init and batch order.
SEED = int(_cfg("SEED"))

# Seed for the sample-level train/val/test split. Kept separate from SEED so
# model init can be reseeded (variance estimates, ensembles) without moving
# samples between splits.
SPLIT_SEED = int(_cfg("SPLIT_SEED"))


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

# ChordEdit encoders loaded from CHORD_EDIT_MODEL_ROOT for image/text embedding
# dims and live encode/predict_raw. Encoders are inherited from the ChordEdit
# pipeline and are always frozen; training/eval embeddings come from the
# packed/scattered caches.
USE_CENTER_CROP = bool(_cfg("USE_CENTER_CROP"))

# How the flattened VAE latents are projected: "linear" (one Linear over the
# 16k flat vector) or "conv" (fold back to (C, S, S) and downsample). The
# latent's spatial layout carries the mask's size and position, which a flat
# Linear cannot see.
IMG_ENCODER = str(_cfg("IMG_ENCODER"))

# Regressor MLP architecture.
IMG_PROJ_DIM = int(_cfg("IMG_PROJ_DIM"))
TEXT_PROJ_DIM = int(_cfg("TEXT_PROJ_DIM"))
MLP_WIDE = int(_cfg("MLP_WIDE"))
MLP_HIDDEN = int(_cfg("MLP_HIDDEN"))
MLP_INNER = int(_cfg("MLP_INNER"))
MLP_PSNR_DROPOUT = float(_cfg("MLP_PSNR_DROPOUT"))
MLP_CLIP_DROPOUT = float(_cfg("MLP_CLIP_DROPOUT"))
T_FOURIER_FREQS = int(_cfg("T_FOURIER_FREQS"))
T_PROJ_DIM = int(_cfg("T_PROJ_DIM"))

# M training hyperparameters.
EPOCHS = int(_cfg("EPOCHS"))
BATCH_SIZE = int(_cfg("BATCH_SIZE"))
LR = float(_cfg("LR"))
WEIGHT_DECAY = float(_cfg("WEIGHT_DECAY"))
NORMALIZE_TARGETS = bool(_cfg("NORMALIZE_TARGETS"))
RANKING_LOSS_WEIGHT = float(_cfg("RANKING_LOSS_WEIGHT"))

# Restrict the ranking loss to pairs whose better cell is in the true top-k of
# its grid. 0 uses every pair. Regret only cares about the top of the grid.
RANKING_TOP_K = int(_cfg("RANKING_TOP_K"))

# Per-target weights on the z-scored MSE loss (train-time only; evaluate()
# reports the unweighted loss so runs stay comparable).
PSNR_LOSS_WEIGHT = float(_cfg("PSNR_LOSS_WEIGHT"))
CLIP_LOSS_WEIGHT = float(_cfg("CLIP_LOSS_WEIGHT"))

# LR schedule over epochs: "none" | "cosine".
LR_SCHEDULER = str(_cfg("LR_SCHEDULER"))

# Stop when the checkpoint metric has not improved for this many epochs
# (after a minimum of 5 epochs). 0 disables early stopping.
EARLY_STOP_PATIENCE = int(_cfg("EARLY_STOP_PATIENCE"))

# Metric used to pick the best-epoch checkpoint:
#   "val_phi_spearman" - median per-sample Spearman between predicted and
#       true phi over each val sample's grid (aligned with T selection),
#   "val_regret" - median per-sample regret on the same grids,
#   "val_loss" - summed z-scored MSE over both targets.
CKPT_METRIC = str(_cfg("CKPT_METRIC"))

# Number of sample grids concatenated per training batch.
GRIDS_PER_BATCH = int(_cfg("GRIDS_PER_BATCH"))

# Exponential moving average of the weights, evaluated and checkpointed in
# place of the live weights. 0 disables it. Val phi-Spearman swings by a few
# points between neighboring epochs, so an averaged iterate is a steadier
# thing to select on.
EMA_DECAY = float(_cfg("EMA_DECAY"))

# Anchor predictions on the train split's average value at each grid cell, so
# the towers predict the per-image deviation from the shared surface rather
# than re-deriving that surface.
USE_CELL_ANCHOR = bool(_cfg("USE_CELL_ANCHOR"))

# Run directory name under OUTPUTS_DIR; empty string means use a timestamp.
RUN_NAME = str(_cfg("RUN_NAME"))


"""
T model settings.

Paper: selector model T(s in S) -> (t*, t**), which evaluates M_hat over all
(t*, t**) in the quantized grid T and picks the argmax of phi.

    Model architecture:
    T(img, src_prompt, tar_prompt) -> (t_start, t_end)
"""

# Scalar objective phi for timestep selection and M_hat ranking loss.
# T_TARGET_PHI takes per-sample normalized deltas Delta; see scores.calc_normalized_deltas.
#
#   "naive" - weighted sum of the deltas; indifferent to how gains are split
#             between the metrics
#   "cara"  - deltas through an exponential utility; concave, so it penalizes
#             regressions superlinearly and biases toward balance
#   "linex" - the average of the two: CARA's regression penalty without its
#             reward cap
#
# PHI_ALPHA sets the curvature and is ignored by scores that do not take it.
# Changing either name or alpha changes the objective itself, so runs either
# side of a change are not comparable.
T_TARGET_FNS = {
    "naive": (naive_score, "Naive Score"),
    "cara": (cara_score, "CARA Score"),
    "linex": (linex_score, "LINEX Score"),
}
T_TARGET_FN = str(_cfg("T_TARGET_FN"))
if T_TARGET_FN not in T_TARGET_FNS:
    raise ValueError(f"Unknown {T_TARGET_FN=}")
_T_SCORE_FN, T_TARGET_LABEL = T_TARGET_FNS[T_TARGET_FN]

PHI_ALPHA = float(_cfg("PHI_ALPHA"))
_T_SCORE_KW = {"alpha": PHI_ALPHA} if "alpha" in _signature(_T_SCORE_FN).parameters else {}
T_TARGET_PHI = partial(_T_SCORE_FN, **_T_SCORE_KW)  # Torch phi(Delta)
T_TARGET_PHI_DF = partial(score_df, score_fn=_T_SCORE_FN, **_T_SCORE_KW)  # DataFrame
T_TARGET_COL = f"{T_TARGET_FN}_score"

# Baseline timestep bounds from the original ChordEdit paper (used in eval).
# phi is measured relative to this cell, so moving it redefines the objective.
DEFAULT_T_START = float(_cfg("DEFAULT_T_START"))
DEFAULT_T_END = float(_cfg("DEFAULT_T_END"))

# Discrete grid axes for T (timestep selector).
GRID_T_START = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID_T_END = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# Which grid cells the dataset exposes:
#   "all"   - every labeled (t_start, t_end), i.e. the full 11 x 11 mesh
#   "lower" - only t_start > t_end, the strict lower triangle the annotation
#             pipeline covered before it filled the rest in
# Restricting this changes both what the model trains on and what T can pick,
# and phi is normalized over whichever candidate set is present, so runs on
# different regions are not directly comparable.
CELL_REGIONS = ("all", "lower")
CELL_REGION = str(_cfg("CELL_REGION"))
if CELL_REGION not in CELL_REGIONS:
    raise ValueError(f"Unknown {CELL_REGION=}; expected one of {CELL_REGIONS}")

# Deviate-or-default gate: minimum predicted phi gain to leave baseline timesteps.
NOISE_FLOOR_PHI = float(_cfg("NOISE_FLOOR_PHI"))
