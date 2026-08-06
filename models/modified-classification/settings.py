# settings.py

"""
Load settings from the settings.json file.
"""

import json as _json
import sys as _sys
from functools import partial
from inspect import signature as _signature
from pathlib import Path

from scores import cara_score, linex_score, naive_score, score_df


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

_CHORD_EDIT_MODEL_CONFIGS = {
    "sd_turbo": (Path("/shared/ssd_30T/mirick/models/sd-turbo"), 512, "sd"),
    "sdxl_turbo": (Path("/shared/ssd_30T/zarageddes/models/sdxl-turbo"), 1024, "sdxl"),
    "flux": (Path("/shared/ssd_30T/zarageddes/models/flux1-schnell"), 1024, "flux"),
}
# NOTE: Choose from "sd_turbo", "sdxl_turbo", or "flux".
CHORD_EDIT_MODEL = str(_cfg("CHORD_EDIT_MODEL"))  
if CHORD_EDIT_MODEL not in _CHORD_EDIT_MODEL_CONFIGS:
    raise ValueError(f"Unknown {CHORD_EDIT_MODEL=}")
CHORD_EDIT_MODEL_ROOT, CHORD_EDIT_IMAGE_SIZE, CHORD_EDIT_PIPELINE_TYPE = _CHORD_EDIT_MODEL_CONFIGS[CHORD_EDIT_MODEL]

# NOTE: Choose from "UltraEdit_Region_<N>", "UltraEdit_Background_1000_v2", or "UltraEdit_Style_1000_v2".
DIR_NAME = str(_cfg("DIR_NAME"))
GENERATED_DIR = Path(f"/shared/ssd_30T/mirick/generated/{CHORD_EDIT_MODEL}/0p0/{DIR_NAME}")
DATASET_DIR = Path(f"/shared/ssd_30T/mirick/datasets/ultra_edit/{DIR_NAME}")
SCATTERED_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/{CHORD_EDIT_MODEL}/{DIR_NAME}")

# Map the package root from next to train_m.py or the copy saved under runs/.
_here = Path(__file__).resolve().parent
_package_dir = _here if (_here / "model_m.py").exists() else _here.parents[2]

# NOTE: Set this to the directory where the model runs will be saved.
RUNS_DIR = _package_dir / "runs" / DIR_NAME
RUNS_DIR.mkdir(parents=True, exist_ok=True)
_runs_gitignore = RUNS_DIR.parent / ".gitignore"
if not _runs_gitignore.exists():
    _runs_gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

# Shared column name for sample IDs.
SAMPLE_ID_COL = "sample_id"

# Column names in the id_to_inputs_*.csv file.
INPUTS_CSV = GENERATED_DIR / f"id_to_inputs_{DIR_NAME.replace("_", "").lower()}.csv"
SOURCE_PROMPT_COL = "source_prompt"
TARGET_PROMPT_COL = "target_prompt"
IMAGE_PATH_COL = "image_path"
MASK_PATH_COL = "mask_image_path"

# Column names in the id_to_metrics_*.csv file.
METRICS_CSV = GENERATED_DIR / f"id_to_metrics_{DIR_NAME.replace("_", "").lower()}.csv"
T_START_COL = "t_start"
T_END_COL = "t_end"
T_DELTA_COL = "t_delta"
PSNR_COL = "psnr_unedit_part"
CLIP_COL = "clip_similarity_target_image_edit_part"

# Column names in the id_to_predictions_*.csv file written after selection.
PRED_T_START_COL = "pred_t_start"
PRED_T_END_COL = "pred_t_end"


"""
Shared model settings.
"""

# Values of t_delta column to train models on. Set to `null` to use every t_delta.
TARGET_T_DELTA = _cfg("TARGET_T_DELTA")

TRAIN_FRAC = float(_cfg("TRAIN_FRAC"))
VAL_FRAC = float(_cfg("VAL_FRAC"))

# SEED and SPLIT_SEED are kept separate so that model init can be
# reseeded without moving samples between splits.
SEED = int(_cfg("SEED"))
SPLIT_SEED = int(_cfg("SPLIT_SEED"))


"""
M model settings.

    Model architecture:
    M(img_emb, mask_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)
"""

# Regression targets in the loaded dataframe (after PSNR_COL/CLIP_COL rename).
M_TARGET_COLS = (PSNR_COL, CLIP_COL)
M_TARGET_LABELS = {PSNR_COL: "PSNR-Unedited", CLIP_COL: "CLIP-Edited"}

# Select from "residual" or "delta". Targets are always per-sample normalized
# deltas Delta (scores.calc_normalized_deltas); "residual" additionally
# subtracts the train split's mean true delta surface (saved to the run as
# mean_surface.pt), so the towers regress how an image deviates from the
# population surface and T adds the surface back at selection time. "delta"
# regresses the full deltas with no offset.
M_TARGET_SPACE = str(_cfg("M_TARGET_SPACE"))
if M_TARGET_SPACE not in ("residual", "delta"):
    raise ValueError(f"Unknown {M_TARGET_SPACE=}; expected 'residual' or 'delta'")


USE_CENTER_CROP = bool(_cfg("USE_CENTER_CROP"))

# Select from "linear" or "conv". "Linear" projects the flattened VAE
# latents with a single Linear layer. "conv" projects with a
# convolutional layer.
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

EPOCHS = int(_cfg("EPOCHS"))
BATCH_SIZE = int(_cfg("BATCH_SIZE"))
LR = float(_cfg("LR"))
WEIGHT_DECAY = float(_cfg("WEIGHT_DECAY"))
NORMALIZE_TARGETS = bool(_cfg("NORMALIZE_TARGETS"))
RANKING_LOSS_WEIGHT = float(_cfg("RANKING_LOSS_WEIGHT"))

# Restrict ranking loss to pairs in the true top-k of its grid.
RANKING_TOP_K = int(_cfg("RANKING_TOP_K"))
# Per-target weights on the z-scored MSE loss.
PSNR_LOSS_WEIGHT = float(_cfg("PSNR_LOSS_WEIGHT"))
CLIP_LOSS_WEIGHT = float(_cfg("CLIP_LOSS_WEIGHT"))
# Select from "none" or "cosine".
LR_SCHEDULER = str(_cfg("LR_SCHEDULER"))
# Stop when the checkpoint metric has not improved for this many epochs.
EARLY_STOP_PATIENCE = int(_cfg("EARLY_STOP_PATIENCE"))
# Metric used to pick the best-epoch checkpoint:
#   "val_phi_spearman" is the median per-sample Spearman between predicted and
#       true phi over each val sample's grid (the model's own ordering,
#       blind to the level of predicted phi).
#   "val_regret" is the median per-sample regret of the deployed selector
#       (predictions plus the mean surface in "residual" space) on the same grids.
#   "val_gain_mean" is the mean true phi at the deployed selector's pick, i.e.
#       the mean gain over the default cell (level-sensitive, sign matters).
#   "val_loss" is the summed z-scored MSE over both targets.
CKPT_METRIC = str(_cfg("CKPT_METRIC"))
# Number of sample grids concatenated per training batch.
GRIDS_PER_BATCH = int(_cfg("GRIDS_PER_BATCH"))
# Exponential moving average of the weights.
EMA_DECAY = float(_cfg("EMA_DECAY"))

# Run directory name under RUNS_DIR; empty string means use a timestamp.
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

# The candidate (t_start, t_end) grid is not configured: the cells labeled in
# METRICS_CSV define it. phi is normalized over that candidate set, so runs on
# datasets labeling different regions are not directly comparable.

# Deviate-or-default gate: minimum predicted phi gain to leave baseline timesteps.
NOISE_FLOOR_PHI = float(_cfg("NOISE_FLOOR_PHI"))
