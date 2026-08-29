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
    """Value of --settings-path on the command line, or None if absent."""
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
SETTINGS_JSON = Path(globals().get("SETTINGS_PATH_OVERRIDE") or _settings_path_from_argv() or _SETTINGS_DIR / "settings.json")
if not SETTINGS_JSON.exists():
    raise FileNotFoundError(f"Missing settings file: {SETTINGS_JSON}")
CONFIG: dict = {k: v for k, v in _json.loads(SETTINGS_JSON.read_text()).items() if not k.startswith("_")}


# Sentinel so that None stays a usable default.
_REQUIRED = object()


# Function to get a value from the settings json.
def _cfg(name, default=_REQUIRED):
    """One value from settings.json, with a readable error when it is absent.

    Keys added after a run was trained pass a default so that the run's
    settings snapshot still loads when selector.py replays it.
    """
    if name not in CONFIG:
        if default is _REQUIRED:
            raise KeyError(f"{name!r} is missing from {SETTINGS_JSON}")
        return default
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

# Map the package root from next to model.py or the copy saved under runs/.
_here = Path(__file__).resolve().parent
_package_dir = _here if (_here / "model.py").exists() else _here.parents[2]

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
"""

# Regression targets in the loaded dataframe.
TARGET_COLS = (PSNR_COL, CLIP_COL)
TARGET_LABELS = {PSNR_COL: "PSNR-Unedited", CLIP_COL: "CLIP-Edited"}

# Select from "residuals" or "deltas". Targets are always per-sample normalized
# deltas Delta (scores.calc_normalized_deltas); "residuals" additionally
# subtracts the train split's mean true delta surface (saved to the run as
# mean_surface.pt), so the towers regress how an image deviates from the
# population surface and T adds the surface back at selection time. "deltas"
# regresses the full deltas with no offset.
PREDICTION_SPACE = str(_cfg("PREDICTION_SPACE"))
if PREDICTION_SPACE not in ("deltas", "residuals"):
    raise ValueError(f"Unknown {PREDICTION_SPACE=}")

_MAX_SAMPLES = _cfg("MAX_SAMPLES")
MAX_SAMPLES = None if _MAX_SAMPLES is None else int(_MAX_SAMPLES)

USE_CENTER_CROP = bool(_cfg("USE_CENTER_CROP"))

# Side of the square latent patches forming the visual tokens. The VAE latent
# is (4, S, S) with S = CHORD_EDIT_IMAGE_SIZE // 8, so PATCH_SIZE p gives
# (S // p) ** 2 tokens of dim 4 * p ** 2, the same F_v attention_predictor
# builds. Those tokens are mean-pooled to one vector for these flat towers, and
# the projection is linear, so pooling commutes with it: p sets how much
# within-patch spatial detail survives the average. p = S keeps all of it.
LATENT_SIDE = CHORD_EDIT_IMAGE_SIZE // 8
PATCH_SIZE = int(_cfg("PATCH_SIZE"))
if PATCH_SIZE < 1 or LATENT_SIDE % PATCH_SIZE != 0:
    raise ValueError(f"Expected {PATCH_SIZE=} to divide {LATENT_SIDE=}")

# Learned positional embedding on the visual tokens.
USE_POS_EMB = bool(_cfg("USE_POS_EMB"))

# Which image representation the regressor sees. "vae" is the flattened SD VAE
# latent the model has always used. "clip" replaces it with a mean-pooled
# CLIP-L/14 embedding, the encoder CLIP-Edited is scored with, pooled the way
# the cached text embeddings were. "vae+clip" keeps the latent and projects the
# CLIP embedding as a second image arm.
IMG_EMB_SOURCE = str(_cfg("IMG_EMB_SOURCE", "vae"))
if IMG_EMB_SOURCE not in ("vae", "clip", "vae+clip"):
    raise ValueError(f"Unknown {IMG_EMB_SOURCE=}; expected 'vae', 'clip' or 'vae+clip'")

# Which prompt representation the regressor sees. "sd" is the sd_turbo text
# encoder's mean-pooled hidden states, the cached source.pt / target.pt. "clip"
# replaces them with CLIP-L/14 text embeddings, so that image and text share one
# space and the CLIP-Edited cosine becomes expressible from the inputs.
TEXT_EMB_SOURCE = str(_cfg("TEXT_EMB_SOURCE", "sd"))
if TEXT_EMB_SOURCE not in ("sd", "clip"):
    raise ValueError(f"Unknown {TEXT_EMB_SOURCE=}; expected 'sd' or 'clip'")

# Regressor MLP architecture.
IMG_PROJ_DIM = int(_cfg("IMG_PROJ_DIM"))
TEXT_PROJ_DIM = int(_cfg("TEXT_PROJ_DIM"))
MLP_WIDE = int(_cfg("MLP_WIDE"))
MLP_HIDDEN = int(_cfg("MLP_HIDDEN"))
MLP_INNER = int(_cfg("MLP_INNER"))
MLP_PSNR_DROPOUT = float(_cfg("MLP_PSNR_DROPOUT"))
MLP_CLIP_DROPOUT = float(_cfg("MLP_CLIP_DROPOUT"))

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
# Metric used to pick the best-epoch checkpoint. Select from
# "val_phi_spearman", "val_regret", "val_gain_mean", or "val_loss".
CKPT_METRIC = str(_cfg("CKPT_METRIC"))
if CKPT_METRIC not in ("val_phi_spearman", "val_regret", "val_gain_mean", "val_loss"):
    raise ValueError(f"Unknown {CKPT_METRIC=}")
# Number of sample grids concatenated per training batch.
GRIDS_PER_BATCH = int(_cfg("GRIDS_PER_BATCH"))
# Exponential moving average of the weights.
EMA_DECAY = float(_cfg("EMA_DECAY"))

# Run directory name under RUNS_DIR; empty string means use a timestamp.
RUN_NAME = str(_cfg("RUN_NAME"))


"""
T model settings.
"""

# Scalar objective phi for timestep selection and M^ ranking loss.
SCORE_FNS = {
    "naive": (naive_score, "Naive Score"),
    "cara": (cara_score, "CARA Score"),
    "linex": (linex_score, "LINEX Score"),
}
SCORE_FN = str(_cfg("SCORE_FN"))
if SCORE_FN not in SCORE_FNS:
    raise ValueError(f"Unknown {SCORE_FN=}")
_SCORE_FN, SCORE_LABEL = SCORE_FNS[SCORE_FN]

PHI_ALPHA = float(_cfg("PHI_ALPHA"))
_SCORE_KW = {"alpha": PHI_ALPHA} if "alpha" in _signature(_SCORE_FN).parameters else {}
SCORE_PHI = partial(_SCORE_FN, **_SCORE_KW)  # Torch phi(Delta)
SCORE_PHI_DF = partial(score_df, score_fn=_SCORE_FN, **_SCORE_KW)  # DataFrame
SCORE_COL = f"{SCORE_FN}_score"

# Baseline timestep bounds from the ChordEdit paper.
# NOTE: Must be set to match (0.9-t_delta, 0.3)
DEFAULT_T_START = float(_cfg("DEFAULT_T_START"))
DEFAULT_T_END = float(_cfg("DEFAULT_T_END"))

# Deviate-or-default gate. Minimum predicted phi gain to leave baseline timesteps.
NOISE_FLOOR_PHI = float(_cfg("NOISE_FLOOR_PHI"))
