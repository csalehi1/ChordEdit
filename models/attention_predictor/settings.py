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
_PIE_BENCH_ARG = "--pie-bench"


def _settings_path_from_argv() -> str | None:
    """Value of --settings-path on the command line, or None if absent."""
    argv = _sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == _SETTINGS_ARG:
            if i + 1 >= len(argv):
                raise ValueError(f"Expected a path after {_SETTINGS_ARG}")
            return argv[i + 1]
        if arg.startswith(f"{_SETTINGS_ARG}="):
            return arg.split("=", 1)[1]
    return None


def _pie_bench_from_argv() -> bool:
    """True when --pie-bench is on the command line."""
    return _PIE_BENCH_ARG in _sys.argv[1:]

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
    """
    One value from settings.json. Keys added after a run was
    trained pass a default.
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

# --pie-bench (argv) or a saved run snapshot's PIE_BENCH: replace the UltraEdit
# test split with labeled PIE-Bench samples. Persist into CONFIG so save_run_settings
# writes it and selector / load_split_df can replay the same test source.
PIE_BENCH = bool(_pie_bench_from_argv() or _cfg("PIE_BENCH", False))
CONFIG["PIE_BENCH"] = PIE_BENCH
PIE_BENCH_DIR_NAME = "PIE_Bench_v1"
PIE_SAMPLE_ID_PREFIX = "pie_"
PIE_GENERATED_DIR = Path(f"/shared/ssd_30T/mirick/generated/{CHORD_EDIT_MODEL}/0p0/{PIE_BENCH_DIR_NAME}")
PIE_SCATTERED_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/{CHORD_EDIT_MODEL}/{PIE_BENCH_DIR_NAME}")
PIE_INPUTS_CSV = PIE_GENERATED_DIR / "id_to_inputs_piebenchv1.csv"
PIE_METRICS_CSV = PIE_GENERATED_DIR / "id_to_metrics_piebenchv1.csv"

# Map the package root from next to model.py or the copy saved under runs/.
_here = Path(__file__).resolve().parent
_package_dir = _here if (_here / "model.py").exists() else _here.parents[2]

# Set this to the directory where the model runs will be saved.
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
Predictor settings.
"""

# Regression targets in the loaded dataframe.
TARGET_COLS = (PSNR_COL, CLIP_COL)
TARGET_LABELS = {PSNR_COL: "PSNR-Unedited", CLIP_COL: "CLIP-Edited"}

# NOTE: Choose from "raws", "deltas", or "residuals".
PREDICTION_SPACE = str(_cfg("PREDICTION_SPACE"))
if PREDICTION_SPACE not in ("raws", "deltas", "residuals"):
    raise ValueError(f"Unknown {PREDICTION_SPACE=}")

_MAX_SAMPLES = _cfg("MAX_SAMPLES")
MAX_SAMPLES = None if _MAX_SAMPLES is None else int(_MAX_SAMPLES)

USE_CENTER_CROP = bool(_cfg("USE_CENTER_CROP"))

# Shared dimension d of the visual tokens, the prompt queries, and the edit descriptor h.
ATTN_DIM = int(_cfg("ATTN_DIM"))

# Heads of the cross-attention layer grounding the prompts in the image.
N_HEADS = int(_cfg("N_HEADS"))
if ATTN_DIM % N_HEADS != 0:
    raise ValueError(f"Expected {N_HEADS=} to divide {ATTN_DIM=}")

# Side of the square latent patches forming the cross-attention keys/values.
# The VAE latent is (4, S, S) with S = CHORD_EDIT_IMAGE_SIZE // 8, so PATCH_SIZE
# p gives (S // p) ** 2 tokens of dim 4 * p ** 2. p = 1 is the paper's F_v: one
# token per latent position, flattened spatially and projected by P_v.
LATENT_SIDE = CHORD_EDIT_IMAGE_SIZE // 8
PATCH_SIZE = int(_cfg("PATCH_SIZE"))
if PATCH_SIZE < 1 or LATENT_SIDE % PATCH_SIZE != 0:
    raise ValueError(f"Expected {PATCH_SIZE=} to divide {LATENT_SIDE=}")

# Learned positional embedding on the visual tokens. The paper's F_v has none.
USE_POS_EMB = bool(_cfg("USE_POS_EMB"))

ATTN_DROPOUT = float(_cfg("ATTN_DROPOUT"))

# Number of pre-LN [cross-attention, FFN] residual blocks grounding the prompt
# queries in the visual tokens. 1 with USE_ATTN_RESIDUAL off is the paper's bare
# nn.MultiheadAttention with no residual, no norm, and no FFN.
ATTN_LAYERS = int(_cfg("ATTN_LAYERS", 1))
if ATTN_LAYERS < 1:
    raise ValueError(f"Expected {ATTN_LAYERS=} >= 1")
USE_ATTN_RESIDUAL = bool(_cfg("USE_ATTN_RESIDUAL", False))
# Hidden width of each block's FFN, as a multiple of ATTN_DIM. 0 drops the FFN.
FFN_MULT = float(_cfg("FFN_MULT", 0.0))

# Visual key/value source. "vae" is the SD VAE latent grid the paper uses.
# "clip" swaps in the pooled CLIP-L/14 image embedding as a single token, and
# "vae+clip" appends it to the latent tokens. CLIP-Edited is scored with
# CLIP-L/14, so that encoder's space is the one the label is expressible in.
IMG_EMB_SOURCE = str(_cfg("IMG_EMB_SOURCE", "vae"))
if IMG_EMB_SOURCE not in ("vae", "clip", "vae+clip"):
    raise ValueError(f"Unknown {IMG_EMB_SOURCE=}")

# Text key/value source. "pooled" is the pipeline's masked-mean prompt vector,
# one query per prompt. "tokens" reads the full (77, D) sequences and their
# padding masks, so every prompt token is its own query and pooling happens
# after grounding rather than before it.
TEXT_EMB_SOURCE = str(_cfg("TEXT_EMB_SOURCE", "pooled"))
if TEXT_EMB_SOURCE not in ("pooled", "tokens"):
    raise ValueError(f"Unknown {TEXT_EMB_SOURCE=}")

# Second pooled vector per prompt, weighted by each token's novelty against the
# other prompt (1 - max cosine similarity). Source and target prompts differ in
# a few words, so the masked mean is mostly shared scaffold and the pooled
# difference is attenuated by ~1/L; this pools what changed. Needs "tokens".
USE_DIFF_SALIENCY = bool(_cfg("USE_DIFF_SALIENCY", False))

# LayerNorm each z_edit segment separately instead of once over the whole
# concatenation, so the small difference segments are not dominated by the two
# large concat segments. Null keeps the single LayerNorm(4 * d).
USE_SEGMENT_NORM = bool(_cfg("USE_SEGMENT_NORM", False))

# Learned key/value token a query can attend to instead of the image. Target
# tokens naming content that is not in the source image otherwise have to spend
# their whole softmax mass on patches that do not match them.
USE_NULL_TOKEN = bool(_cfg("USE_NULL_TOKEN", False))

# Subtract the predicted default cell from every cell, so the prediction there
# is exactly 0 as the true delta is by construction. Removes a degree of
# freedom the heads otherwise spend learning that constraint.
PIN_DEFAULT_CELL = bool(_cfg("PIN_DEFAULT_CELL", False))

# LayerScale on each residual branch, so the block starts near-identity and the
# text path is intact at init. Null is a plain residual add.
_LAYERSCALE_INIT = _cfg("LAYERSCALE_INIT", None)
LAYERSCALE_INIT = None if _LAYERSCALE_INIT is None else float(_LAYERSCALE_INIT)

# Hidden width of each metric head. 0 is the paper's bare Linear(d, n_cells).
HEAD_HIDDEN = int(_cfg("HEAD_HIDDEN", 0))
# Give PSNR and CLIP their own combiner C_theta instead of sharing the edit
# descriptor h. The two surfaces are driven by different things, and a shared h
# lets the PSNR-dominated gradient set the representation for both.
SPLIT_COMBINER = bool(_cfg("SPLIT_COMBINER", False))

# Combiner C_theta mapping the difference-aware edit representation
# z_edit (4 * ATTN_DIM) to the edit descriptor h (ATTN_DIM).
COMBINER_HIDDEN = int(_cfg("COMBINER_HIDDEN"))
COMBINER_DROPOUT = float(_cfg("COMBINER_DROPOUT"))

EPOCHS = int(_cfg("EPOCHS"))
LR = float(_cfg("LR"))
WEIGHT_DECAY = float(_cfg("WEIGHT_DECAY"))

# Phi-space objective terms. A weight of 0 disables that term. top_k null = all cells.
MSE_LOSS_WEIGHT = float(_cfg("MSE_LOSS_WEIGHT"))
_MSE_LOSS_TOP_K = _cfg("MSE_LOSS_TOP_K")
MSE_LOSS_TOP_K = None if _MSE_LOSS_TOP_K is None else int(_MSE_LOSS_TOP_K)
RANKING_LOSS_WEIGHT = float(_cfg("RANKING_LOSS_WEIGHT"))
_RANKING_LOSS_TOP_K = _cfg("RANKING_LOSS_TOP_K")
RANKING_LOSS_TOP_K = None if _RANKING_LOSS_TOP_K is None else int(_RANKING_LOSS_TOP_K)

# Per-column MSE on the two delta surfaces, applied before phi scalarizes them.
# The phi-space terms above let error trade freely between PSNR and CLIP, which
# the PSNR column wins; these hold each head to its own column. Weight 0 for
# both is the phi-only objective.
PSNR_LOSS_WEIGHT = float(_cfg("PSNR_LOSS_WEIGHT", 0.0))
CLIP_LOSS_WEIGHT = float(_cfg("CLIP_LOSS_WEIGHT", 0.0))

# Listwise soft cross-entropy between softmax(pred_phi / tau) and
# softmax(true_phi / tau) over the candidate cells. Pairwise ranking spends most
# of its mass on easy far-apart pairs; this concentrates it at the top of the
# ranking, which is what the selector reads.
LISTWISE_LOSS_WEIGHT = float(_cfg("LISTWISE_LOSS_WEIGHT", 0.0))
LISTWISE_TAU = float(_cfg("LISTWISE_TAU", 0.1))
if LISTWISE_TAU <= 0:
    raise ValueError(f"Expected {LISTWISE_TAU=} > 0")

# Select from "none" or "cosine".
LR_SCHEDULER = str(_cfg("LR_SCHEDULER"))
# Stop when the checkpoint metric has not improved for this many epochs.
EARLY_STOP_PATIENCE = int(_cfg("EARLY_STOP_PATIENCE"))
# Metric used to pick the best-epoch checkpoint.
CKPT_METRIC = str(_cfg("CKPT_METRIC"))
if CKPT_METRIC not in (
    "val_phi_spearman", "val_regret", "val_gain_mean", "val_loss",
    "val_top1_accuracy", "val_top5_accuracy", "val_rho_phi_image",
):
    raise ValueError(f"Unknown {CKPT_METRIC=}")
# Number of sample grids concatenated per training batch.
GRIDS_PER_BATCH = int(_cfg("GRIDS_PER_BATCH"))
# Exponential moving average of the weights.
EMA_DECAY = float(_cfg("EMA_DECAY"))

# Run directory name under RUNS_DIR. Empty string means use a timestamp.
RUN_NAME = str(_cfg("RUN_NAME"))


"""
Selector settings.
"""

# Scalar objective phi. Scores the training loss and ranks cells at selection
# time, so both stages optimize the same trade-off between the two metrics.
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

# Training-only phi. SCORE_PHI above stays the canonical scoreboard at PHI_ALPHA
# with equal weights, so reshaping the objective (sharpening the risk aversion,
# or paying more attention to CLIP) is a lever whose payoff is still measured on
# an unchanged target. Null falls back to the canonical phi exactly.
_TRAIN_PHI_ALPHA = _cfg("TRAIN_PHI_ALPHA", None)
TRAIN_PHI_ALPHA = PHI_ALPHA if _TRAIN_PHI_ALPHA is None else float(_TRAIN_PHI_ALPHA)
_TRAIN_PHI_WEIGHTS = _cfg("TRAIN_PHI_WEIGHTS", None)
TRAIN_PHI_WEIGHTS = None if _TRAIN_PHI_WEIGHTS is None else tuple(float(w) for w in _TRAIN_PHI_WEIGHTS)
if TRAIN_PHI_WEIGHTS is not None and len(TRAIN_PHI_WEIGHTS) != len(TARGET_COLS):
    raise ValueError(f"Expected {len(TARGET_COLS)} weights, got {TRAIN_PHI_WEIGHTS=}")
_TRAIN_SCORE_KW = {"alpha": TRAIN_PHI_ALPHA} if "alpha" in _signature(_SCORE_FN).parameters else {}
TRAIN_SCORE_PHI = partial(_SCORE_FN, **_TRAIN_SCORE_KW)  # Torch phi(Delta) for the loss only

# Selection-time levers over the already-predicted grid, so they can be swept
# without retraining. SELECT_PHI_WEIGHTS reweights phi for ranking only, and
# SELECT_CLIP_FLOOR restricts the argmax to cells clearing a CLIP delta.
# Reported metrics stay on the canonical phi either way.
_SELECT_PHI_WEIGHTS = _cfg("SELECT_PHI_WEIGHTS", None)
SELECT_PHI_WEIGHTS = None if _SELECT_PHI_WEIGHTS is None else tuple(float(w) for w in _SELECT_PHI_WEIGHTS)
_SELECT_CLIP_FLOOR = _cfg("SELECT_CLIP_FLOOR", None)
SELECT_CLIP_FLOOR = None if _SELECT_CLIP_FLOOR is None else float(_SELECT_CLIP_FLOOR)
