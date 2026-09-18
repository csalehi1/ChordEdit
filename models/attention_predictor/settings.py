# settings.py

"""
Load settings from the settings.json file.
"""

import json as _json
import sys as _sys
from functools import partial
from inspect import signature as _signature
from pathlib import Path

from scores import cara_score, linex_score, naive_score


_SETTINGS_DIR = Path(__file__).resolve().parent

def _settings_path_from_argv() -> str | None:
    """Value of --settings-path on the command line, or None if absent."""
    argv = _sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--settings-path":
            if i + 1 >= len(argv):
                raise ValueError(f"Expected a path after --settings-path")
            return argv[i + 1]
        if arg.startswith("--settings-path"):
            return arg.split("=", 1)[1]
    return None


def _pie_bench_from_argv() -> bool:
    """True when --pie-bench is on the command line."""
    return "--pie-bench" in _sys.argv[1:]

# load_run_settings sets SETTINGS_PATH_OVERRIDE so replaying a snapshot does not depend on argv.
SETTINGS_JSON = Path(globals().get("SETTINGS_PATH_OVERRIDE") or _settings_path_from_argv() or _SETTINGS_DIR / "settings.json")
CONFIG: dict = {k: v for k, v in _json.loads(SETTINGS_JSON.read_text()).items() if not k.startswith("_")}


"""
Dataset settings.
"""

_CHORD_EDIT_MODEL_CONFIGS = {
    "sd_turbo": (Path("/shared/ssd_30T/mirick/models/sd-turbo"), 512, "sd"),
    "sdxl_turbo": (Path("/shared/ssd_30T/zarageddes/models/sdxl-turbo"), 1024, "sdxl"),
    "flux": (Path("/shared/ssd_30T/zarageddes/models/flux1-schnell"), 1024, "flux"),
}
# Choose from "sd_turbo", "sdxl_turbo", or "flux".
CHORD_EDIT_MODEL = str(CONFIG["CHORD_EDIT_MODEL"])
_, CHORD_EDIT_IMAGE_SIZE, CHORD_EDIT_PIPELINE_TYPE = _CHORD_EDIT_MODEL_CONFIGS[CHORD_EDIT_MODEL]

# Choose from "PIE_Bench_v1", "UltraEdit_Region_<N>", etc..
DIR_NAME = str(CONFIG["DIR_NAME"])
GENERATED_DIR = Path(f"/shared/ssd_30T/mirick/generated/{CHORD_EDIT_MODEL}/0p0/{DIR_NAME}")
DATASET_DIR = Path(f"/shared/ssd_30T/mirick/datasets/ultra_edit/{DIR_NAME}")
SCATTERED_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/{CHORD_EDIT_MODEL}/{DIR_NAME}")

# Which t_delta values to train on, or null to use every value.
TARGET_T_DELTA = float(CONFIG["TARGET_T_DELTA"]) if CONFIG["TARGET_T_DELTA"] is not None else None

# The fraction of samples to train on, or null to use all samples.
TRAIN_FRAC = float(CONFIG["TRAIN_FRAC"]) if CONFIG["TRAIN_FRAC"] is not None else None
VAL_FRAC = float(CONFIG["VAL_FRAC"])

# Separate seeds so that the model can be reseeded without moving samples between splits.
SEED = int(CONFIG["SEED"])
SPLIT_SEED = int(CONFIG.get("SPLIT_SEED", 42))

# Generation seeds averaged into the training metrics, and the seeds the val/test
# metrics are averaged from, so a held-out generation seed can score the model.
def _metrics_seeds(key: str) -> tuple[int, ...]:
    raw = CONFIG[key]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty list for {key}, got {raw!r}")
    return tuple(int(v) for v in raw)

TRAIN_METRICS_SEEDS = _metrics_seeds("TRAIN_METRICS_SEEDS")
EVAL_METRICS_SEEDS = _metrics_seeds("EVAL_METRICS_SEEDS")

# The maximum number of samples to train on, or null to use all samples.
MAX_SAMPLES = int(CONFIG["MAX_SAMPLES"]) if CONFIG["MAX_SAMPLES"] is not None else None

# Replace the UltraEdit test split with labeled PIE-Bench samples.
PIE_BENCH = _pie_bench_from_argv() or bool(CONFIG["PIE_BENCH"])
CONFIG["PIE_BENCH"] = PIE_BENCH
PIE_BENCH_DIR_NAME = "PIE_Bench_v1"
PIE_SAMPLE_ID_PREFIX = "pie_"
PIE_GENERATED_DIR = Path(f"/shared/ssd_30T/mirick/generated/{CHORD_EDIT_MODEL}/0p0/{PIE_BENCH_DIR_NAME}")
PIE_SCATTERED_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/{CHORD_EDIT_MODEL}/{PIE_BENCH_DIR_NAME}")
PIE_INPUTS_CSV = PIE_GENERATED_DIR / "id_to_inputs_piebenchv1.csv"
PIE_METRICS_CSV = PIE_GENERATED_DIR / "id_to_metrics_piebenchv1.csv"

_here = Path(__file__).resolve().parent
_package_dir = _here if (_here / "model.py").exists() else _here.parents[2]

# The directory where model runs are saved.
RUNS_DIR = _package_dir / "runs" / DIR_NAME
RUNS_DIR.mkdir(parents=True, exist_ok=True)
_runs_gitignore = RUNS_DIR.parent / ".gitignore"
if not _runs_gitignore.exists():
    _runs_gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

# The shared sample-id column name.
SAMPLE_ID_COL = "sample_id"

# The column names in the id_to_inputs CSV.
INPUTS_CSV = GENERATED_DIR / f"id_to_inputs_{DIR_NAME.replace("_", "").lower()}.csv"
SOURCE_PROMPT_COL = "source_prompt"
TARGET_PROMPT_COL = "target_prompt"
IMAGE_PATH_COL = "image_path"
MASK_PATH_COL = "mask_image_path"

# The column names in the id_to_metrics CSV.
METRICS_CSV = GENERATED_DIR / f"id_to_metrics_{DIR_NAME.replace("_", "").lower()}.csv"
T_START_COL = "t_start"
T_END_COL = "t_end"
T_DELTA_COL = "t_delta"
SEED_COL = "seed"
PSNR_COL = "psnr_unedit_part"
CLIP_COL = "clip_similarity_target_image_edit_part"

# The regression targets in the loaded dataframe.
TARGET_COLS = (PSNR_COL, CLIP_COL)


"""
Embedding settings.
"""

# Choose from "vae", "clip", or "vae_clip".
IMG_EMB_TYPE = str(CONFIG["IMG_EMB_TYPE"])

# Pool the visual embedding to a single key/value token.
IMG_EMB_POOL = bool(CONFIG["IMG_EMB_POOL"])

# How the image reaches the edit descriptor: "attn" grounds the prompt tokens in the visual
# tokens by cross-attention, "concat" appends the mean projected visual token as a z_edit
# segment instead, and "none" drops the image path. With one pooled visual token the
# attention has two keys (image, null) and reduces to a gate, so pooled runs use "concat".
IMG_FUSION = str(CONFIG.get("IMG_FUSION", "attn"))
if IMG_FUSION not in ("attn", "concat", "none"):
    raise ValueError(f"Expected IMG_FUSION in ('attn', 'concat', 'none'), got {IMG_FUSION!r}")

# Masked-image CLIP features, the only inputs that express what CLIP-Edited measures:
# "block" is the 768-d masked projection plus five cosine/area scalars as a z_edit segment,
# "tokens" adds the masked-image CLIP tokens as cross-attention keys and keeps only the
# scalars as a segment, "both" does both, "none" drops them. Older snapshots set
# USE_IMG_MASK / USE_ZEDIT_MASK.
MASK_FEATURES = str(CONFIG.get("MASK_FEATURES", "block" if CONFIG.get("USE_IMG_MASK") and CONFIG.get("USE_ZEDIT_MASK") else "none"))
if MASK_FEATURES not in ("none", "block", "tokens", "both"):
    raise ValueError(f"Expected MASK_FEATURES in ('none', 'block', 'tokens', 'both'), got {MASK_FEATURES!r}")

# Pool each prompt to a single masked-mean query; false keeps the full token sequences.
TEXT_EMB_POOL = bool(CONFIG["TEXT_EMB_POOL"])

# Apply a per-token MLP to the projected prompt tokens before pooling. Without it, a linear
# projection followed by the masked mean is the pooled prompt embedding.
TEXT_TOKEN_MLP = bool(CONFIG.get("TEXT_TOKEN_MLP", False))

# Add a second pooled vector per prompt, weighted by novelty against the other prompt.
USE_DIFF_SALIENCY = bool(CONFIG["USE_DIFF_SALIENCY"])

# Scale each residual branch toward identity at init; null means a plain residual add.
LAYERSCALE_INIT = float(CONFIG["LAYERSCALE_INIT"]) if CONFIG["LAYERSCALE_INIT"] is not None else None


"""
Regression settings.

The heads regress deltas versus the default cell in shared units, raw PSNR/CLIP divided by
one scale per column, so predictions invert exactly to dB and CLIP points. Selection
re-normalizes the predicted raw grid per sample (the paper's phi input), so nothing here
changes the scoreboard.
"""

# "median_range" scales each column by the train-median per-sample grid range;
# "custom" uses LOSS_SCALE_VALUES in raw units (e.g. the seed-noise sd per column).
LOSS_SCALE = str(CONFIG.get("LOSS_SCALE", "median_range"))
if LOSS_SCALE not in ("median_range", "custom"):
    raise ValueError(f"Expected LOSS_SCALE in ('median_range', 'custom'), got {LOSS_SCALE!r}")
LOSS_SCALE_VALUES = tuple(float(v) for v in CONFIG.get("LOSS_SCALE_VALUES") or []) or None
if LOSS_SCALE == "custom" and (LOSS_SCALE_VALUES is None or len(LOSS_SCALE_VALUES) != 2):
    raise ValueError(f"LOSS_SCALE custom needs two LOSS_SCALE_VALUES, got {LOSS_SCALE_VALUES}")

# The heads predict the per-image deviation from the train-mean surface ("deviation", the
# mean is added back inside the model) or the full delta surface ("deltas").
HEAD_PARAM = str(CONFIG.get("HEAD_PARAM", "deltas"))
if HEAD_PARAM not in ("deltas", "deviation"):
    raise ValueError(f"Expected HEAD_PARAM in ('deltas', 'deviation'), got {HEAD_PARAM!r}")

# Clamp deltas to [-CLAMP_DELTAS, CLAMP_DELTAS] before phi, in the loss and at selection.
CLAMP_DELTAS = float(CONFIG["CLAMP_DELTAS"]) if CONFIG.get("CLAMP_DELTAS") is not None else None

# These levers reweight, floor, and gate ranking at selection time.
DELTA_WEIGHTS = tuple(float(w) for w in list(CONFIG["DELTA_WEIGHTS"] or [])) or None
DELTA_FLOORS = tuple(float(v) if v is not None else None for v in list(CONFIG["DELTA_FLOORS"] or [])) or None
PHI_FLOOR = float(CONFIG["PHI_FLOOR"]) if CONFIG["PHI_FLOOR"] is not None else None
TEMPERATURE = float(CONFIG["TEMPERATURE"]) if CONFIG["TEMPERATURE"] is not None else None


"""
Model settings.
"""

# Zero the default cell on predicted head outputs.
PIN_DEFAULT_CELL = bool(CONFIG["PIN_DEFAULT_CELL"])

# Shared dimension of the visual tokens, prompt queries, and edit descriptor h.
ATTN_DIM = int(CONFIG["ATTN_DIM"])

# Number of cross-attention heads that ground the prompts in the image.
N_HEADS = int(CONFIG["N_HEADS"])

# Number of pre-LN cross-attention/FFN blocks that ground the prompt queries.
ATTN_LAYERS = int(CONFIG["ATTN_LAYERS"])

# Side of each square latent patch used as a cross-attention key/value.
LATENT_SIDE = CHORD_EDIT_IMAGE_SIZE // 8
PATCH_SIZE = int(CONFIG["PATCH_SIZE"])

# Dropout rate for the attention and FFN layers.
ATTN_DROPOUT = float(CONFIG["ATTN_DROPOUT"])

# The FFN hidden width as a multiple of ATTN_DIM, and 0 drops the FFN.
FFN_MULT = float(CONFIG["FFN_MULT"])

# Hidden width of the combiner C_theta that maps z_edit to the edit descriptor h.
COMBINER_HIDDEN = int(CONFIG["COMBINER_HIDDEN"])
COMBINER_DROPOUT = float(CONFIG["COMBINER_DROPOUT"])

# Give PSNR/CLIP their own combiners so that a shared h is not dominated by the PSNR gradient.
SPLIT_COMBINER = bool(CONFIG["SPLIT_COMBINER"])

# Hidden width of each metric head, 0 means Linear(d, n_cells).
HEAD_HIDDEN = int(CONFIG["HEAD_HIDDEN"])


"""
Training settings.
"""

EPOCHS = int(CONFIG["EPOCHS"])
LR = float(CONFIG["LR"])
WEIGHT_DECAY = float(CONFIG["WEIGHT_DECAY"])
# Choose from "none" or "cosine".
LR_SCHEDULER = str(CONFIG["LR_SCHEDULER"])
EARLY_STOP_PATIENCE = int(CONFIG["EARLY_STOP_PATIENCE"])
EMA_DECAY = float(CONFIG["EMA_DECAY"])
# Choose from "val_dev_corr", "val_phi_spearman", "val_regret", "val_gain_mean", "val_loss", "val_top1_accuracy", "val_top5_accuracy", "val_rho_phi_image".
CKPT_METRIC = str(CONFIG["CKPT_METRIC"])
SAMPLES_PER_BATCH = int(CONFIG["SAMPLES_PER_BATCH"])

# Configurations for the training loss, all in the regression space.
MSE_LOSS_WEIGHT = float(CONFIG["MSE_LOSS_WEIGHT"])
RANKING_LOSS_WEIGHT = float(CONFIG["RANKING_LOSS_WEIGHT"])
RANKING_LOSS_TOP_K = int(CONFIG["RANKING_LOSS_TOP_K"]) if CONFIG["RANKING_LOSS_TOP_K"] is not None else None
COL_LOSS_WEIGHTS = tuple(float(w) for w in list(CONFIG["COL_LOSS_WEIGHTS"]))
# Weight of 1 - corr(predicted deviation, true deviation), the deviation of each cell across
# the samples of a batch; scale-free, so it scores the direction of the per-image structure.
DEV_LOSS_WEIGHT = float(CONFIG.get("DEV_LOSS_WEIGHT", 0.0))

"""
Selector settings.
"""

# SCORE_FN is the scalar phi used for both the training loss and ranking at selection time.
SCORE_FNS = {
    "naive": naive_score,
    "cara": cara_score,
    "linex": linex_score,
}
SCORE_FN = str(CONFIG["SCORE_FN"])
_SCORE_FN = SCORE_FNS[SCORE_FN]

PHI_ALPHA = float(CONFIG["PHI_ALPHA"])
PHI_WEIGHTS = tuple(float(w) for w in list(CONFIG["PHI_WEIGHTS"] or [])) or None
_SCORE_KW = {"alpha": PHI_ALPHA} if "alpha" in _signature(_SCORE_FN).parameters else {}
SCORE_PHI = partial(_SCORE_FN, **_SCORE_KW)  # Torch phi(Delta)

# DEFAULT_T_START and DEFAULT_T_END are the ChordEdit baseline bounds and must match (0.85+t_delta, 0.3).
DEFAULT_T_START = float(CONFIG["DEFAULT_T_START"])
DEFAULT_T_END = float(CONFIG["DEFAULT_T_END"])

# If true, the loss, the selector and eval stats only consider cells with t_start > t_end.
USE_DIAGONAL_MASK = bool(CONFIG.get("USE_DIAGONAL_MASK", False))

# Select on the calibrated raw predictions (deviation amplitudes rescaled per column by the
# slopes fitted on val after training) rather than the raw head outputs.
SELECT_CALIBRATED = bool(CONFIG.get("SELECT_CALIBRATED", False))

# The run directory name under RUNS_DIR, and an empty string uses a timestamp.
RUN_NAME = str(CONFIG["RUN_NAME"])
# Extra wandb tags; an empty list adds nothing beyond the automatic ones.
RUN_TAGS = [str(t) for t in list(CONFIG["RUN_TAGS"] or [])]
