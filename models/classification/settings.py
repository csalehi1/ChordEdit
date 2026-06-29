from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from models.classification.utils import (
    compute_weighted_combined_score,
    compute_agreement_score,
    compute_naive_pareto_score,
    compute_softplus_score,
)

"""
Paths. Root directories and CSV inputs derived from the location of this
file. The outputs subdirectory is named after the metrics CSV stem so
that different datasets write to isolated folders automatically.
"""

_PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = _PARENT_DIR / "data"

# NOTE: Adjust METRICS_CSV dependent on the data
METRICS_CSV = DATA_DIR / "id_to_metrics_sdturbo-tstart.csv"
STRINGS_CSV = DATA_DIR / "id_to_string_pair.csv"

# Files for image data should be named `id_to_metrics_*`
_OUTPUTS_SUBDIR = METRICS_CSV.stem.removeprefix("id_to_metrics_")
OUTPUTS_DIR = _PARENT_DIR / "outputs" / _OUTPUTS_SUBDIR


"""
Parameters that describe the shape and structure of the training data.
N_BUCKETS_* defines the expected number of distinct timestep levels; the
loader raises at import time if the data does not match. The PAPER_T_*
constants reproduce the baseline timestep bounds from the original paper
and are used by the Pareto score functions to identify the reference row
within each sample group.
"""

# NOTE: May be "weighted_combined_score", "agreement_score",
# "naive_pareto_score", or "softplus_score". Choose one.
TARGET_METRIC = "softplus_score"

# NOTE: Must match number of distinct `t_start`, `t_end` values in METRICS_CSV
N_BUCKETS_START = 11
N_BUCKETS_END = 1

# Value in `t_delta` column to select data from 
TARGET_T_DELTA = 0.0

# By default, use DEFAULT_T_START = PAPER_T_START - (PAPER_T_DELTA - TARGET_T_DELTA)
DEFAULT_T_START = 0.9
DEFAULT_T_END = 0.3

# Values from the original paper, should *not* be modified
PAPER_T_START = 0.9
PAPER_T_END = 0.3
PAPER_T_DELTA = 0.15


"""
Computed metrics. Set TARGET_METRIC to one of the keys in _METRIC_REGISTRY.
TARGET_METRIC_COL is the column name for the active metric.
"""


@dataclass(frozen=True)
class MetricOption:
    fn: Callable
    col: str
    label: str


# Data to connect TARGET_METRIC to the appropriate metric function and label.
_LAMBDA_PSNR, _LAMBDA_CLIP = 0.5, 0.5
_PARETO_BIAS_ALPHA = 2.0
_SOFTPLUS_ALPHA, _SOFTPLUS_BETA = 1.0, 2.0
_NORMALIZE = False
_METRIC_REGISTRY: dict[str, MetricOption] = {
    "weighted_combined_score": MetricOption(
        fn=partial(compute_weighted_combined_score, lambda_psnr=_LAMBDA_PSNR, lambda_clip=_LAMBDA_CLIP),
        col=f"weighted_combined_score_lc{_LAMBDA_CLIP:g}-lp{_LAMBDA_PSNR:g}",
        label=f"Combined Score (\\lambda_{{\\text{{PSNR}}}}={_LAMBDA_PSNR}, \\lambda_{{\\text{{CLIP}}}}={_LAMBDA_CLIP})",
    ),
    "agreement_score": MetricOption(
        fn=compute_agreement_score,
        col="agreement_score",
        label="Agreement Score",
    ),
    "naive_pareto_score": MetricOption(
        fn=partial(compute_naive_pareto_score, normalize=_NORMALIZE),
        col=f"naive_pareto_score_n{_NORMALIZE:d}",
        label=f"Naive Pareto Score (n={_NORMALIZE:d})",
    ),
    "softplus_score": MetricOption(
        fn=partial(compute_softplus_score, alpha=_SOFTPLUS_ALPHA, beta=_SOFTPLUS_BETA, normalize=_NORMALIZE),
        col=f"softplus_score_a{_SOFTPLUS_ALPHA:g}-b{_SOFTPLUS_BETA:g}-n{_NORMALIZE:d}",
        label=f"Softplus Score ($\\alpha={_SOFTPLUS_ALPHA}$, $\\beta={_SOFTPLUS_BETA}$, n={_NORMALIZE:d})",
    ),
}

if TARGET_METRIC not in _METRIC_REGISTRY:
    raise ValueError(
        f"Unknown TARGET_METRIC={TARGET_METRIC!r}; "
        f"choose from {sorted(_METRIC_REGISTRY)}"
    )

_active = _METRIC_REGISTRY[TARGET_METRIC]
TARGET_METRIC_COL = _active.col
TARGET_METRIC_COL_FN = _active.fn
TARGET_METRIC_COL_LABEL = _active.label
METRIC_REGISTRY = _METRIC_REGISTRY


"""
Metric columns. _METRICS maps the base signal column names to their
display labels. METRIC_LABELS extends that mapping with the active
computed metric so any plot or table that iterates over all tracked
columns can use a single dict.
"""

_METRICS = {
    "psnr": "Whole PSNR",
    "clip_edited": "CLIP-Edited",
}

METRIC_COLS = list(_METRICS.keys())
METRIC_LABELS = {
    **_METRICS,
    TARGET_METRIC_COL: TARGET_METRIC_COL_LABEL,
}


"""
Model architecture. ENCODER_MODEL names the HuggingFace checkpoint
used as the Siamese backbone. FREEZE_ENCODER prevents its weights from
updating during training; set to False to fine-tune end-to-end.
HEAD_TYPE selects between ordinal regression ("CORAL"), plain
mean-squared-error ("MSE"), and cost-sensitive multiclass CE ("CE").
When HEAD_TYPE is "CE", CE_LOSS_TYPE selects the CE loss function.
USE_CLASS_WEIGHTS re-weights the loss by inverse class frequency to
counteract label imbalance in the training split.
"""

# Name of HuggingFace checkpoint for text-encoder
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
USE_CLASS_WEIGHTS = False
# Softens overconfident majority-class collapse in CE training.
LABEL_SMOOTHING = 0.1

"""
Training hyperparameters. ENCODER_LR and MLP_LR are kept separate because
the encoder backbone and the MLP head typically benefit from different
learning rates. MLP_WIDE / MLP_HIDDEN / MLP_INNER define the three hidden
layer widths of the head network.
"""

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32

# Only used when FREEZE_ENCODER is False
ENCODER_LR = 2e-5

# Body of the model
MLP_WIDE = 512
MLP_HIDDEN = 256
MLP_INNER = 128
MLP_DROPOUT = 0.1
MLP_LR = 1e-3
WEIGHT_DECAY = 0.01
