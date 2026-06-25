from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from models.classification.utils import (
    compute_weighted_combined_score,
    compute_agreement_score,
    compute_naive_pareto_score,
    compute_pareto_biased_score,
)

"""
Paths. Root directories and CSV inputs derived from the location of this
file. The outputs subdirectory is named after the metrics CSV stem so
that different datasets write to isolated folders automatically.
"""

_PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = _PARENT_DIR / "data"

# NOTE: Adjust METRICS_CSV dependent on the data
METRICS_CSV = DATA_DIR / "id_to_metrics_sdturbo_tstart.csv"
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

N_BUCKETS_START = 11
N_BUCKETS_END = 1
# Value in `t_delta` column to select data from 
T_DELTA_TARGET = 0.0

PAPER_T_START = 0.9
PAPER_T_END = 0.3
PAPER_T_DELTA = 0.15


"""
Computed metrics. Each MetricOption bundles the DataFrame column name,
the callable that produces  it, and a display label. Parameterized
variants (e.g. weighted combined score)  use functools.partial so
every option has the same zero-argument-from-df call  signature. To
switch the metric used throughout training and evaluation, change the
key passed to _COMPUTED_METRIC_OPTIONS on the _ACTIVE line. The module-level
constants below it are then derived automatically.
"""


@dataclass(frozen=True)
class MetricOption:
    col: str
    fn: Callable
    label: str


# Possible computed metric options linked to their associated functions
_LAMBDA_PSNR, _LAMBDA_CLIP = 0.5, 0.5
_PARETO_BIAS_ALPHA = 2.0
_COMPUTED_METRIC_OPTIONS: dict[str, MetricOption] = {
    "weighted_combined_score": MetricOption(
        col="weighted_combined_score",
        fn=partial(compute_weighted_combined_score, lambda_psnr=_LAMBDA_PSNR, lambda_clip=_LAMBDA_CLIP),
        label=(
            "Combined Score" f" $\\lambda_{{\\text{{PSNR}}}}={_LAMBDA_PSNR}, \\lambda_{{\text{{CLIP}}}}={_LAMBDA_CLIP}$"
        ),
    ),
    "agreement_score": MetricOption(
        col="agreement_score",
        fn=compute_agreement_score,
        label="Agreement Score",
    ),
    "naive_pareto_score": MetricOption(
        col="naive_pareto_score",
        fn=compute_naive_pareto_score,
        label="Naive Pareto Score",
    ),
    "pareto_biased_score": MetricOption(
        col="pareto_biased_score",
        fn=partial(compute_pareto_biased_score, alpha=_PARETO_BIAS_ALPHA),
        label=f"Pareto Biased Score $\\alpha={_PARETO_BIAS_ALPHA}$",
    ),
}

# NOTE: May be "weighted_combined_score", "agreement_score",
# "naive_pareto_score", or "pareto_biased_score". Select preference.
_ACTIVE = _COMPUTED_METRIC_OPTIONS["pareto_biased_score"]
COMPUTED_METRIC_COL = _ACTIVE.col
COMPUTED_METRIC_FN = _ACTIVE.fn
COMPUTED_METRIC_LABEL = _ACTIVE.label


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
    COMPUTED_METRIC_COL: COMPUTED_METRIC_LABEL,
}


"""
Model architecture. ENCODER_MODEL names the HuggingFace checkpoint
used as the Siamese backbone. FREEZE_ENCODER prevents its weights from
updating during training; set to False to fine-tune end-to-end.
HEAD_TYPE selects between ordinal regression ("CORAL"), plain
mean-squared-error ("MSE"), and cost-sensitive multiclass CE ("CE").
USE_CLASS_WEIGHTS re-weights the loss by inverse class frequency to
counteract label imbalance in the training split.
"""

# Name of HuggingFace checkpoint for text-encoder
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Prevent encoder weights from updating during training
FREEZE_ENCODER = True
# Select head type to use for last step of model,
# NOTE: May be "CORAL", "MSE", or "CE"
HEAD_TYPE = "CE"
# Counteract label imbalance in the training split.
USE_CLASS_WEIGHTS = False
# Softens overconfident majority-class collapse in CE training.
LABEL_SMOOTHING = 0.1


"""
Training hyperparameters. TARGET_COLUMN is the regression/ordinal
target; ENCODER_LR and MLP_LR are kept separate because the encoder
backbone and the MLP head typically benefit from different learning
rates. MLP_WIDE / MLP_HIDDEN / MLP_INNER define the three hidden layer
widths of the head network.
"""

TARGET_COLUMN = COMPUTED_METRIC_COL

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32

ENCODER_LR = 2e-5
WEIGHT_DECAY = 0.01

MLP_WIDE = 512
MLP_HIDDEN = 256
MLP_INNER = 128
MLP_DROPOUT = 0.1
MLP_LR = 1e-3
