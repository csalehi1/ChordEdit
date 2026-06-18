from pathlib import Path

from models.classification.utils import (
    compute_combined_score,
    compute_agreement_score
)

PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = PARENT_DIR / "data"
METRICS_CSV = DATA_DIR / "id_to_metrics_sdturbo.csv"
STRINGS_CSV = DATA_DIR / "id_to_string_pair.csv"
OUTPUTS_DIR = PARENT_DIR / "outputs/sdturbo"

# Expected number of distinct t_start/t_end levels in the training data.
# Values must be in [0, 1]. Raise an error at load time if the data differs.
N_BUCKETS_START = 11
N_BUCKETS_END = 1

# Computed metric from data
COMPUTED_METRIC_FN = compute_combined_score
COMPUTED_METRIC_COL = "combined_score"
COMPUTED_METRIC_LABEL = "Combined Score"

# Train on computed metric
TARGET_COLUMN = COMPUTED_METRIC_COL
DELTA_VALUE = 0.0

# Pretrained transformer for the Siamese Encoder
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FREEZE_ENCODER = False

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
ENCODER_LR = 2e-5
BODY_LR = 1e-3
