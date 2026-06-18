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

# Pretrained transformer for the Siamese Encoder
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Computed metric from data
COMPUTED_METRIC_FN = compute_combined_score
COMPUTED_METRIC_COL = "combined_score"
COMPUTED_METRIC_LABEL = "Combined Score"

# Train on computed metric
TARGET_COLUMN = COMPUTED_METRIC_COL
DELTA_VALUE = 0.0

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
SEED = 42
