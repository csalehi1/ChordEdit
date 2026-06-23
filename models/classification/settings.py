from pathlib import Path

from models.classification.utils import (
    compute_combined_score,
    compute_weighted_combined_score,
    compute_agreement_score,
    compute_naive_pareto_score,
)

PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = PARENT_DIR / "data"

METRICS_CSV = DATA_DIR / "id_to_metrics_sdturbo.csv"
STRINGS_CSV = DATA_DIR / "id_to_string_pair.csv"

OUTPUTS_SUBDIR = "sdturbo"
OUTPUTS_DIR = PARENT_DIR / "outputs" / OUTPUTS_SUBDIR

# Expected number of distinct t_start/t_end levels in the training data.
# Values must be in [0, 1]. Raise an error at load time if the data differs.
N_BUCKETS_START = 11
N_BUCKETS_END = 11

# Baseline timestep bounds from the paper
PAPER_T_START = 0.9
PAPER_T_END  = 0.3
PAPER_T_DELTA = 0.15

# Computed metric to add to data from PSNR/CLIP
COMPUTED_METRIC_FN = compute_naive_pareto_score
COMPUTED_METRIC_COL = f"naive_pareto_score"
COMPUTED_METRIC_LABEL = f"Naive Pareto Score$"

# _lambda_psnr, _lambda_clip = 0.5, 0.5
# COMPUTED_METRIC_FN = lambda *args: compute_weighted_combined_score(lambda_psnr=_lambda_psnr, lambda_clip=_lambda_clip, *args)
# COMPUTED_METRIC_COL = f"combined_score_p{_lambda_psnr}_c_{_lambda_clip}"
# COMPUTED_METRIC_LABEL = f"Combined Score $\\lambda_{{\\text{{PSNR}}}}={_lambda_psnr}, \\lambda_{{\\text{{CLIP}}}}={_lambda_clip}$"

#
METRIC_COLS = ["psnr", "clip_target_similarity", COMPUTED_METRIC_COL]
METRIC_LABELS = {
    "psnr": "Whole PSNR",
    "clip_target_similarity": "CLIP Target Similarity",
    COMPUTED_METRIC_COL: COMPUTED_METRIC_LABEL,
}

# Train on computed metric
TARGET_COLUMN = COMPUTED_METRIC_COL
# The variable 
DELTA_VALUE = 0.15

# Pretrained transformer for the Siamese Encoder
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Prevent the encoding model from training
FREEZE_ENCODER = True
# Choose between "CORAL" or "MSE"
HEAD_TYPE = "CORAL"
# Weight loss by inverse class frequency to counteract imbalance
USE_CLASS_WEIGHTS = True   

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