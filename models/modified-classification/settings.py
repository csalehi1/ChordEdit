"""
Configuration for the metric-predictor model.

Unlike models/classification (which predicts the best timesteps from a prompt
pair), this model is a *surrogate* for the editing metrics: given the source
image, the prompt pair, and a candidate (t_start, t_end), it predicts what the
resulting PSNR and CLIP scores would be.

    M(img_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)
"""

from pathlib import Path

_PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = _PARENT_DIR / "data"

# Grid-ablation metrics for the 10-sample SD-Turbo subset. Each row is one
# (sample_id, t_delta, t_start, t_end) cell with its measured psnr/clip and a
# path to the rendered cell image (whose folder also holds the source image).
METRICS_CSV = DATA_DIR / "id_to_metrics_sdturbo_random10.csv"
STRINGS_CSV = DATA_DIR / "id_to_string_pair.csv"

_OUTPUTS_SUBDIR = METRICS_CSV.stem.removeprefix("id_to_metrics_")
OUTPUTS_DIR = _PARENT_DIR / "outputs" / _OUTPUTS_SUBDIR
if not OUTPUTS_DIR.exists():
    OUTPUTS_DIR.mkdir(parents=True)


"""
Data columns. The random10 CSV stores PSNR under `whole_psnr`; everything is
remapped to the canonical names below at load time.
"""

PSNR_COL = "whole_psnr"
CLIP_COL = "clip_edited"
IMAGE_PATH_COL = "image_path"

# Targets the model regresses, in order. Output tensor columns follow this list.
TARGET_COLS = ("psnr", "clip")
TARGET_LABELS = {"psnr": "Whole PSNR", "clip": "CLIP-Edited"}
TARGET_METRIC_COL = "-".join(c for c in TARGET_COLS)


# The source image lives this many parent directories above each cell image:
# <sample_dir>/t_delta_<x>/cells/<cell>.png  ->  <sample_dir>/source.png
SOURCE_IMAGE_NAME = "source.png"
SOURCE_IMAGE_PARENT_LEVEL = 2


"""
Data selection. The model signature takes only (t_start, t_end), so a single
t_delta slice is used. Set to None to train across every t_delta in the data.
"""

TARGET_T_DELTA = 0.0


"""
Encoders. A ChordEditPipeline is loaded from SD_TURBO_ROOT and reused for
text and VAE image encoding (same paths and preprocessing as inference).
"""

SD_TURBO_ROOT = Path("/shared/ssd_30T/mirick/sd-turbo")
IMAGE_SIZE = 512
USE_CENTER_CROP = True
FREEZE_ENCODERS = True


"""
Model architecture (regressor MLP body widths).
"""

MLP_WIDE = 256
MLP_HIDDEN = 128
MLP_INNER = 64
MLP_DROPOUT = 0.2


"""
Training hyperparameters. Targets are standardized (z-scored) using train-split
statistics so PSNR (~9-36) and CLIP (~0.06-0.35) contribute comparably to the
MSE objective; predictions are de-standardized before metrics are reported.
"""

SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.01
NORMALIZE_TARGETS = True
