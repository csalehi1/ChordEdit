#!/usr/bin/env bash
# Generate the 121-image t_start by t_end grid for every source image in
# UltraEdit_Region_1000 and score each cell with PnPInversion's psnr +
# clip_similarity_target_image_edit_part. Results go to outputs/UltraEdit_Region_1000.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_1000"
OUTPUT_ROOT="${REPO_ROOT}/outputs/UltraEdit_Region_1000"

conda run -n chordedit python "${REPO_ROOT}/scripts/daniel_run_pie_grid_pnp_metrics.py" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples 10
