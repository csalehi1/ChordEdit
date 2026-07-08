#!/usr/bin/env bash
# Generate the entire image set and then label it end to end. Environment
# variables (GPUS, DATA_ROOT, OUTPUT_ROOT, MAX_SAMPLES, OVERVIEW_GRIDS, PY) are
# forwarded to both stages.
#
# GPUS is required and must be set explicitly. Usage:
#   GPUS="0 1 2 3" bash daniel_create/script_both.sh
#   GPUS="0 1 2 3 4 5 6 7" bash daniel_create/script_both.sh
#   MAX_SAMPLES=10 GPUS="0" bash daniel_create/script_both.sh
#   OVERVIEW_GRIDS=1 GPUS="0" bash daniel_create/script_both.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[both] === stage 1/2: generate ==="
bash "${HERE}/script_generate.sh"

echo "[both] === stage 2/2: label ==="
bash "${HERE}/script_label.sh"

echo "[both] done"
