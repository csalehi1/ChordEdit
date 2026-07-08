#!/usr/bin/env bash
# Generate then label. Forwards all args to both stages.
#
#   bash daniel_create/script_both.sh --gpus 0 1 2 3
#   bash daniel_create/script_both.sh --max-samples 10 --grids --gpus 0
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[both] === generate ==="
bash "${HERE}/script_generate.sh" "$@"
echo "[both] === label ==="
bash "${HERE}/script_label.sh" "$@"
echo "[both] done"
