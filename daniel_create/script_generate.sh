#!/usr/bin/env bash
# Generate cells. Thin wrapper; sharding is handled by --gpus in generate_grid.py.
#
#   bash daniel_create/script_generate.sh --gpus 0 1 2 3
#   bash daniel_create/script_generate.sh --max-samples 10 --grids --gpus 0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/data/home/mirick/miniconda3/envs/chordedit/bin/python}"

exec "${PY}" "${HERE}/generate_grid.py" "$@"
