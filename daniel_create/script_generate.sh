#!/usr/bin/env bash
# Generate the entire t_start by t_end image set for every source image in the
# dataset. Shards the sample list across GPUs (one process per GPU, disjoint
# round-robin slices); each shard writes cells under OUTPUT_ROOT.
# Pass OVERVIEW_GRIDS=1 to also write grid_clean.png overviews.
#
# GPUS is required and must be set explicitly. Usage:
#   GPUS="0 1 2 3" bash daniel_create/script_generate.sh
#   GPUS="0 1 2 3 4 5 6 7" bash daniel_create/script_generate.sh
#   MAX_SAMPLES=10 GPUS="0" bash daniel_create/script_generate.sh
#   OVERVIEW_GRIDS=1 GPUS="0" bash daniel_create/script_generate.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/UltraEdit_Region_1000}"
# GPUS is a requirement; it never defaults to a multi-GPU list. When unset, fall
# back to the minimum requirement of a single GPU (index 0).
GPUS="${GPUS:-}"
if [[ -z "${GPUS}" ]]; then
  echo "[generate] GPUS not set; assuming the single-GPU requirement (GPUS=0)." >&2
  GPUS="0"
fi
# Call the env python directly (conda run doesn't forward signals and leaves orphans).
PY="${PY:-/data/home/mirick/miniconda3/envs/chordedit/bin/python}"

EXTRA_ARGS=()
[[ -n "${MAX_SAMPLES:-}" ]] && EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
[[ -n "${OVERVIEW_GRIDS:-}" ]] && EXTRA_ARGS+=(--overview-grids)

read -ra GPU_ARR <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARR[@]}"
echo "[generate] launching ${NUM_SHARDS} shard(s) on GPUs: ${GPUS}"

cleanup() { echo "[generate] stopping shards ..."; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM

pids=()
shard=0
for gpu in "${GPU_ARR[@]}"; do
  "${PY}" "${HERE}/generate_grid.py" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device "cuda:${gpu}" \
    --num-shards "${NUM_SHARDS}" \
    --shard "${shard}" \
    "${EXTRA_ARGS[@]}" &
  pids+=($!)
  shard=$((shard + 1))
done

rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done
echo "[generate] done (rc=${rc}) -> ${OUTPUT_ROOT}"
exit "${rc}"
