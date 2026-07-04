#!/usr/bin/env bash
# Run the UltraEdit_Region_1000 grid ablation across multiple GPUs by sharding the
# sample list: one process per GPU, each handling a disjoint round-robin slice.
# Each shard writes its own result_shard<NN>.csv; they are merged into result.csv
# at the end. Cells/grids go under the shared output root (disjoint sample_ids).
#
# Usage:
#   bash scripts/daniel_run_optimized_grid_multi_gpu.sh              # GPUs 0-3
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/daniel_run_optimized_grid_multi_gpu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_1000"
OUTPUT_ROOT="${REPO_ROOT}/outputs/UltraEdit_Region_1000"
GPUS="${GPUS:-0 1 2 3}"
# Call the env python directly (not `conda run`, which doesn't forward signals and
# leaves orphaned workers on Ctrl-C).
PY="${PY:-/data/home/mirick/miniconda3/envs/chordedit/bin/python}"

read -ra GPU_ARR <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARR[@]}"
echo "[multi_gpu] launching ${NUM_SHARDS} shard(s) on GPUs: ${GPUS}"

# Ensure Ctrl-C / termination kills the background shard workers too (otherwise
# they keep running detached and collide with a re-launch).
cleanup() { echo "[multi_gpu] stopping shards ..."; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM

pids=()
shard=0
for gpu in "${GPU_ARR[@]}"; do
  "${PY}" "${REPO_ROOT}/scripts/daniel_run_pie_grid_pnp_metrics.py" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device "cuda:${gpu}" \
    --num-shards "${NUM_SHARDS}" \
    --shard "${shard}" &
  pids+=($!)
  shard=$((shard + 1))
done

# Wait for all shards; fail if any shard failed.
rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done

# Merge per-shard CSVs into a single result.csv (header once, then all data rows).
merged="${OUTPUT_ROOT}/result.csv"
first=1
: > "${merged}"
for f in "${OUTPUT_ROOT}"/result_shard*.csv; do
  [[ -e "${f}" ]] || continue
  if [[ "${first}" -eq 1 ]]; then
    cat "${f}" >> "${merged}"
    first=0
  else
    tail -n +2 "${f}" >> "${merged}"
  fi
done
echo "[multi_gpu] merged shard CSVs -> ${merged}"

exit "${rc}"
