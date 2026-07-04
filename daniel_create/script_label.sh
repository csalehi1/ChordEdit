#!/usr/bin/env bash
# Label (score) an entire already-generated image set with PSNR + CLIP metrics.
# Shards the sample list across GPUs; each shard writes result_shard<NN>.csv plus
# grid_psnr/grid_clip overlays, then the shard CSVs are merged into result.csv.
#
# Run script_generate.sh first. GPUS is required and must be set explicitly. Usage:
#   GPUS="0 1 2 3" bash daniel_create/script_label.sh
#   GPUS="0 1 2 3 4 5 6 7" bash daniel_create/script_label.sh
#   MAX_SAMPLES=10 GPUS="0" bash daniel_create/script_label.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/UltraEdit_Region_1000}"
# GPUS is a requirement; it never defaults to a multi-GPU list. When unset, fall
# back to the minimum requirement of a single GPU (index 0).
GPUS="${GPUS:-}"
if [[ -z "${GPUS}" ]]; then
  echo "[label] GPUS not set; assuming the single-GPU requirement (GPUS=0)." >&2
  GPUS="0"
fi
PY="${PY:-/data/home/mirick/miniconda3/envs/chordedit/bin/python}"

EXTRA_ARGS=()
[[ -n "${MAX_SAMPLES:-}" ]] && EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")

read -ra GPU_ARR <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARR[@]}"
echo "[label] launching ${NUM_SHARDS} shard(s) on GPUs: ${GPUS}"

cleanup() { echo "[label] stopping shards ..."; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM

pids=()
shard=0
for gpu in "${GPU_ARR[@]}"; do
  "${PY}" "${HERE}/label_grid.py" \
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

# Merge per-shard CSVs into result.csv (header once, then all data rows).
if [[ "${NUM_SHARDS}" -gt 1 ]]; then
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
  echo "[label] merged shard CSVs -> ${merged}"
fi

echo "[label] done (rc=${rc}) -> ${OUTPUT_ROOT}"
exit "${rc}"
