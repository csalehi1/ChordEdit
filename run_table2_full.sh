#!/usr/bin/env bash
set -euo pipefail

GPU=0
MAX_SAMPLES=700

MODEL_ROOT="$HOME/models/sd-turbo"
PIE_ROOT="$HOME/datasets/PIE-Bench_v1"
EXPORT_ROOT="/shared/ssd_30T/zarageddes/chordedit_original_ablations"

mkdir -p "$EXPORT_ROOT"

run_job () {
  local name="$1"
  shift

  echo
  echo "============================================================"
  echo "Running: $name"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=$GPU python run_pie_bench.py \
    --model-root "$MODEL_ROOT" \
    --pie-root "$PIE_ROOT" \
    --export-root "$EXPORT_ROOT" \
    --method-name "$name" \
    --max-samples "$MAX_SAMPLES" \
    --overwrite \
    "$@"
}

# Original ChordEdit paper Table 2-style comparison:
# Naive delta=0 vs Ours delta=0.15
# With and without proximal refinement.

run_job table2_naive_delta000_w_prox \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --cleanup

run_job table2_ours_delta015_w_prox \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --cleanup

run_job table2_naive_delta000_wo_prox \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --no-cleanup

run_job table2_ours_delta015_wo_prox \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --no-cleanup

echo
echo "Done. Outputs are under:"
echo "$EXPORT_ROOT/output"