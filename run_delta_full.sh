#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Original ChordEdit SD-Turbo ablation pilot
# ============================================================
#
# This script reruns the original-paper ablations that are
# compatible with the SD-Turbo setup we are using.
#
# Pilot mode:
#   MAX_SAMPLES=20
#
# Full benchmark later:
#   change MAX_SAMPLES=20 to MAX_SAMPLES=700
#
# Outputs are written to /shared/ssd_30T so we do not fill /data.
# ============================================================

GPU=1
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

# ============================================================
# 1. Table 2-style core ablation:
#    Naive delta=0 vs Ours delta=0.15,
#    with and without proximal refinement.
# ============================================================

run_job sweep_delta_000 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.00 --cleanup

run_job sweep_delta_005 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.05 --cleanup

run_job sweep_delta_010 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.10 --cleanup

run_job sweep_delta_015 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --cleanup

run_job sweep_delta_020 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.20 --cleanup

run_job sweep_delta_025 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.25 --cleanup




echo
echo "============================================================"
echo "All original SD-Turbo ablation pilot runs finished."
echo "Outputs are under:"
echo "$EXPORT_ROOT/output"
echo "============================================================"