#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="$HOME/models/sd-turbo"
PIE_ROOT="$HOME/datasets/PIE-Bench_v1"
EXPORT_ROOT="/shared/ssd_30T/zarageddes/llm_timestep_policy"

GPU="${GPU:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"

T_END="0.30"
T_DELTA="0.0"
STEP_SCALE="1.0"

for T_START in 0.30 0.40 0.50 0.60 0.70 0.80 0.90; do
  METHOD_NAME="pilot_t${T_START//./}_delta0_tc030"

  echo "Running ${METHOD_NAME}"

  CUDA_VISIBLE_DEVICES="$GPU" python run_pie_bench.py \
    --model-root "$MODEL_ROOT" \
    --pie-root "$PIE_ROOT" \
    --export-root "$EXPORT_ROOT" \
    --method-name "$METHOD_NAME" \
    --t-start "$T_START" \
    --t-delta "$T_DELTA" \
    --t-end "$T_END" \
    --step-scale "$STEP_SCALE" \
    --max-samples "$MAX_SAMPLES" \
    --overwrite
done
