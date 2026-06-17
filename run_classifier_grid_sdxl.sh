#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="/shared/ssd_30T/zarageddes/models/sdxl-base-1.0"
PIE_ROOT="$HOME/datasets/PIE-Bench_v1"
EXPORT_ROOT="/shared/ssd_30T/zarageddes/llm_timestep_policy"

MAPPING_FILE="mapping_file.json"

GPU="${GPU:-6}"

T_END="0.30"
T_DELTA="0.0"
STEP_SCALE="1.0"

for T_START in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  T_TAG=$(echo "$T_START" | tr -d '.')
  METHOD_NAME="classifier_full_sdxl_t${T_TAG}_delta0_tc030"

  echo "============================================================"
  echo "Running $METHOD_NAME"
  echo "t_start=$T_START delta=$T_DELTA t_end=$T_END"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$GPU" python run_pie_bench.py \
    --model-root "$MODEL_ROOT" \
    --pie-root "$PIE_ROOT" \
    --mapping-file "$MAPPING_FILE" \
    --export-root "$EXPORT_ROOT" \
    --method-name "$METHOD_NAME" \
    --t-start "$T_START" \
    --t-delta "$T_DELTA" \
    --t-end "$T_END" \
    --step-scale "$STEP_SCALE" \
    --overwrite
done
