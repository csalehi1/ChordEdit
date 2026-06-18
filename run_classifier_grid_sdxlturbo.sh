#!/usr/bin/env bash
set -euo pipefail

cd ~/research/ChordEdit
source ~/venvs/chordedit/bin/activate

export HF_HOME=/shared/ssd_30T/zarageddes/hf_cache
export TRANSFORMERS_CACHE=/shared/ssd_30T/zarageddes/hf_cache
export HF_HUB_CACHE=/shared/ssd_30T/zarageddes/hf_cache/hub
export HF_HUB_DISABLE_XET=1

MODEL_ROOT=/shared/ssd_30T/zarageddes/models/sdxl-turbo
PIE_ROOT=~/datasets/PIE-Bench_v1
EXPORT_ROOT=/shared/ssd_30T/zarageddes/llm_timestep_policy

for i in 0 1 2 3 4 5 6 7 8 9 10; do
  T_START=$(python - <<PY
i = $i
print(f"{i/10:.1f}")
PY
)
  METHOD_NAME=$(printf "classifier_full_sdxlturbo_t%02d_delta0_tc030" "$i")

  echo "===== Running ${METHOD_NAME}, t_start=${T_START} ====="

  python run_pie_bench.py \
    --model-root "$MODEL_ROOT" \
    --model-type auto \
    --pie-root "$PIE_ROOT" \
    --export-root "$EXPORT_ROOT" \
    --method-name "$METHOD_NAME" \
    --t-start "$T_START" \
    --t-end 0.3 \
    --t-delta 0.0 \
    --image-size 512 \
    --copy-source \
    --overwrite \
    --no-safety-checker \
    --log-every 25
done
