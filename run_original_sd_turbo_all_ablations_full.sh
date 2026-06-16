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

run_job table2_naive_delta000_w_prox \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --cleanup

run_job table2_ours_delta015_w_prox \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --cleanup

run_job table2_naive_delta000_wo_prox \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --no-cleanup

run_job table2_ours_delta015_wo_prox \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --no-cleanup


# ============================================================
# 2. Delta sweep:
#    Vary chord window delta.
#    Hold t_start=0.9, t_end=0.3 fixed.
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


# ============================================================
# 3. t_start sweep:
#    Vary main chord/edit timestep.
#    Hold delta=0.15, t_end=0.3 fixed.
# ============================================================

run_job sweep_tstart_060 \
  --t-start 0.60 --t-end 0.3 --t-delta 0.15 --cleanup

run_job sweep_tstart_070 \
  --t-start 0.70 --t-end 0.3 --t-delta 0.15 --cleanup

run_job sweep_tstart_080 \
  --t-start 0.80 --t-end 0.3 --t-delta 0.15 --cleanup

run_job sweep_tstart_090 \
  --t-start 0.90 --t-end 0.3 --t-delta 0.15 --cleanup

run_job sweep_tstart_100 \
  --t-start 1.00 --t-end 0.3 --t-delta 0.15 --cleanup


# ============================================================
# 4. t_end / proximal refinement timestep sweep:
#    Vary refinement time.
#    Hold t_start=0.9, delta=0.15 fixed.
# ============================================================

run_job sweep_tend_010 \
  --t-start 0.9 --t-end 0.10 --t-delta 0.15 --cleanup

run_job sweep_tend_020 \
  --t-start 0.9 --t-end 0.20 --t-delta 0.15 --cleanup

run_job sweep_tend_030 \
  --t-start 0.9 --t-end 0.30 --t-delta 0.15 --cleanup

run_job sweep_tend_040 \
  --t-start 0.9 --t-end 0.40 --t-delta 0.15 --cleanup

run_job sweep_tend_050 \
  --t-start 0.9 --t-end 0.50 --t-delta 0.15 --cleanup


# ============================================================
# 5. Step scale / lambda sweep:
#    Vary edit update magnitude.
#    Hold t_start=0.9, delta=0.15, t_end=0.3 fixed.
# ============================================================

run_job sweep_lambda_060 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --step-scale 0.60 --cleanup

run_job sweep_lambda_080 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --step-scale 0.80 --cleanup

run_job sweep_lambda_100 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --step-scale 1.00 --cleanup

run_job sweep_lambda_120 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --step-scale 1.20 --cleanup

run_job sweep_lambda_140 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --step-scale 1.40 --cleanup


# ============================================================
# 6. Noise sample ablation:
#    Original paper studies whether multiple noise samples help.
#    Run both naive delta=0 and ChordEdit delta=0.15.
# ============================================================

run_job noise_naive_delta000_n1 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --noise-samples 1 --cleanup

run_job noise_naive_delta000_n2 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --noise-samples 2 --cleanup

run_job noise_naive_delta000_n3 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --noise-samples 3 --cleanup

run_job noise_naive_delta000_n4 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --noise-samples 4 --cleanup

run_job noise_ours_delta015_n1 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --noise-samples 1 --cleanup

run_job noise_ours_delta015_n2 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --noise-samples 2 --cleanup

run_job noise_ours_delta015_n3 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --noise-samples 3 --cleanup

run_job noise_ours_delta015_n4 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --noise-samples 4 --cleanup


# ============================================================
# 7. Integration step count ablation:
#    Original paper compares how performance changes as
#    n_steps increases, for naive delta=0 and ChordEdit delta=0.15.
# ============================================================

run_job steps_naive_delta000_s1 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --n-steps 1 --cleanup

run_job steps_naive_delta000_s2 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --n-steps 2 --cleanup

run_job steps_naive_delta000_s4 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --n-steps 4 --cleanup

run_job steps_naive_delta000_s6 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --n-steps 6 --cleanup

run_job steps_naive_delta000_s8 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.0 --n-steps 8 --cleanup

run_job steps_ours_delta015_s1 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --n-steps 1 --cleanup

run_job steps_ours_delta015_s2 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --n-steps 2 --cleanup

run_job steps_ours_delta015_s4 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --n-steps 4 --cleanup

run_job steps_ours_delta015_s6 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --n-steps 6 --cleanup

run_job steps_ours_delta015_s8 \
  --t-start 0.9 --t-end 0.3 --t-delta 0.15 --n-steps 8 --cleanup


echo
echo "============================================================"
echo "All original SD-Turbo ablation pilot runs finished."
echo "Outputs are under:"
echo "$EXPORT_ROOT/output"
echo "============================================================"