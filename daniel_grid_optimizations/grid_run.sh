# Test scripts for grid_generate.py and grid_eval.py.

cd /data/home/mirick/ChordEdit/daniel-grid-optimizations/daniel_grid_optimizations
conda activate chordedit

# Generate grids for the first 10 samples across 2 GPUs.
python grid_generate.py --data-root /shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_100 --model-root /shared/ssd_30T/mirick/models/sd-turbo --gpus 0 1 --max-samples 10

# Evaluate a shared generated dataset folder across 2 GPUs.
python grid_eval.py \
    --generated-root /shared/ssd_30T/mirick/generated/ultra_edit/UltraEdit_Region_10 \
    --include-psnr --include-lpips --include-clip \
    --gpus 0 1




CUDA_VISIBLE_DEVICES=5 python grid_generate.py \
    --data-root /shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_100 \
    --model-root /shared/ssd_30T/mirick/models/sd-turbo \
    --generated-root /shared/ssd_30T/mirick/generated_0p0 \
    --add-plots \
    --skip-embeddings \
    --diagonal-optimization


CUDA_VISIBLE_DEVICES=0 python grid_generate.py \
    --data-root /shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_100 \
    --model-root /shared/ssd_30T/mirick/models/sd-turbo \
    --generated-root /shared/ssd_30T/mirick/generated_0p15 \
    --add-plots \
    --skip-embeddings \
    --diagonal-optimization


# Regenerate scattered embeddings (no grids) on GPUs 5,6,7. Also backfills
# the per-token *_tokens.pt files for datasets generated before they existed.
# Add --cache-masks to also save per-sample mask.pt VAE latents.
python grid_generate.py \
    --data-root /shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_10000 \
    --model-root /shared/ssd_30T/mirick/models/sd-turbo \
    --embeddings-root /shared/ssd_30T/mirick/embeddings/sd_turbo \
    --skip-generated --gpus 5 6 7

python grid_generate.py \
    --data-root /shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Background_1000_v2 \
    --model-root /shared/ssd_30T/mirick/models/sd-turbo \
    --embeddings-root /shared/ssd_30T/mirick/embeddings/sd_turbo \
    --skip-generated --gpus 5 6 7