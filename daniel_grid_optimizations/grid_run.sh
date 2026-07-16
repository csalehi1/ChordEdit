# Test scripts for grid_generate.py and grid_eval.py.

# Generate grids for the first 10 samples across 2 GPUs.
python grid_generate.py --data-root /shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_100 --model-root /shared/ssd_30T/mirick/models/sd-turbo --gpus 0 1 --max-samples 10

# Evaluate the grids for the first 10 samples across 2 GPUs.
python grid_eval.py --generated-root /data/home/mirick/ChordEdit/daniel-grid-optimizations/daniel_grid_optimizations/generated/UltraEdit_Region_100_n10 --include-psnr --include-lpips --include-clip --gpus 0 1