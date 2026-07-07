CUDA_VISIBLE_DEVICES=0 python run_pie_bench.py \
    --model-root checkpoints/FLUX.1-schnell \
    --pie-root datasets/PIE-Bench_v1/ \
    --model-type flux \
    --image-size 1024 \
    --t-start 0.96 \
    --t-end 0.5 \
    --t-delta 0.02 \