# Qwen-Image-Bench evaluation runners

Runs the existing `QwenFullMetricsJudge` / `QwenSelectionJudge` classes
(`evaluation/full_metrics_judge.py`, `evaluation/selection_judge.py`) with
`model_id="Qwen/Qwen-Image-Bench"` (27B) instead of the default
`Qwen/Qwen3-VL-8B-Instruct` (8B), over the UltraEdit 100-sample data-root.

## Requirements

- Access to `/shared/ssd_30T/zarageddes/ultraedit_100_v2_dataroot` (the
  consolidated 100-sample data-root: mapping file + source images + 121-cell
  SD-Turbo grids per sample). Built/tested on the `seribizon` research box.
- At least 2 GPUs with ~49GB+ each. The model is ~54GB in bf16 and does not
  fit on a single GPU at that size, so each "replica" needs `device_map="auto"`
  splitting it across (at least) 2 GPUs -- this is naive pipeline sharding,
  not true tensor parallelism (this model architecture has no `_tp_plan` in
  `transformers` yet).

## Getting real throughput

Since each replica needs 2+ GPUs just to fit the model, use multiple
replicas in parallel for actual throughput -- one process per replica, each
pinned to its own GPU pair via `CUDA_VISIBLE_DEVICES`, each handling a
different `--shard` of the sample list:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 run_full_metrics_judge.py --shard 0 --nshards 2 --out shard0.csv &
CUDA_VISIBLE_DEVICES=2,3 python3 run_full_metrics_judge.py --shard 1 --nshards 2 --out shard1.csv &
wait
python3 merge_shards.py shard0.csv shard1.csv --out full_metrics_qwen_image_bench.csv
```

Same pattern for `run_selection_judge_tournament.py`.

Use `--max-samples N` to limit to the first N samples (sorted by ID) for a
quick test before committing to the full 100.

## Before running at scale

Run `sanity_check.py` first (needs 2 GPUs) -- it loads the model once and
runs a single cell through both judges with `enable_thinking` True and
False, printing raw output, generated-token counts, and whether the
existing JSON-extraction logic parses each. Useful for confirming the model
still produces valid, sensible output before committing GPU-hours to a full
run, and for sanity-checking real per-call timing.

## Notable findings from initial runs (as of this writing)

- `enable_thinking=True` (the model's own documented default) produces
  meaningfully different scores from `enable_thinking=False` on at least
  some factors (not just noise) -- e.g. one comparison showed
  `spatial_relationship` scored 7 vs 1, `alignment` scored 7 vs 4, on the
  identical image pair. Turning thinking off is a real speed/behavior
  tradeoff, not a free win.
- Thinking mode generates far more tokens (~2,700-4,700 vs. ~40-600 with it
  off), which is the dominant cost driver given generation is autoregressive
  and this model needs 2-GPU pipeline sharding (each decode step pays
  cross-GPU communication latency).
- Batch size is tuned empirically from observed GPU memory headroom (see
  comment in `run_full_metrics_judge.py`), not calculated from a formula --
  worth re-checking if the data-root or hardware changes.
