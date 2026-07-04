# daniel_create

Self-contained tools to generate and label ChordEdit `t_start` by `t_end` image grids over a PIE-Bench style dataset. Generation and labeling are split into two independent stages that share cell images on disk.

## Layout

| File | Purpose |
| --- | --- |
| `settings.py` | All config: paths, grid values, edit configs, CLIP model, plot constants. |
| `common.py` | Shared helpers: paths, dataset/mask loading, pipeline loader. |
| `pipeline_ops.py` | Factorized grid generation (optimized `u_estimate` + batched cleanup/decode). |
| `grid_render.py` | Builds `grid_clean` / `grid_psnr` / `grid_clip` composites. |
| `generate_grid.py` | Entry point: generate the entire image set. |
| `label_grid.py` | Entry point: score + label an entire generated image set. |
| `script_generate.sh` | Generate the entire image set (multi-GPU sharded). |
| `script_label.sh` | Label a generated image set with PSNR + CLIP (sharded, merges CSVs). |
| `script_both.sh` | Run both stages end to end. |
| `DESIGN.md` | Optimizations behind generation and labeling. |

## Output layout

```
<output-root>/
  id_to_prompts.csv                   # sample_id -> source image path + prompts
  result.csv                          # merged PSNR + CLIP scores
  <sample_id>/
    grid_clean.png                    # labeled images
    grid_psnr.png  grid_clip.png      # metric overlays
    cells/t_start_<..>__t_end_<..>.jpg
```

## Usage

Both stages skip work that is already done, so runs are resumable and shardable.
`GPUS` is required and must be set explicitly; when unset the scripts fall back to
the minimum requirement of a single GPU (index 0).

```bash
# Generate every grid, then score them.
GPUS="0 1 2 3" bash daniel_create/script_both.sh

# Alternatively, run the stages separately.
GPUS="0 1 2 3" bash daniel_create/script_generate.sh
GPUS="0 1 2 3" bash daniel_create/script_label.sh

# Run scripts on more GPUs.
GPUS="0 1 2 3 4 5 6 7" bash daniel_create/script_both.sh
# Smoke test with 10 image samples on one GPU.
MAX_SAMPLES=10 GPUS="0" bash daniel_create/script_generate.sh
```

The Python entry points can also be called directly (see `--help`):

```bash
python daniel_create/generate_grid.py --data-root ... --output-root ... --device cuda:0
python daniel_create/label_grid.py  --data-root ... --output-root ... --device cuda:0
```

## Datasets

Which `mapping_file.json` keys hold each field is configured by the `FIELD_*`
settings in `settings.py`. Defaults target UltraEdit (`original_prompt` for the
source prompt, `editing_prompt` for the target prompt); override them to point at
another dataset's naming.

## Metrics

`label_grid.py` inlines two PnPInversion metrics so everything runs in the
`chordedit` env:

- `psnr` — whole-image PSNR (source vs. edited, `data_range=1.0`).
- `clip_similarity_target_image_edit_part` — CLIP similarity (`100 × cosine`) of
  the masked edit region to the target prompt.