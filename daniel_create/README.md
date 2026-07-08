# daniel_create

Tools to generate and label ChordEdit `t_start` by `t_end` image grids over a PIE-Bench style dataset. Generation and labeling are split into two independent stages that share cell images on disk.

## Output layout

```
<output-root>/
  id_to_inputs_<suffix>.csv
  id_to_metrics_<suffix>.csv
  <sample_id>/
    grid_clean.png             # only with --overview-grids
    grid_psnr.png  grid_clip.png  # only with --overview-grids
    cells/t_start_<..>__t_end_<..>.jpg
```

`<suffix>` is the output directory stem in lowercase with underscores removed
(e.g. `UltraEdit_Region_1000` → `ultraeditregion1000`).

`id_to_inputs_*.csv` columns: `sample_id,source_prompt,target_prompt,image_path,mask_image_path`

`id_to_metrics_*.csv` columns: `sample_id,t_start,t_end,t_delta,whole_psnr,clip_edited,cell_path`
(`cell_path` is relative to the CSV, e.g. `00000000/cells/t_start_0p9__t_end_0p3.jpg`)

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

# Also write overview images (grid_clean / grid_psnr / grid_clip).
OVERVIEW_GRIDS=1 GPUS="0 1 2 3" bash daniel_create/script_both.sh

# Run scripts on more GPUs.
GPUS="0 1 2 3 4 5 6 7" bash daniel_create/script_both.sh
# Smoke test with 10 image samples on one GPU.
MAX_SAMPLES=10 GPUS="0" bash daniel_create/script_generate.sh
```

The Python entry points can also be called directly (see `--help`):

```bash
python daniel_create/generate_grid.py --data-root ... --output-root ... --device cuda:0
python daniel_create/label_grid.py  --data-root ... --output-root ... --device cuda:0
# Optional overview images:
python daniel_create/generate_grid.py ... --overview-grids
python daniel_create/label_grid.py  ... --overview-grids
```

## Datasets

Which `mapping_file.json` keys hold each field is configured by the `FIELD_*`
settings in `settings.py`. Defaults target UltraEdit (`original_prompt` for the
source prompt, `editing_prompt` for the target prompt); override them to point at
another dataset's naming.

## Metrics

`label_grid.py` inlines two PnPInversion metrics so everything runs in the
`chordedit` env:

- `whole_psnr` — whole-image PSNR (source vs. edited, `data_range=1.0`).
- `clip_edited` — CLIP similarity (`100 × cosine`) of the masked edit region to
  the target prompt.
