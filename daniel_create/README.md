# daniel_create

Generate and label ChordEdit `t_start` × `t_end` grids over a PIE-Bench-style dataset.
Two stages share cell images on disk.

## Output layout

```
daniel_create/generated/<Path(data_root).name>/   # always (next to generate_grid.py)
  id_to_inputs_<suffix>.csv
  id_to_metrics_<suffix>.csv
  <sample_id>/
    grid_clean.png                   # only with --grids
    grid_psnr.png  grid_clip.png     # only with --grids
    cells/t_start_<..>__t_end_<..>.jpg
```

`<suffix>` = output folder stem, lowercase, underscores removed
(`UltraEdit_Region_1000` → `ultraeditregion1000`).

- `id_to_inputs_*.csv`: `sample_id,source_prompt,target_prompt,image_path,mask_image_path`
- `id_to_metrics_*.csv`: `sample_id,t_start,t_end,t_delta,whole_psnr,clip_edited,cell_path`
  (`cell_path` relative to the CSV, e.g. `00000000/cells/t_start_0p9__t_end_0p3.jpg`)

## Args

Only these flags are supported:

| Flag | Meaning |
|------|---------|
| `--data-root` | Dataset root (default UltraEdit_Region_1000) |
| `--model-root` | SD weights root (generate only) |
| `--max-samples` | Cap samples for a smoke test |
| `--grids` | Write overview images |
| `--gpus 0 1 2 3` | GPU list; one shard process per GPU (default `0`) |

## Usage

Runs are resumable. One process is spawned per GPU in `--gpus`.

```bash
bash daniel_create/script_both.sh --gpus 0 1 2 3
bash daniel_create/script_generate.sh --gpus 0 1 2 3
bash daniel_create/script_label.sh --gpus 0 1 2 3

# Overview images + small smoke test
bash daniel_create/script_both.sh --max-samples 10 --grids --gpus 0
```

```bash
python daniel_create/generate_grid.py --data-root ... --model-root ... --gpus 0
python daniel_create/label_grid.py  --data-root ... --gpus 0
python daniel_create/generate_grid.py --grids --gpus 0 1
```

## Datasets / metrics

`FIELD_*` in `settings.py` map `mapping_file.json` keys (UltraEdit defaults:
`original_prompt` / `editing_prompt`).

- `whole_psnr` — whole-image PSNR (source vs edited, `data_range=1.0`)
- `clip_edited` — CLIP similarity (`100 × cosine`) of the masked edit region to the target prompt
