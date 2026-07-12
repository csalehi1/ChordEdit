# daniel_create

Generate ChordEdit `t_start` by `t_end` image grids over a PIE-Bench-style dataset.

## Output layout

```
<output-root>/<dataset-folder-name>[_n<max-samples>]/ # e.g. UltraEdit_Region_10000_n8
  id_to_inputs_<suffix>.csv
  <sample_id>/
    grid_clean.png                   # only with --grids
    cells/t_start_<..>__t_end_<..>.jpg
```

## Args

| Flag | Meaning |
|------|---------|
| `--data-root` | Dataset root (default UltraEdit_Region_1000) |
| `--output-root` | Parent output directory; results are saved under `<dataset-folder-name>[_n<max-samples>]` |
| `--model-root` | SD weights root |
| `--max-samples` | Cap samples for a smoke test; output folder appends `_n<max-samples>` |
| `--grids` | Write `grid_clean.png` overview images |
| `--diagonal-optimization` | Only generate cells where `t_start > t_end` (55 cells vs 121) |
| `--gpus 0 1 2 3` | GPU list; one shard process per GPU (default `0`) |

## Usage

Runs are resumable. One process is spawned per GPU in `--gpus`.

```bash
python daniel_create/grid_generate.py --data-root ... --model-root ... --gpus 0
python daniel_create/grid_generate.py --data-root ... --output-root ./results --gpus 0
python daniel_create/grid_generate.py --grids --gpus 0 1
python daniel_create/grid_generate.py --diagonal-optimization --gpus 0
```
