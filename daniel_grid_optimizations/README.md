# Daniel Grid Optimizations

## Environments

Generation and evaluation have different dependencies and should use separate conda environments:

| Script | Dependencies |
| --- | --- |
| `grid_generate.py` | `torch`, `diffusers`, ChordEdit pipeline |
| `grid_eval.py` | `torch`, `torchmetrics`, `transformers` (version-matched for `CLIPScore`) |

The evaluation metrics come from [PnPInversion](https://github.com/cure-lab/PnPInversion) and require compatible `torchmetrics`/`transformers` versions. The `chordedit` environment will fail on `CLIPScore` due to a version mismatch.

## Grid Generation

### Generation Arguments

| Flag | Meaning |
| --- | --- |
| `--data-root` | |
| `--model-root` | |
| `--embeddings-root` | |
| `--generated-root` | |
| `--gpus` | |
| `--max-samples` | |

| Flag | Meaning |
| --- | --- |
| `--add-plots` | |
| `--skip-embeddings` | |
| `--skip-generated` | |
| `--diagonal-optimization` | |

### Generation Usage

Runs are resumable. One process is spawned per GPU in `--gpus`.

```bash
python daniel_create/grid_generate.py --data-root ... --model-root ... --gpus 0
python daniel_create/grid_generate.py --data-root ... --generated-root ./results --gpus 0
python daniel_create/grid_generate.py --add-plots --gpus 0 1
python daniel_create/grid_generate.py --skip-embeddings --gpus 0
python daniel_create/grid_generate.py --skip-generated --gpus 0
python daniel_create/grid_generate.py --diagonal-optimization --gpus 0
```

## Grid Evaluation

### Evaluation Arguments

| Flag | Meaning |
| --- | --- |
| `--generated-root` | |
| `--result-path` | |
| `--gpus` | |
| `--max-samples` | |

### Evaluation Usage

```bash
python grid_eval.py --generated-root /shared/ssd_30T/mirick/generated/ultra_edit/UltraEdit_Region_10
```
