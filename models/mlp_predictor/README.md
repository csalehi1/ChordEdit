# Modified Classification (M + T)

Metric surrogate **M^** predicts `(psnr, clip)` from image/prompt embeddings and `(t_start, t_end)`. Timestep selector **T** uses M to pick the best grid cell.

**M^** bottlenecks image/mask/text embeddings, encodes timesteps with Fourier features, then predicts via separate towers: a standard MLP for PSNR (timesteps concatenated) and a FiLM-conditioned MLP for CLIP (timesteps as modulation). **T** has no trainable weights: it queries M over the discrete grid and selects the best cell.

## Setup

```bash
cd models/surrogate_selector
conda create -n surrogate-selector python=3.12 -y
conda activate surrogate-selector
pip install -r requirements.txt
```

Training and eval use precomputed embeddings. Paths and model choice come from `settings.json` / `settings.py`.

## Metric surrogate, M^

### 1. Train

```bash
python train_m.py
```

Writes a run to `runs/<dataset>/` (named by `RUN_NAME`, else a timestamp) with `regressor_weights.pt`, the `settings.json` it used, train/val/test splits, and `mean_surface.pt` (the train split's mean true delta surface, computed from the labels alone).

Targets are per-sample normalized deltas Delta. With `M_TARGET_SPACE="residual"` (recommended) the towers regress each image's deviation from the train mean surface and T adds the surface back at selection time, so an uninformative prediction falls back to the population-best cell instead of never deviating. `"delta"` regresses the full deltas with no offset.

### Timestep selector, T

> [!IMPORTANT]
> Only run after M^ training completes.

### 1. Evaluate selection metrics

```bash
python train_t.py
# optional: python train_t.py --run-dir runs/<dataset>/<run>
```

Reports regret, Spearman correlation, and deviate-gate precision/recall on the test split. Saves `selection_metrics.json` and `selections.json` for the test split, and `id_to_selections_<slug>.csv` with selected timesteps for every sample_id in the run (`sample_id`, `<scorer>_t_start`, `<scorer>_t_end`, e.g. `linex_a2_t_start`).

## Configuration

`settings.json` holds every tunable. `settings.py` reads it once at import and derives the fixed structure - paths, column names, the phi partials - from those values. To run a different configuration, edit `settings.json` or pass `--settings-path` pointing at another copy:

```bash
CUDA_VISIBLE_DEVICES=4 python train_m.py --settings-path /tmp/my_config.json
```

Every run saves the config it used to `<run_dir>/settings.json`, and `train_t.py` (and the analysis scripts) replay a run from that snapshot when given `--run-dir` - so evaluation always sees the training config, whatever `--settings-path` says. A key missing from the file is an error, not a silent default.

`settings.py` binds its constants at import time, before any script's argparse runs, so it reads `--settings-path` straight off `sys.argv`; the entry points declare the flag as well so it appears in `--help`.
