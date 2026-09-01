# Modified Classification (M + T)

Metric surrogate **M^** predicts `(psnr, clip)` from image/prompt embeddings and `(t_start, t_end)`. Timestep selector **T** uses M to pick the best grid cell.

**M^** featurizes both modalities with the same modules `attention_predictor`
uses - `VisionFeaturizer` turns the `(C, S, S)` VAE latent into `(N_v, d_v)`
patch tokens, `TextFeaturizer` turns the pooled prompt pair into `(2, d_t)`
tokens. `VisualProjector` mean-pools the visual tokens to `(1, d_v)`; that and
the two prompt tokens flatten and concatenate into one `(d_v + 2 * d_t)` vector
feeding two towers, one per target metric, each predicting the whole
`(n_cells, 2)` grid in one shot. No timestep is an input. **T** has no
trainable weights: it reads M's grid and selects the best cell.

> [!NOTE]
> `VisionProjector` is a bare `nn.Linear`, and averaging commutes with it, so
> `mean(P_v(tokens)) == P_v(mean(tokens))`: the pooled visual feature is a
> linear map of the *mean patch*, and only `4 * PATCH_SIZE ** 2` numbers per
> image survive the pool. `PATCH_SIZE=1` leaves 4 (the per-channel spatial
> mean); `PATCH_SIZE=64` leaves the whole 16384-d latent. Raise `PATCH_SIZE`,
> or put a nonlinearity between projection and pooling, if the visual path
> should see more than that.

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
python train.py
```

Writes a run to `runs/<dataset>/` (named by `RUN_NAME`, else a timestamp) with `regressor_weights.pt`, the `settings.json` it used, train/val/test splits, and `mean_surface.pt` (the train split's mean true delta surface, computed from the labels alone).

Targets are per-sample normalized deltas Delta. With `PREDICTION_SPACE="residuals"` (recommended) the towers regress each image's deviation from the train mean surface and T adds the surface back at selection time, so an uninformative prediction falls back to the population-best cell instead of never deviating. `"deltas"` regresses the full deltas with no offset.

### Timestep selector, T

> [!IMPORTANT]
> Only run after M^ training completes.

### 1. Evaluate selection metrics

```bash
python selector.py
# optional: python selector.py --run-dir runs/<dataset>/<run>
```

Reports regret, rank correlation, gain, and Top-K accuracy on the test split. Saves `selection_metrics.json` and `selections.json` for the test split, and `id_to_selections_<slug>.csv` with selected timesteps for every sample_id in the run (`sample_id`, `<scorer>_t_start`, `<scorer>_t_end`, e.g. `linex_a2_t_start`).

## Configuration

`settings.json` holds every tunable. `settings.py` reads it once at import and derives the fixed structure - paths, column names, the phi partials - from those values. To run a different configuration, edit `settings.json` or pass `--settings-path` pointing at another copy:

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --settings-path /tmp/my_config.json
```

Every run saves the config it used to `<run_dir>/settings.json`, and `selector.py` (and the analysis scripts) replay a run from that snapshot when given `--run-dir` - so evaluation always sees the training config, whatever `--settings-path` says. A key missing from the file is an error, not a silent default.

`settings.py` binds its constants at import time, before any script's argparse runs, so it reads `--settings-path` straight off `sys.argv`; the entry points declare the flag as well so it appears in `--help`.

## Embeddings

`embeddings.py` reads the same per-sample files `attention_predictor` does:
`image_tokens.pt` (the `(C, S, S)` VAE latent) and the pipeline's pooled
`source.pt` / `target.pt`, packed to `(n, C, S, S)` and `(n, 1, D)` tables under
`.cache/packed_embeddings/` with layout `img_tokens_src_tar_pooled_v2`. Both
tables reach the featurizers in that shape, unflattened.

`TEXT_EMB_TYPE="clip"` still swaps in CLIP-L/14 prompt vectors, reshaped to
the same `(n, 1, D)` layout. `IMG_EMB_TYPE` must be `"vae"`: `"clip"` and
`"vae_clip"` carry no latent grid for `VisionFeaturizer` to patch, so
`_apply_img_emb_source` rejects them instead of failing deeper in the model.

The values are identical to the older `image.pt` / `source.pt` layout -
`image.pt` is exactly `image_tokens.pt` flattened - but the layout bump
invalidates any `img_src_tar_v1` cache, so the first run after this change
repacks from the scattered files.

## Metrics

Every metric reported by `train.py` and `selector.py` is computed by
`metrics.py`, which takes score surfaces of shape `(S, N)` -- true surface
first, predicted second, without exception -- and imports no `settings`,
`model`, or `_data`. Both entry points call its aggregators (`training_metrics`,
`selection_metrics`, `per_component_metrics`) rather than assembling their own
dicts, so the two key sets cannot drift apart. Only `loss` is assembled by
`train.py`. `attention_predictor` shares this module verbatim, so runs from
either model are directly comparable.

## Tracking (wandb)

`train.py` streams the training curve to wandb.ai via `_wandb.py`. The knobs are
globals in `_wandb.py`, not settings keys -- they steer telemetry, not the
model, so they stay out of the per-run `settings.json` that `selector.py`
replays:

```python
USE_WANDB = True
WANDB_ENTITY = "dfmirick-harvard-university"
WANDB_PROJECT = "mlp-predictor"
WANDB_MODE = "online"     # "online", "offline", or "disabled"
WANDB_GROUP = ""
```

The API key lives in `.env` beside `train.py`, loaded by `python-dotenv`. That
file is gitignored and must never be committed.

Logged per epoch: every `eval_regression` and `eval_selection` key under
`train/` and `val/` prefixes (long column names shortened to `psnr` / `clip`),
plus `lr` and `epoch_seconds`. Test metrics and the best epoch's validation
metrics go to `run.summary`, so the runs table ranks on final quality rather
than the last epoch.

Tracking never breaks training. If wandb is missing, or `WANDB_API_KEY` is
absent in online mode, or `wandb.init` fails, `train.py` prints a warning and
continues untracked. `selector.py` does not report to wandb.
