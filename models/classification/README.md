# Classification Model

Predicts timestep parameters (`t_start`, `t_end`), equivalent to $(t^*, t^{**})$, from precomputed ChordEdit embeddings of the source image, edit mask, and `(source_prompt, target_prompt)` string pair.

## Architecture

The classifier consumes four precomputed embeddings per sample: the flattened VAE latents of the source image and edit mask, and the pooled text embeddings of the source and target prompts, produced by the same frozen ChordEdit encoders that generated the metric labels.

The image and mask latents each pass through a projection (`Linear` $\rightarrow$ `LayerNorm` $\rightarrow$ `ReLU`, dimension `IMG_PROJ_DIM = 512`; or a small conv stack when `IMG_ENCODER = "conv"`). The text pair is combined as $\langle A \mid B \mid A - B \mid A \odot B \rangle$ where $\odot$ is the Hadamard product, then projected to `TEXT_PROJ_DIM = 256`. The concatenated context vector ($512 \times 2 + 256 = 1280$) is passed through an MLP body of `Linear` with $1280 \rightarrow 256$, `LayerNorm`, `ReLU`, `Dropout` with $0.2$, `Linear` with $256 \rightarrow 128$, `ReLU`, `Dropout` with $0.2$, and `Linear` with $128 \rightarrow 64$. The $64$-dimensional output is then routed to two parallel heads with `head1` for `t_start` and `head2` for `t_end`.

By default (`HEAD_TYPE = "CE"`), each head is a `ClassificationHead` that outputs $K$ logits per target; training uses standard cross-entropy with optional label smoothing, and inference picks the argmax bucket index. Alternative head types are available: `CORAL` for ordinal threshold classification and `MSE` for scalar regression snapped to the nearest bucket. Each decoded index lies in $\{0, \dots, k_i-1\}$ where $k_i$ is the number of distinct bins for $t_i$, and maps to a float value in $[0.0, 1.0]$.

*See [DESIGN.md](DESIGN.md) for full details.*

## Setup

### Create a Conda environment

```bash
conda create -n chordedit python=3.12
conda activate chordedit
pip install -r requirements.txt
```

## Workflow

### 1. Configure `settings.json`

Every tunable lives in [settings.json](settings.json); [settings.py](settings.py) reads that file and derives the rest (paths, column names, the score partials). To run a different configuration, edit `settings.json` or point `CE_SETTINGS_JSON` at another copy of it:

```bash
CE_SETTINGS_JSON=/tmp/my_config.json python train.py
```

Every key is required — a missing key raises `KeyError` at import. Keys starting with `_` are treated as comments and ignored. The most important data keys:

| Key | What to set |
|---|---|
| `DIR_NAME` | Dataset folder name under the generated/datasets/embeddings roots (also names `RUNS_DIR`). |
| `CHORD_EDIT_MODEL` | `"sd_turbo"`, `"sdxl_turbo"`, or `"flux"`. Selects the encoder stack, image size, and which embedding caches apply. |
| `TARGET_T_DELTA` | `t_delta` value used to select rows from the data. Reflects $\delta$ values used to generate images. `null` uses every `t_delta`. |
| `CELL_SUBSET` | Candidate grid cells: `"all"` (all 121 positions) or `"lower"` (the 55 with `t_end < t_start`, the only region labeled before the annotation pass). Changes the labels and phi's scale, not just the argmax domain. |
| `TRAIN_FRAC` / `VAL_FRAC` / `TEST_FRAC` | Sample-level split ratios (split is by `sample_id`, seeded by `SPLIT_SEED`). |

`GRID_T_START` / `GRID_T_END` (the fixed bucket grids) live in `settings.py` as code constants. Missing cells in the CSV are fine (those classes simply get no labels); off-grid values raise at load time.

### 2. Study data in `eval_data.ipynb`

Open [eval_data.ipynb](eval_data.ipynb) to study input data before training.

### 3. Select a computed metric

The training target is a scalar score over **per-sample normalized deltas**, not raw metrics. For each `sample_id`, every metric in `C_TARGET_COLS` (default: PSNR and CLIP) is min-max scaled across that sample's candidate $(t^*, t^{**})$ cells, then the baseline cell at `(DEFAULT_T_START, DEFAULT_T_END)` is subtracted:

$$\Delta_i = \frac{s_i - s_i^0}{\max_T s_i - \min_T s_i} \in [-1, 1]$$

so $\Delta_i > 0$ is an improvement over the baseline edit and the baseline row itself is $\Delta = 0$. `scores.calc_normalized_deltas` builds $\Delta$; `scores.score_df` packs a metrics DataFrame into a $(B, N, C)$ tensor, applies a score in one batched call, and returns a `pd.Series` aligned to `df.index`.

The active score is selected in `settings.json`:

```json
"C_TARGET_FN": "linex",
"C_TARGET_ALPHA": 2.0,
```

| Function | $\varphi(\Delta)$ | Behaviour |
|---|---|---|
| `"naive"` | $\sum_i w_i \Delta_i$ | Linear and interpretable, but indifferent to balance: it cannot tell a candidate that improves both metrics moderately from one that maximizes one and tanks the other. |
| `"cara"` | $\frac{1}{\alpha}\sum_i w_i\left(1 - e^{-\alpha \Delta_i}\right)$ | Strictly concave (exponential/CARA utility): regressions are penalized exponentially, so a severe regression cannot be offset elsewhere. Gains saturate at $w_i/\alpha$. |
| `"linex"` **(default)** | $\frac{1}{2}\sum_i w_i\left[\Delta_i + \frac{1 - e^{-\alpha \Delta_i}}{\alpha}\right]$ | Mean of the two. Keeps CARA's superlinear regression penalty without the reward cap, so it still separates strong improvements. Curvature is half of CARA's at matched $\alpha$. |

All three take `deltas` of shape $(\dots, N, C)$ plus optional `weights` of shape $(C,)$, reduce the trailing metric axis, recover `naive_score` as $\alpha \to 0^+$, and map $\Delta = 0 \mapsto 0$ — so `idxmax` per `sample_id` returns the best improving cell, or the baseline itself when nothing improves on it.

`C_TARGET_COL` is derived as `f"{C_TARGET_FN}_score"`. Changing `C_TARGET_FN` or `C_TARGET_ALPHA` changes the labels themselves, so runs either side of a change are not comparable.

### 4. Adjust remaining settings

If wanted, further edit [settings.json](settings.json) to adjust the model and training behavior before running.

| Key | Description |
|---|---|
| `C_TARGET_FN` / `C_TARGET_ALPHA` | Active score function and its risk-aversion $\alpha$ (larger values penalize regressions harder). |
| `DEFAULT_T_START` / `DEFAULT_T_END` | Baseline $(t^*, t^{**})$ cell that deltas are measured against. Exactly one row per `sample_id` must match, or scoring raises. |
| `IMG_ENCODER` | `"linear"` (one Linear over the flat VAE latent) or `"conv"` (fold back to $(C, S, S)$ and downsample). |
| `IMG_PROJ_DIM` / `TEXT_PROJ_DIM` | Widths of the image/mask and text projections. |
| `HEAD_TYPE` | `"CE"` (multiclass, default), `"CORAL"` (ordinal), or `"MSE"` (regression). |
| `CE_LOSS_TYPE` | `"one_hot_ce_loss"` or `"cost_sensitive_ce_loss"` (only when `HEAD_TYPE = "CE"`). |
| `USE_CLASS_WEIGHTS` | Weight loss by inverse class frequency to counteract label imbalance. |
| `LABEL_SMOOTHING` | Label smoothing for CE training (default `0.15`; only used when `HEAD_TYPE = "CE"`). |
| `SEED` | Seeds model init and batch order. |
| `SPLIT_SEED` | Seeds the sample-level split. Held apart from `SEED` so repeats over `SEED` measure init noise on one fixed test set. |
| `EPOCHS`, `BATCH_SIZE` | Training loop hyperparameters. |
| `LR`, `WEIGHT_DECAY` | AdamW optimizer settings. |
| `LR_SCHEDULER` | Per-epoch learning-rate decay: `"none"`, `"cosine"`, `"linear"`, `"step"`, or `"plateau"`. |
| `LR_MIN_FACTOR` | Floor of the decay as a fraction of `LR` (ignored by `"step"`). |
| `CKPT_METRIC` | Validation metric that selects the checkpoint: `"bal_acc_t_start"`, `"bal_acc_t_end"`, `"acc_both"`, `"loss"`, `"regret_median"`, or `"top1_hit_rate"`. |
| `MLP_WIDE`, `MLP_HIDDEN`, `MLP_INNER` | Hidden layer widths of the MLP body. |
| `MLP_DROPOUT` | Dropout rate applied inside the MLP body. |
| `RUN_NAME` | Run directory name under `RUNS_DIR`; empty string means use a timestamp. |

### 5. Train the model

From this directory:

```bash
python train.py
```

This loads the CSVs, computes the score column if the metrics CSV does not already carry it, selects the best cell per sample, materializes the embeddings, trains the model, and saves to `RUNS_DIR/<run>/`:

| File | Role |
|---|---|
| `settings.json` | Exact config used for the run |
| `classifier_weights.pt` | Best checkpoint |
| `id_to_split.csv` | Sample-to-split mapping |
| `train_metrics.json` | Best-epoch + val/test metrics |
| `id_to_preds.csv` | Per-sample `(pred_t_start, pred_t_end)` picks |
| `train.log` | Full training stdout/stderr |

Embeddings are served by [embeddings.py](embeddings.py) from a packed table under `.cache/packed_embeddings/` when one matches the current settings, else packed on the fly from the scattered per-sample files under `EMBEDDINGS_DIR`. If neither cache can cover the requested samples, `get_embeddings` raises with instructions.

### 6. Evaluate in `eval_model.ipynb`

Open [eval_model.ipynb](eval_model.ipynb) to inspect model performance after training. The notebook replays the run's saved settings via `_helpers.load_run_settings` (so evaluation sees exactly the training config), reloads the splits from `id_to_split.csv`, and walks through:

- Per-split metrics
- Confusion matrices
- Error distributions and scatter plots
- Per-bucket accuracy breakdowns

## Files

| File | Purpose |
|---|---|
| `settings.json` | All tunables (required keys; `_`-prefixed keys are comments) |
| `settings.py` | Reads `settings.json` and derives paths, column names, and score partials |
| `_helpers.py` | Run-settings snapshot/replay, run-dir and device resolution, inputs loader |
| `_data.py` | Data loading, per-sample label selection, splits, and training tensors |
| `embeddings.py` | Packed/scattered embedding caches |
| `model.py` | `OrdinalPairClassifier`, embedding projections, and shared decode helpers |
| `head_ce.py` | CE multiclass head (default) |
| `head_coral.py` | CORAL ordinal head |
| `head_mse.py` | MSE regression head |
| `selection.py` | Selection (regret) metrics: judges the picked grid cell against the sample's whole grid |
| `scores.py` | Torch score functions (`naive_score`, `cara_score`, `linex_score`), delta helpers (`calc_normalized`, `calc_deltas`, `calc_normalized_deltas`), and the `score_df` DataFrame adapter |
| `train.py` | Training entry point |
| `eval_model.ipynb` | Post-training evaluation notebook |
| `eval_data.ipynb` | Dataset exploration notebook |
| `DESIGN.md` | Detailed architecture and design notes |
