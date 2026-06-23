# Classification Model

Predicts timestep parameters (`t_start`, `t_end`), equivalent to $(t^*, t^{**})$, from a `(source_prompt, target_prompt)` string pair.

## Architecture

Strings `source_prompt` and `target_prompt` are fed into `SiameseEncoder` which outputs the concatenated embeded vector $\langle A \mid B \mid A - B \mid A \odot B \rangle$ where $\odot$ is the Hadamard product, element-wise multuplication. This output vector has size $4 \times 384 = 1536$.

The $1536$-dimensional vector is passed through an MLP body of `Linear` with $1536 \rightarrow 512$, `LayerNorm`, `ReLU`, `Dropout` with $0.1$, `Linear` with $512 \rightarrow 256$, `ReLU`, `Dropout` with $0.1$, and `Linear` with $256 \rightarrow 128$. The $128$-dimensional output is then routed to two parallel heads with `head1` for `t_start` and `head2` for `t_end`. The head type is configurable: `CORAL` for ordinal threshold classification or `MSE` for scalar regression. Each head's output is decoded to a bucket index in $\{0, \dots, k_i-1\}$ where $k_i$ is the number of distinct bins for $t_i$, which maps to a float value in $[0.0, 1.0]$.

*See [DESIGN.md](DESIGN.md) for full details.*

## Setup

### Create a Conda environment

```bash
conda create -n chordedit python=3.12
conda activate chordedit
pip install -r requirements.txt
```



## Workflow

### 1. Configure `settings.py`

Before running anything, open [settings.py](settings.py) and update these settings to match your data:

| Setting | What to set |
|---|---|
| `METRICS_CSV` | Path to your `id_to_metrics_*.csv` file (e.g. `DATA_DIR / "id_to_metrics_sdturbo.csv"`). `OUTPUTS_DIR` is derived from this name automatically. |
| `N_BUCKETS_START` | Number of distinct `t_start` ($t^{*}$) values in your CSV. The loader raises at runtime if the data doesn't match. |
| `N_BUCKETS_END` | Number of distinct `t_end` ($t^{**}$) values in your CSV. Same enforcement applies. |
| `DELTA_VALUE` | `t_delta` value used to select rows from the data. Reflects $\delta$ values used to generate image. |

Wrong values here will cause a load-time error, so set them before anything else.

### 2. Study data in `eval_data.ipynb`

Open [eval_data.ipynb](eval_data.ipynb) to study input data before training.

### 3. Select a computed metric

The training target is derived from the raw PSNR and CLIP columns. To change which metric is used, set `_ACTIVE` in [settings.py](settings.py) to one of the keys in `_COMPUTED_METRIC_OPTIONS`:

```python
_ACTIVE = _COMPUTED_METRIC_OPTIONS["naive_pareto_score"]
```

| Key | Description |
|---|---|
| `"naive_pareto_score"` | Pareto improvement relative to the paper's baseline row per sample group. |
| `"agreement_score"` | Similarity between PSNR and CLIP similarity scores. |
| `"weighted_combined_score"` | Weighted blend of normalized PSNR and CLIP, adjust `_LAMBDA_PSNR` and `_LAMBDA_CLIP`. |

Each option is a `MetricOption(col, fn, label)`. The three module-level constants `COMPUTED_METRIC_COL`, `COMPUTED_METRIC_FN`, and `COMPUTED_METRIC_LABEL` are derived from `_ACTIVE` automatically and do not need to be edited directly.

### 4. Adjust remaining settings

If wanted, further edit [settings.py](settings.py) to adjust training behavior before running the model.

| Setting | Description |
|---|---|
| `TARGET_COLUMN` | Which metric column to train on; defaults to `COMPUTED_METRIC_COL`. |
| `ENCODER_MODEL` | Pretrained sentence-transformer checkpoint for the Siamese encoder. |
| `FREEZE_ENCODER` | If `True`, encoder weights are frozen during training. |
| `HEAD_TYPE` | `"CORAL"` (ordinal, default) or `"MSE"` (regression). |
| `USE_CLASS_WEIGHTS` | Weight loss by inverse class frequency to counteract label imbalance. |
| `SEED` | Global random seed. |
| `EPOCHS`, `BATCH_SIZE` | Training loop hyperparameters. |
| `ENCODER_LR`, `WEIGHT_DECAY` | Optimizer settings for the encoder. |
| `MLP_LR` | Learning rate for the MLP body and heads. |
| `MLP_WIDE`, `MLP_HIDDEN`, `MLP_INNER` | Hidden layer widths of the MLP body. |
| `MLP_DROPOUT` | Dropout rate applied inside the MLP body. |

### 5. Train the model

From the repo root:

```bash
python -m models.classification.classify
```

This loads the CSVs, (re)computes the computed metric column, trains the model, and saves checkpoints and metric plots to `OUTPUTS_DIR`.

### 6. Evaluate in `eval_model.ipynb`

Open [eval_model.ipynb](eval_model.ipynb) to inspect model performance after training. The notebook loads the most recent saved checkpoint and walks through:

- Per-split metrics
- Confusion matrices
- Error distributions and scatter plots
- Per-bucket accuracy breakdowns

## Files

| File | Purpose |
|---|---|
| `settings.py` | All configuration |
| `model.py` | `OrdinalPairClassifier` and `SiameseEncoder` definitions |
| `head_coral.py` | CORAL ordinal head |
| `head_mse.py` | MSE regression head |
| `head_mae.py` | MAE bucket utilities |
| `utils.py` | Scoring functions and data helpers |
| `classify.py` | Training entry point |
| `eval_model.ipynb` | Post-training evaluation notebook |
| `eval_data.ipynb` | Dataset exploration notebook |
| `DESIGN.md` | Detailed architecture and design notes |
