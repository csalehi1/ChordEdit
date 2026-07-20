# Classification Model

Predicts timestep parameters (`t_start`, `t_end`), equivalent to $(t^*, t^{**})$, from a `(source_prompt, target_prompt)` string pair.

## Architecture

Strings `source_prompt` and `target_prompt` are fed into `SiameseEncoder` which outputs the concatenated embedded vector $\langle A \mid B \mid A - B \mid A \odot B \rangle$ where $\odot$ is the Hadamard product, element-wise multiplication. This output vector has size $4 \times 384 = 1536$.

The $1536$-dimensional vector is passed through an MLP body of `Linear` with $1536 \rightarrow 256$, `LayerNorm`, `ReLU`, `Dropout` with $0.2$, `Linear` with $256 \rightarrow 128`, `ReLU`, `Dropout` with $0.2$, and `Linear` with $128 \rightarrow 64$. The $64$-dimensional output is then routed to two parallel heads with `head1` for `t_start` and `head2` for `t_end`.

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

### 1. Configure `settings.py`

Before running anything, open [settings.py](settings.py) and update these settings to match your data:

| Setting | What to set |
|---|---|
| `DIR_NAME` | Dataset folder name under the generated/datasets roots (also names `OUTPUTS_DIR`). |
| `GENERATED_DIR` / `INPUTS_CSV` / `METRICS_CSV` | Paths to your `id_to_inputs_*.csv` and `id_to_metrics_*.csv` files. |
| `GRID_T_START` / `GRID_T_END` | Fixed `t_start` / `t_end` bucket grids used by the model. Missing cells in the CSV are fine (those classes simply get no labels); off-grid values raise at load time. |
| `TARGET_T_DELTA` | `t_delta` value used to select rows from the data. Reflects $\delta$ values used to generate images. |

Wrong values here will cause a load-time error, so set them before anything else.

### 2. Study data in `eval_data.ipynb`

Open [eval_data.ipynb](eval_data.ipynb) to study input data before training.

### 3. Select a computed metric

The training target is derived from the raw columns in `C_TARGET_COLS` (default: PSNR and CLIP). To change which score is used, set `C_TARGET_FUNC`, `C_TARGET_COL`, and `C_TARGET_LABEL` together in [settings.py](settings.py):

```python
_FUNC_ALPHA, _FUNC_BETA, _FUNC_NORM = 1.0, 2.0, True
C_TARGET_FUNC = lambda df: compute_softplus_score(
    df, *C_TARGET_COLS, alpha=_FUNC_ALPHA, beta=_FUNC_BETA, normalize=_FUNC_NORM
)
C_TARGET_COL = f"softplus_score_a{_FUNC_ALPHA:g}-b{_FUNC_BETA:g}-n{_FUNC_NORM:d}"
C_TARGET_LABEL = f"Softplus Score ($\\alpha={_FUNC_ALPHA}$, $\\beta={_FUNC_BETA}$, $n={_FUNC_NORM:d}$)"
```

| Function | Description |
|---|---|
| `compute_softplus_score` | Smooth baseline-relative score; configure `_FUNC_ALPHA` and `_FUNC_BETA`. |
| `compute_naive_pareto_score` | $\max(0, \Delta\text{PSNR}) \cdot \max(0, \Delta\text{CLIP})$ relative to the baseline row per sample group. |
| `compute_agreement_score` | Similarity between the configured metric columns. |
| `compute_weighted_combined_score` | Weighted blend of the configured metric columns. |

Parameterized metrics should encode their kwargs in `C_TARGET_COL`, e.g. `softplus_score_a1-b2-n1`.

### 4. Adjust remaining settings

If wanted, further edit [settings.py](settings.py) to adjust training behavior before running the model.

| Setting | Description |
|---|---|
| `C_TARGET_COLS` / `C_TARGET_LABELS` | Raw metric columns (and display labels) fed into `C_TARGET_FUNC`. |
| `C_TARGET_FUNC` / `C_TARGET_COL` / `C_TARGET_LABEL` | Active score function, column name, and plot label. |
| `DEFAULT_T_START` / `DEFAULT_T_END` | Baseline $(t^*, t^{**})$ used by baseline-relative scores. |
| `ENCODER_MODEL` | Pretrained sentence-transformer checkpoint for the Siamese encoder. |
| `FREEZE_ENCODER` | If `True`, encoder weights are frozen during training. Default `True`. |
| `HEAD_TYPE` | `"CE"` (multiclass, default), `"CORAL"` (ordinal), or `"MSE"` (regression). |
| `CE_LOSS_TYPE` | `"one_hot_ce_loss"` or `"cost_sensitive_ce_loss"` (only when `HEAD_TYPE = "CE"`). |
| `USE_CLASS_WEIGHTS` | Weight loss by inverse class frequency to counteract label imbalance. |
| `LABEL_SMOOTHING` | Label smoothing for CE training (default `0.15`; only used when `HEAD_TYPE = "CE"`). |
| `SEED` | Global random seed. |
| `EPOCHS`, `BATCH_SIZE` | Training loop hyperparameters. |
| `ENCODER_LR`, `WEIGHT_DECAY` | Optimizer settings for the encoder. |
| `MLP_LR` | Learning rate for the MLP body and heads. |
| `MLP_WIDE`, `MLP_HIDDEN`, `MLP_INNER` | Hidden layer widths of the MLP body. |
| `MLP_DROPOUT` | Dropout rate applied inside the MLP body. |

### 5. Train the model

From the repo root:

```bash
cd ./models/classification
python train.py
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
| `_helpers.py` | Shared data-loading helper(s) |
| `model.py` | `OrdinalPairClassifier`, `SiameseEncoder`, and shared decode helpers |
| `head_ce.py` | CE multiclass head (default) |
| `head_coral.py` | CORAL ordinal head |
| `head_mse.py` | MSE regression head |
| `scores.py` | Scoring functions and data helpers |
| `train.py` | Training entry point |
| `eval_model.ipynb` | Post-training evaluation notebook |
| `eval_data.ipynb` | Dataset exploration notebook |
| `DESIGN.md` | Detailed architecture and design notes |
