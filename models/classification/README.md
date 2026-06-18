# Classification Model

Predicts two diffusion timestep parameters (`t_start`, `t_end`) from a `(source_prompt, target_prompt)` string pair. These parameters control where in the diffusion trajectory an image edit is applied.

## Architecture

A **Siamese encoder** (`all-MiniLM-L6-v2`, frozen by default) embeds both strings in a single batched pass and combines them as `[A | B | A-B | A⊙B]`. The resulting 1536-dim vector is passed through a shared MLP body, then routed to two independent heads — one per output value.

Two head types are supported:
- **CORAL** (default) — ordinal classification using cumulative threshold logits; guarantees rank-consistent predictions.
- **MSE** — scalar regression snapped to the nearest bucket.

See [DESIGN.md](DESIGN.md) for full architecture details.

## Workflow

### 1. Configure `settings.py`

Edit [settings.py](settings.py) to point at your data and tune training behavior before running anything else.

| Setting | Description |
|---|---|
| `METRICS_CSV` | Path to `id_to_metrics_*.csv` |
| `OUTPUTS_SUBDIR` | Where checkpoints and plots are saved |
| `N_BUCKETS_START`, `N_BUCKETS_END` | Number of distinct `t_start` / `t_end` values in the data |
| `COMPUTED_METRIC_FN` | Scoring function used to select the best row per prompt pair |
| `TARGET_COLUMN` | Which metric column to train on, defaults to computed metric |
| `HEAD_TYPE` | May be `"CORAL"` (ordinal) or `"MSE"` (regression) |
| `FREEZE_ENCODER` | Freeze encoder weights |
| `USE_CLASS_WEIGHTS` | Weight loss values by inverse class frequency to handle label imbalance |
| `EPOCHS`, `BATCH_SIZE`, `ENCODER_LR`, `BODY_LR` | Training hyperparameters |

### 2. Train the model

From the repo root:

```bash
python -m models.classification.classify
```

This loads the CSVs, builds the dataset, trains the model, and saves checkpoints and metric plots to `OUTPUTS_DIR`.

### 3. Evaluate in `eval_model.ipynb`

Open [eval_model.ipynb](eval_model.ipynb) to inspect model performance after training. The notebook loads the most recent saved checkpoint and walks through:

- Per-split metrics
- *Confusion matrices*
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
| `visualizer.ipynb` | Additional visualization utilities |
| `DESIGN.md` | Detailed architecture and design notes |
