# Modified Classification (M + T)

Metric surrogate **M** predicts `(psnr, clip)` from image/prompt embeddings and `(t_start, t_end)`. Timestep selector **T** uses M to pick the best grid cell (no extra trainable weights).

Configure data paths and hyperparameters in `settings.py` before running.

## Architecture

```mermaid
flowchart TB
  subgraph inputs [Inputs]
    img[Image]
    mask[Mask]
    src[Source prompt]
    tar[Target prompt]
    ts["(t_start, t_end)"]
  end

  subgraph frozen [Frozen ChordEdit encoders]
    vae[VAE image encoder]
    te[Text encoder + mean-pool]
  end

  img --> vae
  mask --> vae
  src --> te
  tar --> te

  vae --> img_emb[img_emb]
  vae --> mask_emb[mask_emb]
  te --> src_emb[src_emb]
  te --> tar_emb[tar_emb]

  subgraph regressor [Trainable MetricRegressor]
    img_proj[img_proj]
    mask_proj[mask_proj]
    text_combine["combine_text: cat, diff, product"]
    text_proj[text_proj]
    fourier["Fourier features of t"]
    t_enc[t_encoder]

    img_emb --> img_proj
    mask_emb --> mask_proj
    src_emb --> text_combine
    tar_emb --> text_combine
    text_combine --> text_proj
    ts --> fourier --> t_enc

    context["context = cat(img, mask, text)"]
    img_proj --> context
    mask_proj --> context
    text_proj --> context

    psnr_tower["PSNR MLP: cat(context, t_feat)"]
    clip_tower["CLIP FiLM-MLP: context ⊕ t_feat"]
    context --> psnr_tower
    t_enc --> psnr_tower
    context --> clip_tower
    t_enc --> clip_tower

    psnr_tower --> psnr[psnr]
    clip_tower --> clip[clip]
  end

  subgraph T [T — timestep selector]
    grid["Evaluate M on (t_start, t_end) grid"]
    scalar["Scalarize → combined score m"]
    argmax["argmax + deviate-or-default gate"]
    psnr --> grid
    clip --> grid
    grid --> scalar --> argmax
    argmax --> out["(t_start*, t_end*)"]
  end
```

**M** bottlenecks image/mask/text embeddings, encodes timesteps with Fourier features, then predicts via separate towers: a standard MLP for PSNR (timesteps concatenated) and a FiLM-conditioned MLP for CLIP (timesteps as modulation). **T** has no trainable weights — it queries M over the discrete grid and selects the best cell.

## Setup

```bash
conda activate chordedit
cd models/modified-classification
```

Requires SD-Turbo weights at `SD_TURBO_ROOT` (see repo root `README.md`).

## M — metric surrogate

**1. Train**

```bash
python train_m.py
```

Writes a timestamped run to `outputs/<dataset>/`, including `regressor_weights.pt`, a copy of `settings.py`, and train/val/test splits.

**2. Evaluate**

Open and run all cells in `eval_m.ipynb`. Set `CUDA_VISIBLE_DEVICES` in the setup cell to pick the GPU. The notebook loads the latest run (or set `RUN_DIR`) and uses that run's saved `settings.py` for paths/hyperparameters. It reports prediction error, calibration, and grid-surface fidelity.

## T — timestep selector

Run after M training completes.

**1. Evaluate selection metrics**

```bash
python train_t.py
# optional: python train_t.py --run-dir outputs/UltraEdit_Region_100/<timestamp> --gpu 1
```

Reports regret, Spearman correlation, and deviate-gate precision/recall on the test split. Saves `t_train_metrics.json` and `t_test_selections.json`.

**2. Visualize**

Open and run all cells in `eval_t.ipynb` for selection diagnostics and plots.

## Shared modules

| File | Role |
|------|------|
| `model_t.py` | T selection, scalarization, and eval metrics (regret, Spearman, gate) |
| `_helpers.py` | Shared helpers: run settings save/load, pooling, normalization, ranking loss |
