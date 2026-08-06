# Modified Classification (M + T)

Metric surrogate **M^** predicts `(psnr, clip)` from image/prompt embeddings and `(t_start, t_end)`. Timestep selector **T** uses M to pick the best grid cell.

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

  subgraph regressor [Trainable SurrogateRegressor]
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

**M^** bottlenecks image/mask/text embeddings, encodes timesteps with Fourier features, then predicts via separate towers: a standard MLP for PSNR (timesteps concatenated) and a FiLM-conditioned MLP for CLIP (timesteps as modulation). **T** has no trainable weights: it queries M over the discrete grid and selects the best cell.

## Setup

```bash
conda activate chordedit
cd models/modified-classification
```

Requires ChordEdit weights at `CHORD_EDIT_MODEL_ROOT` for the selected `CHORD_EDIT_MODEL` (see `settings.py`).

## Metric surrogate, M^

### 0. Cache embeddings (optional)

Pack scattered per-sample embedding `.pt` files into a packed table at `.cache/packed_embeddings/`:

```bash
python train_m.py --skip-model  # CPU-only; builds the caches, then exits.
```

Scattered files live under `EMBEDDINGS_DIR/annotation_embeddings/<sample_id>/` as packing-ready flat float32 vectors (`image.pt`, `mask.pt`, `source.pt`, `target.pt`), so packing is a pure stack. `mask.pt` may not exist yet for every sample while the annotation pipeline backfills it; those samples pack as zero mask rows, and the cache repacks itself once the files appear. Each packed cache carries a `meta` dict (model, pipeline type, layout, dims, provenance).

Later `train_m` and eval reuse the packed cache. Encoders are inherited from the ChordEdit pipeline and are never trainable. If no packed or scattered cache can cover the request, `embeddings.get_embeddings` raises.

### 1. Train

```bash
python train_m.py
```

Writes a run to `runs/<dataset>/` (named by `RUN_NAME`, else a timestamp) with `regressor_weights.pt`, the `settings.json` it used, train/val/test splits, and `mean_surface.pt` (the train split's mean true delta surface, computed from the labels alone).

Targets are per-sample normalized deltas Delta. With `M_TARGET_SPACE="residual"` (recommended) the towers regress each image's deviation from the train mean surface and T adds the surface back at selection time, so an uninformative prediction falls back to the population-best cell instead of never deviating. `"delta"` regresses the full deltas with no offset (the uncentered comparison arm). Runs from before this patch can be retrofitted with `python calibrate_t.py --run-dir <run_dir>`.

### 2. Evaluate

Open and run all cells in `eval_m.ipynb`. Set `CUDA_VISIBLE_DEVICES` in the setup cell to pick the GPU. The notebook loads the latest run (or set `RUN_DIR`) and uses that run's saved `settings.json` for paths/hyperparameters. It reports prediction error, calibration, and grid-surface fidelity.

### Timestep selector, T

Run after M^ training completes.

### 1. Evaluate selection metrics

```bash
python train_t.py
# optional: python train_t.py --run-dir runs/UltraEdit_Region_100/<timestamp> --gpu 1
```

Reports regret, Spearman correlation, and deviate-gate precision/recall on the test split. Saves `t_train_metrics.json`, `t_test_selections.json`, and `id_to_predictions_<commit>.csv` (`sample_id`, `pred_t_start`, `pred_t_end`) tagged with the commit that produced it.

### 2. Visualize

Open and run all cells in `eval_t.ipynb` for selection diagnostics and plots.

## Configuration

`settings.json` holds every tunable. `settings.py` reads it once at import and derives the fixed structure - paths, column names, the phi partials - from those values. To run a different configuration, edit `settings.json` or pass `--settings-path` pointing at another copy:

```bash
CUDA_VISIBLE_DEVICES=4 python train_m.py --settings-path /tmp/my_config.json
```

Every run saves the config it used to `<run_dir>/settings.json`, and `train_t.py` (and the analysis scripts) replay a run from that snapshot when given `--run-dir` - so evaluation always sees the training config, whatever `--settings-path` says. A key missing from the file is an error, not a silent default.

`settings.py` binds its constants at import time, before any script's argparse runs, so it reads `--settings-path` straight off `sys.argv`; the entry points declare the flag as well so it appears in `--help`.
