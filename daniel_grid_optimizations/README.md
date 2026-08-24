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
| `--embeddings-root` | Parent dir for per-sample embeddings; the dataset dir name is appended (e.g. `.../embeddings/sd_turbo` -> `.../embeddings/sd_turbo/UltraEdit_Region_10000`). |
| `--generated-root` | |
| `--gpus` | |
| `--max-samples` | |

| Flag | Meaning |
| --- | --- |
| `--add-plots` | |
| `--cache-masks` | Also save `mask.pt` per sample (flat VAE latent of the RGB annotation mask). Incompatible with `--skip-embeddings`. |
| `--skip-embeddings` | Do not write per-sample embedding files. |
| `--skip-generated` | Encode-only mode: save embeddings, skip all UNet/grid work. |
| `--diagonal-optimization` | |

### Embedding format

All embeddings derive from the same pipeline tensors that condition
generation. Two sets are written per sample: a packing-ready flat/pooled
set (one float32 vector per file) consumed by the MLP/classification
models, and a per-token set consumed by the attention_predictor model.

Flat/pooled set:

- `image.pt`: flattened VAE latent, `(16384,)` for sd-turbo at 512px.
- `source.pt` / `target.pt`: masked-mean-pooled CLIP text vectors, `(1024,)`,
  numerically identical to the classification repo's `mean_pool` /
  `encode_text_pooled` (attention masks from re-tokenizing the
  bracket-stripped prompts).
- `mask.pt` (only with `--cache-masks`): flattened VAE latent of the
  RGB-converted annotation mask, same shape as `image.pt`; encoded in the
  same batch=2 VAE forward as the image. Skipped for samples without a
  mask, and the `id_to_embeddings` CSV gains a `mask_embedding` column
  (empty for maskless samples).

Per-token set (for the attention_predictor model; not listed in the
`id_to_embeddings` CSV, which keeps its existing schema):

- `image_tokens.pt`: unflattened VAE latent, `(4, 64, 64)` for sd-turbo
  at 512px (same values as `image.pt`, shape preserved).
- `source_tokens.pt` / `target_tokens.pt`: raw CLIP `last_hidden_state`
  rows, `(77, 1024)`, unpooled and with no attention mask applied.

### Backfilling token embeddings

Datasets generated before the per-token set existed only have the
flat/pooled files. The resume check requires both sets, so re-running in
encode-only mode backfills the missing `*_tokens.pt` files (and rewrites
the flat/pooled files with identical content):

```bash
python grid_generate.py \
    --data-root /shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_10000 \
    --model-root /shared/ssd_30T/mirick/models/sd-turbo \
    --embeddings-root /shared/ssd_30T/mirick/embeddings/sd_turbo \
    --skip-generated --gpus 5 6 7
```

Pass `--cache-masks` when the dataset was originally built with it, so the
resume check keeps requiring `mask.pt`.

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

`--generated-root` is the dataset folder (not the parent `ultra_edit/` dir):

```text
/shared/ssd_30T/mirick/generated/ultra_edit/UltraEdit_Region_10/
  id_to_inputs_ultraeditregion10.csv
  grids/{sample_id}/cells/t_start_*__t_end_*.jpg
```

```bash
python grid_eval.py --generated-root /shared/ssd_30T/mirick/generated/ultra_edit/UltraEdit_Region_10
```
