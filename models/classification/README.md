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

### Configure `settings.json`

Every tunable lives in [settings.json](settings.json); [settings.py](settings.py) reads that file and derives the rest (paths, column names, the score partials). To run a different configuration, edit `settings.json` or point `CE_SETTINGS_JSON` at another copy of it.