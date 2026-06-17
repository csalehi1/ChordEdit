# Classification Model Design

## Task

Given a `(source_prompt, target_prompt)` string pair, predict two diffusion timestep
parameters — `t_start` and `t_end` — each taking a value from the ordered set
`{0.0, 0.3, 0.6, 0.9, 1.0}`. These parameters control where in the diffusion
trajectory the edit is applied.

## Architecture

`source_prompt` and `target_prompt` are fed into `SiameseEncoder` which outputs as the concatenated embeded vector $\langle A \mid B \mid A - B \mid A \odot B \rangle$ where $\odot$ is the Hadamard product, element-wise multuplication. This output vector has size $4 \times 384 = 1536$.

The $1536$-dimensional vector is passed through an MLP body of `Linear` with $1536 \rightarrow 256$, `ReLU`, `Dropout` with $0.2$, `Linear` with $1536 \rightarrow 128$, and lastly `ReLU`. The $128$-dimensional output is then routed to two parallel heads with `head1` for `t_start` and `head2` for `t_end`. Each `Linear` with $128 \rightarrow k_i-1$ produces $k_i-1$ threshold logits where $k_i$ is the number of distinct bins for $t_i$. Each head's logits are decoded to a bucket index in $\{0, \dots, k\}$, which maps to a float value in $[0.0, 1.0]$.

### Encoder: `SiameseEncoder`

Uses `sentence-transformers/all-MiniLM-L6-v2` that outputs a hidden dimension of $384$. Both strings are encoded with *shared weights* in a single-batched forward pass. Token embeddings are reduced to a fixed-size sentence vector via mask-weighted mean pooling, which is more robust than the `[CLS]` token for sentence-level tasks.

Weights for the the encoder are frozen by default during training, meaning that the MLP is the only component that learns. This is appropriate when training data is small because the pretrained embeddings already capture semantic similarity well.

### Combiner: `OrdinalPairClassifier._combine`

The four-part interaction vector $\langle A \mid B \mid A - B \mid A \odot B \rangle$ caputes individual semantics of each prompt, irection and magnitude of the edit (differences between the prompts), and element-wise co-activation (similarities between the prompts).

### Ordinal Output Heads

Each head emits $k_i-1$ logits that represent cumulative thresholds. Decoding counts
how many thresholds are exceeded (`logit > 0`), giving a bucket index 0–4.

This ordinal encoding naturally penalises large misses more than off-by-one errors
because each class is encoded as a prefix of 1s:

```
class 0 -> [0, 0, 0, 0]
class 1 -> [1, 0, 0, 0]
class 2 -> [1, 1, 0, 0]
class 3 -> [1, 1, 1, 0]
class 4 -> [1, 1, 1, 1]
```

Training uses binary cross-entropy over these threshold targets (instead of softmax and
cross-entropy), which respects the ordered structure of the output space.

## Data Pipeline

1. **Filter** `id_to_metrics.csv` to rows where `t_delta == 0.0`
2. **Select best row per sample** — pick the row with the highest `combined_score`
   (equal-weighted average of min-max normalised PSNR and CLIP similarity to the
   target image)
3. **Join** with `id_to_string_pair.csv` to get `(source_prompt, target_prompt)`
4. **Map** discrete `t_start` / `t_end` float values to ordinal indices `0, 1, 2, …`
5. **Split** 80 / 10 / 10 (train / val / test)

## Training

| Hyperparameter | Value |
|---|---|
| Encoder | `all-MiniLM-L6-v2` (frozen) |
| Optimizer | Adam, lr = 1e-3 |
| Epochs | $20$ |
| Batch size | 32 |
| Dropout | 0.2 |

## Evaluation Metrics

- **MAE** over bucket indices (primary — rewards near-misses; MAE of 1.0 = off by one bucket)
- **Accuracy** per head and **joint accuracy** (both heads correct simultaneously)
