# Classification Model Design

## Task

Given a `(source_prompt, target_prompt)` string pair, predict two diffusion timestep parameters $t^*$ and $t^{**}$, referenced in code as `t_start` and `t_end`. Each takes a value from an ordered linear space in $[0.0, 1.0]$. The choice of these parameters controls where in the diffusion trajectory the edit is applied.

## Architecture

Strings `source_prompt` and `target_prompt` are fed into `SiameseEncoder` which outputs as the concatenated embeded vector $\langle A \mid B \mid A - B \mid A \odot B \rangle$ where $\odot$ is the Hadamard product, element-wise multuplication. This output vector has size $4 \times 384 = 1536$.

The $1536$-dimensional vector is passed through an MLP body of `Linear` with $1536 \rightarrow 512$, `LayerNorm`, `ReLU`, `Dropout` with $0.2$, `Linear` with $512 \rightarrow 256$, `ReLU`, `Dropout` with $0.2$, `Linear` with $256 \rightarrow 128$, and lastly `ReLU`. The $128$-dimensional output is then routed to two parallel CORAL heads with `head1` for `t_start` and `head2` for `t_end`. Each head has a single shared weight vector and $k_i-1$ per-threshold scalar biases, producing $k_i-1$ threshold logits where $k_i$ is the number of distinct bins for $t_i$. Each head's logits are decoded to a bucket index in $\{0, \dots, k\}$, which maps to a float value in $[0.0, 1.0]$.

### Encoder: `SiameseEncoder`

Uses `sentence-transformers/all-MiniLM-L6-v2` that outputs a hidden dimension of $384$. Both strings are encoded with *shared weights* in a single-batched forward pass. Token embeddings are reduced to a fixed-size sentence vector via mask-weighted mean pooling, which is more robust than the `[CLS]` token for sentence-level tasks.

Weights for the the encoder are frozen by default during training, meaning that the MLP is the only component that learns. This is appropriate when training data is small because the pretrained embeddings already capture semantic similarity well.

### Combiner: `OrdinalPairClassifier._combine`

The four-part interaction vector $\langle A \mid B \mid A - B \mid A \odot B \rangle$ captures individual semantics for each prompt, direction and magnitude of the edit (differences between the prompts), and element-wise co-activation (similarities between the prompts).

### Ordinal Output Heads: `CoralHead`

Each head uses the CORAL (Consistent RAnk Logits) formulation from [Cao et al. 2020](https://arxiv.org/abs/1901.07884). Instead of $k_i-1$ independent linear projections, each head has a single shared weight vector $w$ with $k_i-1$ per-threshold scalar biases $b_0, b_1, \ldots, b_{k_i-2}$:

$$\text{logit}_j = w^\top x + b_j, \quad j = 0, \ldots, k_i-2$$

Decoding counts how many thresholds are exceeded (`logit > 0`), giving a bucket index $[0, k_i-1]$. The ordinal encoding assigns each class a prefix of 1s:

```
Class 0: [0, 0, ..., 0, 0]
Class 1: [1, 0, ..., 0, 0]
...
Class k_i-2: [1, 1, ..., 1, 0]
Class k_i-1: [1, 1, ..., 1, 1]
```

Training uses binary cross-entropy over these threshold targets (instead of softmax and cross-entropy), which respects the ordered structure of the output space.

The shared-weight structure gives CORAL two properties that independent `Linear` heads with $128 \rightarrow k_i - 1$ do not have:

1. **Rank consistency**: because all $k_i-1$ thresholds are a rigid shift of the same dot product $\mathbf{w}^\top \mathbf{x}$, the decoded probabilities are monotone by construction: $P(Y > 0) \geq P(Y > 1) \geq \cdots \geq P(Y > k_i-2)$ for every input. Independent heads can produce incoherent patterns such as $[0, 1, 0, 1]$ that correspond to no valid ordinal class.

2. **Coherent gradient signal** — with independent heads, the BCE gradients for different thresholds are decoupled and can conflict (e.g., threshold 1 being pushed up while threshold 0 is pushed down for the same sample). With shared weights every threshold gradient accumulates into the same $\mathbf{w}$, so the updates always agree on direction.

Biases are initialised as `linspace(2, −2)` so that the implied class probabilities are spread across the full range from the first training step rather than all starting at 0.5.

## Data Pipeline

1. Filter `id_to_metrics.csv` to rows where `t_delta == 0.0`
2. Pick the row with the highest `combined_score`, equal-weighted average of min-max normalised PSNR and CLIP similarity to the target image
3. Join with `id_to_string_pair.csv` to get `(source_prompt, target_prompt)`
4. Map discrete `t_start` / `t_end` float values to ordinal indices `0, 1, 2, …`
5. Random split 80/10/10 for train/val/test

## Training

| Hyperparameter | Value |
|---|---|
| Encoder | `all-MiniLM-L6-v2` (frozen) |
| Optimizer | Adam, lr = 1e-3 |
| Epochs | $20$ |
| Batch size | 32 |
| Dropout | 0.2 |
