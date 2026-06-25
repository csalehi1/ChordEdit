# Classification Model Design

## Task

Given a `(source_prompt, target_prompt)` string pair, predict two diffusion timestep parameters $t^*$ and $t^{**}$, referred to in code as `t_start` and `t_end`.

## Architecture

The diagram below shows the default CE path connected to the shared encoder and MLP body. CORAL and MSE are alternative head implementations selected at construction time via `HEAD_TYPE`; they replace the CE heads but are not part of the default forward path.

```mermaid
%%{init: {"flowchart": {"padding": 20}} }%%
flowchart TD
    IN_A["`**Source Prompt String**`"] & IN_B["`**Target Prompt String**`"]

    IN_A --> ENC
    IN_B --> ENC

    subgraph SIAM ["SiameseEncoder with shared weights"]
        ENC["`**AutoModel**
        Share weights to enforce a consistent embedding space. Hugging Face embedding model.`"]
        POOL["`**Mean-Pool**
        Discard [CLS] token and compute a mask-weighted average over all token positions. Generalizes better than [CLS] for sentence-level tasks.`"]
        ENC --> POOL
    end

    POOL -->|"split on batch dimension"| EMB_A["`**Source Embedding**`"] & EMB_B["`**Target Embedding**`"]

    EMB_A & EMB_B --> COMB

    subgraph COMB_BOX ["Combiner"]
        COMB["`**Concat. [ A | B | A − B | A ⊙ B ]**
        Concatenate to preserve individual semantics with difference for changes and Hadamard for similarities.`"]
    end

    COMB --> MLP1

    subgraph BODY ["MLP Body"]
        MLP1["`**Linear(4h, mlp_wide=512)**
        Expands the combiner vector into a wider representation to learn cross-modal interactions before compression.`"]
        LN1["`**LayerNorm(mlp_wide=512)**
        Stabilize training when the encoder is fine-tuned.`"]
        ACT1["`**ReLU()**
        Introduce nonlinearity after the first compression.`"]
        DROP1["`**Dropout(dropout=0.1)**`"]
        MLP2["`**Linear(mlp_wide=512, mlp_hidden=256)**
        Force the network to retain the most discriminative features with progressive compression.`"]
        ACT2["`**ReLU()**
        No LayerNorm because the distribution is already
        well-conditioned.`"]
        DROP2["`**Dropout(dropout=0.1)**`"]
        MLP3["`**Linear(mlp_hidden=256, mlp_inner=128)**
        Final shared representation before the task-specific heads. Both heads read from the same feature vector with independent weights.`"]
        MLP1 --> LN1 --> ACT1 --> DROP1 --> MLP2 --> ACT2 --> DROP2 --> MLP3
    end

    MLP3 --> CE1 & CE2

    subgraph H1_CE ["Head 1 for t_start (default: CE)"]
        CE1["`**ClassificationHead(K₁)**
        Linear layer producing K₁ logits. Trained with standard cross-entropy and label smoothing.`"]
        CE1 -->|"decode_classification"| IDX1["`**bucket1_index**`"]
    end

    subgraph H2_CE ["Head 2 for t_end (default: CE)"]
        CE2["`**ClassificationHead(K₂)**
        Linear layer producing K₂ logits. Same loss and decode path as head 1.`"]
        CE2 -->|"decode_classification"| IDX2["`**bucket2_index**`"]
    end

    IDX1 -->|"buckets1[bucket1_index]"| OUT1["`**t_start**`"]
    IDX2 -->|"buckets2[bucket2_index]"| OUT2["`**t_end**`"]

    subgraph ALT_CORAL ["Alternative: CORAL (HEAD_TYPE = CORAL)"]
        direction TB
        C1["`**CoralHead (t_start)**
        K₁−1 per-threshold classifiers. Ordinal BCE loss penalises large misses more than off-by-one errors.`"]
        C2["`**CoralHead (t_end)**
        K₂−1 per-threshold classifiers.`"]
        C1 -->|"decode_ordinal"| CIDX1["bucket1_index"]
        C2 -->|"decode_ordinal"| CIDX2["bucket2_index"]
    end

    subgraph ALT_MSE ["Alternative: MSE (HEAD_TYPE = MSE)"]
        direction TB
        M1["`**RegressionHead (t_start)**
        Sigmoid bounds output to (0,1), matching the bucket range.`"]
        M2["`**RegressionHead (t_end)**
        Same scalar regression head.`"]
        M1 -->|"decode_regression"| MIDX1["bucket1_index"]
        M2 -->|"decode_regression"| MIDX2["bucket2_index"]
    end
```

Strings `source_prompt` and `target_prompt` are fed into `SiameseEncoder` which outputs the concatenated embedded vector $\langle A \mid B \mid A - B \mid A \odot B \rangle$ where $\odot$ is the Hadamard product, element-wise multiplication. This output vector has size $4 \times 384 = 1536$.

The $1536$-dimensional vector is passed through an MLP body of `Linear` with $1536 \rightarrow 512$, `LayerNorm`, `ReLU`, `Dropout` with $0.1$, `Linear` with $512 \rightarrow 256$, `ReLU`, `Dropout` with $0.1$, and `Linear` with $256 \rightarrow 128$. The $128$-dimensional output is then routed to two parallel heads with `head1` for `t_start` and `head2` for `t_end`.

`HEAD_TYPE` selects which head implementation is wired in at construction time. The default is `"CE"`: each head outputs $K_i$ logits, training minimises cross-entropy against one-hot bucket labels (with optional label smoothing), and inference takes the argmax class. Alternative types are `CORAL` (ordinal thresholds) and `MSE` (scalar regression snapped to the nearest bucket). Each decoded index lies in $\{0, \dots, k_i-1\}$ and maps to a float value in $[0.0, 1.0]$ via the ordered `buckets1` / `buckets2` tensors.

### Encoder: `SiameseEncoder`

Uses `sentence-transformers/all-MiniLM-L6-v2` that outputs a hidden dimension of $384$. Both strings are encoded with *shared weights* in a single-batched forward pass. Token embeddings are reduced to a fixed-size sentence vector via mask-weighted mean pooling, which is more robust than the `[CLS]` token for sentence-level tasks.

The encoder is fine-tuned by default (`FREEZE_ENCODER = False`). When the encoder is frozen, only the MLP and heads learn. When fine-tuning is enabled, the optimizer assigns a lower learning rate to the encoder ($2 \times 10^{-5}$) than to the MLP ($10^{-3}$) to avoid destabilising the pretrained representations early in training.

### Combiner: `OrdinalPairClassifier._combine`

The four-part interaction vector $\langle A \mid B \mid A - B \mid A \odot B \rangle$ captures individual semantics for each prompt, direction and magnitude of the edit (differences between the prompts), and element-wise co-activation (similarities between the prompts).

### Classification Output Heads (default): `ClassificationHead`

Each head is a single `Linear(in_features, K)` layer producing $K$ raw logits per sample, where $K$ is the number of distinct bucket values for that target. Training uses `one_hot_ce_loss` (standard cross-entropy with optional `LABEL_SMOOTHING` and optional inverse-frequency class weights). Decoding picks the highest-logit class:

$$\hat{k} = \arg\max_j \; \text{logit}_j$$

`decode_head_pair` in `model.py` centralises the CORAL / CE / MSE decode paths and raises on unknown `head_type` values.

### Ordinal Output Heads (alternative): `CoralHead`

When `HEAD_TYPE = "CORAL"`, each head uses the CORAL (Consistent RAnk Logits) formulation from [Cao et al. 2020](https://arxiv.org/abs/1901.07884). The head maintains $k_i-1$ independent per-threshold weight vectors:

$$\text{logit}_j = w_j^\top x + b_j, \quad j = 0, \ldots, k_i-2$$

The ordinal structure is enforced entirely through the loss function rather than architectural weight sharing. Decoding counts how many thresholds are exceeded (`logit > 0`), giving a bucket index $[0, k_i-1]$. The ordinal encoding assigns each class a prefix of 1s:

```
Class 0: [0, 0, ..., 0, 0]
Class 1: [1, 0, ..., 0, 0]
...
Class k_i-2: [1, 1, ..., 1, 0]
Class k_i-1: [1, 1, ..., 1, 1]
```

Training uses binary cross-entropy over these threshold targets. Biases are initialised as `linspace(2, −2)` so that implied class probabilities are spread across the full range from the first training step.

### Regression Output Heads (alternative): `RegressionHead`

When `HEAD_TYPE = "MSE"`, each head is a single `Linear` layer with a sigmoid activation, producing one scalar per sample in $(0, 1)$:

$$\hat{y} = \sigma\!\left(w^\top x + b\right)$$

At inference, the continuous prediction is snapped to the nearest bucket index. Training minimises mean-squared error between the predicted scalar and the true bucket float value.

## Data Pipeline

1. Load `id_to_metrics_*.csv` and filter to rows where `t_delta == T_DELTA_TARGET`
2. Compute the configured target score (default: `naive_pareto_score`) and pick the row with the highest score per `sample_id`
3. Join with `id_to_string_pair.csv` to get `(source_prompt, target_prompt)`
4. Map discrete `t_start` / `t_end` float values to ordinal indices $0, 1, 2, \ldots$ using the sorted levels from the full (unfiltered) metrics CSV
5. Random split 80/10/10 for train/val/test

### Target Score Options

The score used to select the best row per `sample_id` is configurable in `settings.py` via `TARGET_METRIC`.

- **`weighted_combined_score`**: $\lambda_{\text{PSNR}} \cdot \hat{p} + \lambda_{\text{CLIP}} \cdot \hat{c}$, where $\hat{p}$ and $\hat{c}$ are min-max normalised PSNR and CLIP similarity, with $\lambda_{\text{PSNR}} = \lambda_{\text{CLIP}} = 0.5$ by default.

- **`agreement_score`**: $1 - \lvert p - c \rvert \,/\, \max(\lvert p - c \rvert)$, measuring how closely PSNR and CLIP agree on raw values.

- **`naive_pareto_score`** (active default): For each `sample_id` group, the baseline row is identified at $t_{\text{start}} = \text{PAPER\_T\_START} - \text{PAPER\_T\_DELTA}$, $t_{\text{end}} = \text{PAPER\_T\_END}$ (defaults $(0.75, 0.3)$). Each row receives score $\max(0, \Delta\text{PSNR}) \cdot \max(0, \Delta\text{CLIP})$ relative to that baseline; the baseline itself scores $0$.

- **`pareto_biased_score`**: Uses the same baseline as `naive_pareto_score`. With $s(t) = \operatorname{softplus}(t) - \log 2$, each row receives $m(a,b) = s(a-A) + s(b-B) + \alpha\, s(a-A)\, s(b-B)$ where $a$, $b$ are PSNR and CLIP and $A$, $B$ are the baseline values. Default $\alpha = 2$ (`_PARETO_BIAS_ALPHA` in `settings.py`). The baseline scores $0$; improvements are rewarded smoothly and regressions penalised.

## Training

| Hyperparameter | Value |
|---|---|
| Encoder | `all-MiniLM-L6-v2` (fine-tuned by default) |
| Head type | `CE` (default), `CORAL`, or `MSE` |
| CE loss | Standard cross-entropy with `LABEL_SMOOTHING = 0.1` |
| Optimizer | AdamW with `WEIGHT_DECAY = 0.01`; encoder lr $= 2 \times 10^{-5}$, MLP lr $= 10^{-3}$ |
| Epochs | $20$ |
| Batch size | $32$ |
| Dropout | $0.1$ |
| Class weights | Off by default (`USE_CLASS_WEIGHTS = False`) |
| Checkpoint | Best model by validation balanced accuracy on `t_start` |
