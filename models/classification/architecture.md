# Classification Model Architecture

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

    MLP3 --> HEAD1 & HEAD2

    subgraph H1 ["Head 1 for predicting t_start"]
        HEAD1{{"`**head_type?**`"}}
        HEAD1 -->|CORAL| C1["`**CoralHead**
        Per-threshold unshared weights give each boundary its own hyperplane. Ordinal BCE loss penalises large misses more than off-by-one errors.`"]
        HEAD1 -->|MSE| M1["`**RegressionHead**
        Sigmoid bounds output to (0,1), matching the bucket range. Snap to nearest-bucket.`"]
        C1 -->|"decode_ordinal"| IDX1["`**bucket1_index**`"]
        M1 -->|"decode_regression"| IDX1
    end

    subgraph H2 ["Head 2 for predicting t_start"]
        HEAD2{{"`**head_type?**`"}}
        HEAD2 -->|CORAL| C2["`**CoralHead**
        Per-threshold unshared weights give each boundary its own hyperplane. Ordinal BCE loss penalises large misses more than off-by-one errors.`"]
        HEAD2 -->|MSE| M2["`**RegressionHead**
        Sigmoid bounds output to (0,1), matching the bucket range. Snap to nearest-bucket.`"]
        C2 -->|"decode_ordinal"| IDX2["`**bucket2_index**`"]
        M2 -->|"decode_regression"| IDX2
    end

    IDX1 -->|"buckets1[bucket1_index]"| OUT1["`**t_start**`"]
    IDX2 -->|"buckets2[bucket2_index]"| OUT2["`**t_end**`"]
```

## Shapes at a glance

| Stage | Output shape | Notes |
|---|---|---|
| SiameseEncoder | `(2N, h)` | both strings in one batched call |
| emb_A, emb_B | `(N, h)` | split on batch dim |
| Combiner | `(N, 4h)` | concat + diff + Hadamard |
| MLP Body | `(N, mlp_inner)` | three Linear layers |
| CoralHead logits | `(N, K−1)` | K = number of buckets |
| RegressionHead output | `(N,)` | sigmoid-bounded scalar |
| Bucket indices | `(N,)` | decoded from either head |
| Final prediction | `(N,)` each | looked up from buckets1 / buckets2 |

## Loss functions (training only)

| Head type | Loss | File |
|---|---|---|
| CORAL | `ordinal_loss` — binary CE over K−1 cumulative threshold targets | [head_coral.py](head_coral.py) |
| MSE | `regression_loss` — MSE between sigmoid output and true bucket value | [head_mse.py](head_mse.py) |

## Evaluation metric

`mae_buckets` ([head_mae.py](head_mae.py)) — Mean Absolute Error over bucket indices. An MAE of 1.0 = off by one bucket on average.
