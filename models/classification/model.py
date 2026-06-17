"""
Ordinal String-Pair Classifier
===============================
Accepts two strings and predicts two independent values (t_start and t_end),
each from their respective set of discrete values, using ordinal threshold encoding.

Model architecture
------------------
    1. Shared Siamese encoder (pretrained transformer + mean-pool) encodes both
        strings in a single batched forward pass.
    2. Combiner builds [emb_A | emb_B | emb_A-emb_B | emb_A⊙emb_B].
    3. Shared MLP body produces a common feature vector.
    4. Two independent linear heads emit K1-1 and K2-1 raw logits respectively,
        where K1 and K2 are the number of distinct buckets for each output.
    5. At inference, counting exceeded thresholds gives the bucket index.

The diff and Hadamard (element-wise product) interaction terms capture
*how* two strings differ regardless of how many words changed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
"""Pretrained transformer for the Siamese encoder."""


"""
Loss and decoding utilities.
"""


def ordinal_loss(logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """
    Ordinal binary cross-entropy loss over K-1 cumulative thresholds.

    Each class index c is encoded as a binary vector where the first c
    entries are 1 and the rest are 0 (shown here for K=5 buckets):

        class 0 -> [0, 0, 0, 0]
        class 1 -> [1, 0, 0, 0]
        class 2 -> [1, 1, 0, 0]
        class 3 -> [1, 1, 1, 0]
        class 4 -> [1, 1, 1, 1]

    Because off-by-one errors flip fewer thresholds than large misses, the
    loss naturally penalises large errors more — consistent with an ordinal
    scale.

    Arguments:
        logits:     (batch, K-1) raw head logits.
        target_idx: (batch,) integer class indices in [0, K].

    Returns:
        Scalar mean BCE loss.
    """
    k_minus_1 = logits.size(-1)
    thresholds = torch.arange(k_minus_1, device=logits.device).unsqueeze(0)   # (1, K-1)
    targets = (thresholds < target_idx.unsqueeze(1)).float()                   # (batch, K-1)
    return F.binary_cross_entropy_with_logits(logits, targets)


def decode_ordinal(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw head logits to bucket indices by counting exceeded thresholds.

    A threshold is considered exceeded when sigmoid(logit) > 0.5, which is
    equivalent to logit > 0.

    Arguments:
        logits: (..., K-1) tensor of raw logits.

    Returns:
        LongTensor of shape (...) with values in [0, K].
    """
    return (logits > 0).long().sum(dim=-1)


def mae_buckets(pred_idx: torch.Tensor, true_idx: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error over bucket indices.

    Preferred over accuracy because it rewards near-misses. An MAE of 1.0
    means the model is off by one bucket on average.

    Arguments:
        pred_idx: (N,) predicted bucket indices.
        true_idx: (N,) ground-truth bucket indices.

    Returns:
        Scalar MAE.
    """
    return (pred_idx - true_idx).abs().float().mean()


"""
Encoder.
"""


class SiameseEncoder(nn.Module):
    """
    Shared pretrained transformer encoder with mean-pooling.

    Both strings are encoded with the same weights (Siamese architecture).
    Gradients flow through the transformer, enabling fine-tuning.

    Mean-pooling over token embeddings (weighted by the attention mask) is
    used instead of the [CLS] token because it tends to generalise better
    for sentence-level tasks.
    """

    def __init__(self, model_name: str = ENCODER_MODEL):
        """
        Arguments:
            model_name: HuggingFace model identifier.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @property
    def hidden_dim(self) -> int:
        """Output embedding size."""
        return self.model.config.hidden_size

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute mask-weighted mean over the token dimension.

        Arguments:
            last_hidden:    (batch, seq, hidden)
            attention_mask: (batch, seq) with 1 for real tokens, 0 for padding.

        Returns:
            (batch, hidden) sentence embeddings.
        """
        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, sentences: list[str]) -> torch.Tensor:
        """
        Encode a list of strings into fixed-size embeddings.

        Arguments:
            sentences: N strings.

        Returns:
            (N, hidden_dim) embedding tensor on the encoder's device.
        """
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        output = self.model(**tokens)
        return self._mean_pool(output.last_hidden_state, tokens["attention_mask"])


"""
Classifier.
"""


class OrdinalPairClassifier(nn.Module):
    """
    Full string-pair ordinal classifier.

    Given strings A and B, predicts two independent scalar values (t_start and
    t_end) via cumulative-threshold ordinal decoding. Each head's output size
    is determined by the number of distinct values in the corresponding target.
    """

    def __init__(
        self,
        encoder_name: str = ENCODER_MODEL,
        mlp_hidden: int = 256,
        mlp_inner: int = 128,
        dropout: float = 0.2,
        buckets1: torch.Tensor | None = None,
        buckets2: torch.Tensor | None = None,
        freeze_encoder: bool = True,
    ):
        """
        Arguments:
            encoder_name:  HuggingFace model name for the Siamese encoder.
            mlp_hidden:    Width of the first MLP projection.
            mlp_inner:     Width of the second MLP projection (head input size).
            dropout:       Dropout probability between MLP layers.
            buckets1:      1-D tensor of ordered output values for t_start (head1).
                           Defaults to 5 evenly-spaced values in [0, 1].
            buckets2:      1-D tensor of ordered output values for t_end (head2).
                           Defaults to 5 evenly-spaced values in [0, 1].
            freeze_encoder: If True, encoder weights are fixed during training.
                            Recommended when training data is small (< ~1000 pairs).
        """
        super().__init__()

        self.encoder = SiameseEncoder(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if buckets1 is None:
            buckets1 = torch.linspace(0, 1, 5)
        if buckets2 is None:
            buckets2 = torch.linspace(0, 1, 5)
        self.register_buffer("buckets1", buckets1.float())
        self.register_buffer("buckets2", buckets2.float())

        h = self.encoder.hidden_dim

        # Combiner projects 4h features (concat + diff + hadamard) down to mlp_inner.
        # LayerNorm after the first linear stabilises training when the encoder is
        # fine-tuned, since embedding norms can shift significantly early in training.
        self.body = nn.Sequential(
            nn.Linear(h * 4, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_inner),
            nn.ReLU(),
        )

        self.head1 = nn.Linear(mlp_inner, len(buckets1) - 1)
        self.head2 = nn.Linear(mlp_inner, len(buckets2) - 1)

    """
    Internal helpers.
    """

    @staticmethod
    def _combine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Build the four-part interaction vector for an embedding pair.

        Concatenating the raw embeddings preserves individual semantics;
        the difference highlights what changed between the strings;
        the Hadamard product captures element-wise co-activation patterns.

        Returns:
            (batch, 4x hidden_dim)
        """
        return torch.cat([a, b, a - b, a * b], dim=-1)

    def _encode_pair(
        self, strings_a: list[str], strings_b: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both lists in a single batched encoder call to halve I/O cost.

        Returns:
            (emb_a, emb_b) each (N, hidden_dim).
        """
        n = len(strings_a)
        all_emb = self.encoder(strings_a + strings_b)   # (2N, h)
        return all_emb[:n], all_emb[n:]

    """
    Forward and predict.
    """

    def forward(
        self,
        strings_a: list[str],
        strings_b: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute raw logits for both output heads.

        Arguments:
            strings_a: N input strings (first of each pair).
            strings_b: N input strings (second of each pair).

        Returns:
            (logits1, logits2) — each (N, num_thresholds).
            Feed to ordinal_loss() for training or decode_ordinal() for inference.
        """
        emb_a, emb_b = self._encode_pair(strings_a, strings_b)
        combined = self._combine(emb_a, emb_b)              # (N, 4h)
        features = self.body(combined)                      # (N, mlp_inner)
        return self.head1(features), self.head2(features)

    def predict(
        self,
        strings_a: list[str],
        strings_b: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference and return bucket values (not indices).

        Arguments:
            strings_a: N input strings.
            strings_b: N input strings.

        Returns:
            (values1, values2) — each (N,) float tensors with elements in
            {0.0, 0.3, 0.6, 0.9, 1.0}.
        """
        self.eval()
        with torch.no_grad():
            l1, l2 = self(strings_a, strings_b)
        return self.buckets1[decode_ordinal(l1)], self.buckets2[decode_ordinal(l2)]
