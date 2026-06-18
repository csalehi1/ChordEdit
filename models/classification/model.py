"""
Ordinal String-Pair Classifier. Accepts two strings and predicts two
independent values (t_start and t_end), each from their respective set of
discrete values.

    1.  Shared Siamese encoder (pretrained transformer, mean-pool) encodes both
        strings in a single batched forward pass.
    2.  Combiner builds [emb_A | emb_B | emb_A-emb_B | emb_A⊙emb_B].
    3.  Shared MLP body produces a common feature vector.
    4.  Two independent heads — CORAL (ordinal) or MSE (scalar regression) —
        each predict one output value.
    5.  At inference, the head output is mapped to a bucket index.

"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from models.classification.settings import ENCODER_MODEL
from models.classification.head_coral import CoralHead, decode_ordinal
from models.classification.head_mse import RegressionHead, decode_regression
from models.classification.head_mae import mae_buckets  # noqa: F401


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
        """
        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, sentences: list[str]) -> torch.Tensor:
        """
        Encode a list of strings into fixed-size embeddings.
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
    Full string-pair classifier.

    Given strings A and B, predicts two independent scalar values (t_start and
    t_end). Head type is selected at construction time: 'CORAL' for cumulative-
    threshold ordinal decoding, 'MSE' for scalar regression snapped to the
    nearest bucket.
    """

    def __init__(
        self,
        encoder_name: str = ENCODER_MODEL,
        mlp_wide: int = 512,
        mlp_hidden: int = 256,
        mlp_inner: int = 128,
        dropout: float = 0.2,
        buckets1: torch.Tensor | None = None,
        buckets2: torch.Tensor | None = None,
        freeze_encoder: bool = True,
        head_type: str = "CORAL",
    ):
        """
        Arguments:
            encoder_name:  HuggingFace model name for the Siamese encoder.
            mlp_wide:      Width of the first MLP projection.
            mlp_hidden:    Width of the second MLP projection.
            mlp_inner:     Width of the third MLP projection (head input size).
            dropout:       Dropout probability between MLP layers.
            buckets1:      1-D tensor of ordered output values for t_start (head1).
                           Defaults to 10 evenly-spaced values in [0, 1].
            buckets2:      1-D tensor of ordered output values for t_end (head2).
                           Defaults to 10 evenly-spaced values in [0, 1].
            freeze_encoder: If True, encoder weights are fixed during training.
                            Recommended when training data is small (< ~1000 pairs).
            head_type:     "CORAL" for ordinal classification, "MSE" for scalar regression.
        """
        if head_type not in ("CORAL", "MSE"):
            raise ValueError(f"head_type must be 'CORAL' or 'MSE', got {head_type!r}")
        super().__init__()

        self.head_type = head_type
        self.encoder = SiameseEncoder(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if buckets1 is None:
            buckets1 = torch.linspace(0, 1, 10)
        if buckets2 is None:
            buckets2 = torch.linspace(0, 1, 10)
        self.register_buffer("buckets1", buckets1.float())
        self.register_buffer("buckets2", buckets2.float())

        h = self.encoder.hidden_dim

        # Combiner projects 4h features (concat + diff + hadamard) down to mlp_inner.
        # LayerNorm after the first linear stabilises training when the encoder is
        # fine-tuned, since embedding norms can shift significantly early in training.
        self.body = nn.Sequential(
            nn.Linear(h * 4, mlp_wide),
            nn.LayerNorm(mlp_wide),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_wide, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_inner),
            nn.ReLU(),
        )

        if head_type == "CORAL":
            self.head1 = CoralHead(mlp_inner, len(buckets1) - 1)
            self.head2 = CoralHead(mlp_inner, len(buckets2) - 1)
        elif head_type == "MSE":
            self.head1 = RegressionHead(mlp_inner)
            self.head2 = RegressionHead(mlp_inner)

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
        """
        return torch.cat([a, b, a - b, a * b], dim=-1)

    def _encode_pair(
        self, strings_a: list[str], strings_b: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both lists in a single batched encoder call to halve I/O cost.
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
        Compute raw head outputs for both targets.
        """
        emb_a, emb_b = self._encode_pair(strings_a, strings_b)
        combined = self._combine(emb_a, emb_b)
        features = self.body(combined)
        return self.head1(features), self.head2(features)

    def predict(
        self,
        strings_a: list[str],
        strings_b: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference and return bucket values (not indices).
        """
        training = self.training
        self.eval()
        with torch.no_grad():
            l1, l2 = self(strings_a, strings_b)
        self.train(training)

        if self.head_type == "CORAL":
            return self.buckets1[decode_ordinal(l1)], self.buckets2[decode_ordinal(l2)]
        elif self.head_type == "MSE":
            return self.buckets1[decode_regression(l1, self.buckets1)], self.buckets2[decode_regression(l2, self.buckets2)]
