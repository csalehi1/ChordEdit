"""
Prototype: extend OrdinalPairClassifier with a frozen, precomputed CLIP
source-image embedding, concatenated into the existing text-pair combiner
vector before the shared MLP body. Everything else (SiameseEncoder, CE
heads, bucket decoding) is reused unchanged from model.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.classification.model import SiameseEncoder, decode_head_pair
from models.classification.head_ce import ClassificationHead
from models.classification.settings import (
    ENCODER_MODEL, MLP_WIDE, MLP_HIDDEN, MLP_INNER, MLP_DROPOUT,
)


class OrdinalPairClassifierWithImage(nn.Module):
    def __init__(
        self,
        image_dim: int,
        encoder_name: str = ENCODER_MODEL,
        mlp_wide: int = MLP_WIDE,
        mlp_hidden: int = MLP_HIDDEN,
        mlp_inner: int = MLP_INNER,
        dropout: float = MLP_DROPOUT,
        buckets1: torch.Tensor | None = None,
        buckets2: torch.Tensor | None = None,
        freeze_encoder: bool = True,
        head_type: str = "CE",
    ):
        if head_type != "CE":
            raise ValueError("prototype only supports head_type='CE'")
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
        self.predict_start = len(buckets1) > 1
        self.predict_end = len(buckets2) > 1

        h = self.encoder.hidden_dim
        combined_dim = h * 4 + image_dim

        self.body = nn.Sequential(
            nn.Linear(combined_dim, mlp_wide),
            nn.LayerNorm(mlp_wide),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_wide, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_inner),
        )

        if self.predict_start:
            self.head1 = ClassificationHead(mlp_inner, len(buckets1))
        if self.predict_end:
            self.head2 = ClassificationHead(mlp_inner, len(buckets2))

    def _encode_pair(self, strings_a: list[str], strings_b: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(strings_a)
        all_emb = self.encoder(strings_a + strings_b)
        return all_emb[:n], all_emb[n:]

    def decode_bucket_indices(self, out1, out2, *, batch_size: int | None = None):
        if batch_size is None:
            batch_size = out1.size(0) if out1 is not None else out2.size(0)
        device = self.buckets1.device
        return decode_head_pair(
            self.head_type, out1, out2,
            buckets1=self.buckets1, buckets2=self.buckets2,
            predict_start=self.predict_start, predict_end=self.predict_end,
            batch_size=batch_size, device=device,
        )

    def forward(self, strings_a: list[str], strings_b: list[str], image_emb: torch.Tensor):
        emb_a, emb_b = self._encode_pair(strings_a, strings_b)
        text_combined = torch.cat([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], dim=-1)
        combined = torch.cat([text_combined, image_emb], dim=-1)
        features = self.body(combined)
        out1 = self.head1(features) if self.predict_start else None
        out2 = self.head2(features) if self.predict_end else None
        return out1, out2
