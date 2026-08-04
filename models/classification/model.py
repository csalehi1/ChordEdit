"""
Ordinal Embedding-Pair Classifier. Accepts precomputed ChordEdit embeddings
(image, mask, source prompt, target prompt) and predicts two independent
values (t_start and t_end), each from their respective set of discrete values.

    C(img_emb, mask_emb, src_emb, tar_emb) -> (t_start, t_end)

    1.  Image and mask projections bottleneck the flattened VAE latents;
        the text projection consumes [src | tar | src-tar | src*tar]
        (mirroring the modified model's SurrogateRegressor input side).
    2.  Shared MLP body produces a common feature vector.
    3.  Two independent heads - CORAL (ordinal), MSE (scalar regression), or
        CE (multiclass) - each predict one output value.
    4.  At inference, the head output is mapped to a bucket index.

Embeddings come from the frozen ChordEdit encoders via embeddings.py; no
encoder lives inside the model.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from head_coral import CoralHead, bucket_scores_ordinal, decode_ordinal
from head_mse import RegressionHead, bucket_scores_regression, decode_regression
from head_ce import ClassificationHead, bucket_scores_classification, decode_classification

from settings import *


def mae_buckets(pred_idx: torch.Tensor, true_idx: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error over bucket indices. An MAE of 1.0 means off by one bucket on average."""
    return (pred_idx - true_idx).abs().float().mean()


def combine_text_embeddings(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concat, difference, and Hadamard product of an embedding pair (4 * dim).

    Concatenating the raw embeddings preserves individual semantics; the
    difference highlights what changed between the prompts; the Hadamard
    product captures element-wise co-activation patterns.
    """
    return torch.cat([a, b, a - b, a * b], dim=-1)


def decode_head_pair(
    head_type: str,
    out1: torch.Tensor | None,
    out2: torch.Tensor | None,
    *,
    buckets1: torch.Tensor,
    buckets2: torch.Tensor,
    predict_start: bool,
    predict_end: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw head outputs to bucket indices for both targets."""
    
    if predict_start:
        if out1 is None:
            raise ValueError("out1 is required when t_start has >1 bucket")
        if head_type == "CORAL":
            p1 = decode_ordinal(out1)
        elif head_type == "CE":
            p1 = decode_classification(out1)
        elif head_type == "MSE":
            p1 = decode_regression(out1, buckets1)
        else:
            raise ValueError(f"head_type must be 'CORAL', 'MSE', or 'CE', got {head_type!r}")
    else:
        p1 = torch.zeros(batch_size, dtype=torch.long, device=device)

    if predict_end:
        if out2 is None:
            raise ValueError("out2 is required when t_end has >1 bucket")
        if head_type == "CORAL":
            p2 = decode_ordinal(out2)
        elif head_type == "CE":
            p2 = decode_classification(out2)
        elif head_type == "MSE":
            p2 = decode_regression(out2, buckets2)
        else:
            raise ValueError(f"head_type must be 'CORAL', 'MSE', or 'CE', got {head_type!r}")
    else:
        p2 = torch.zeros(batch_size, dtype=torch.long, device=device)

    return p1, p2


"""
Embedding projections.
"""


class ConvImageProjector(nn.Module):
    """Encode a flattened VAE latent with convolutions instead of one Linear.

    The image and mask embeddings are flattened (C, S, S) VAE latents, so a
    single Linear over 16k inputs discards all spatial structure - including
    how large and where the edit mask is. This folds the latent back to
    (C, S, S) and downsamples.
    """

    def __init__(self, flat_dim: int, out_dim: int, channels: int = 4):
        super().__init__()
        spatial_sq = flat_dim // channels
        side = int(round(spatial_sq ** 0.5))
        if channels * side * side != flat_dim:
            raise ValueError(f"flat_dim={flat_dim} is not {channels} x S x S for an integer S")
        self.channels, self.side = channels, side

        widths = [channels, 32, 64, 128, 128]
        layers: list[nn.Module] = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Conv2d(a, b, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, b), nn.SiLU()]
        self.stem = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1] * (side // 2 ** (len(widths) - 1)) ** 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.channels, self.side, self.side)
        return self.head(self.stem(x))


"""
Classifier.
"""


class OrdinalPairClassifier(nn.Module):
    """
    Full embedding-pair classifier.

    Given the four precomputed embeddings for a sample, predicts two
    independent scalar values (t_start and t_end). Head type is selected at
    construction time: 'CORAL' for cumulative-threshold ordinal decoding,
    'MSE' for scalar regression snapped to the nearest bucket, or 'CE' for
    multiclass classification.
    """

    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        *,
        img_proj_dim: int = IMG_PROJ_DIM,
        text_proj_dim: int = TEXT_PROJ_DIM,
        mlp_wide: int = MLP_WIDE,
        mlp_hidden: int = MLP_HIDDEN,
        mlp_inner: int = MLP_INNER,
        dropout: float = MLP_DROPOUT,
        buckets1: torch.Tensor = torch.linspace(0, 1, 10),
        buckets2: torch.Tensor = torch.linspace(0, 1, 10),
        head_type: str = HEAD_TYPE,
    ):
        super().__init__()

        self.head_type = head_type
        self.register_buffer("buckets1", buckets1.float())
        self.register_buffer("buckets2", buckets2.float())
        self.predict_start = len(buckets1) > 1
        self.predict_end = len(buckets2) > 1

        # Project the embeddings to the MLP input dimension.
        def image_projection() -> nn.Module:
            if IMG_ENCODER == "conv":
                return ConvImageProjector(img_dim, img_proj_dim)
            elif IMG_ENCODER == "linear":
                return nn.Sequential(
                    nn.Linear(img_dim, img_proj_dim),
                    nn.LayerNorm(img_proj_dim),
                    nn.ReLU(),
                )
            raise ValueError(f"Unknown {IMG_ENCODER=}")

        self.img_proj = image_projection()
        self.mask_proj = image_projection()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim * 4, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.ReLU(),
        )

        # Shared MLP body over the concatenated projections. LayerNorm after
        # the first linear stabilizes training against embedding-norm spread.
        body_in = img_proj_dim * 2 + text_proj_dim
        self.body = nn.Sequential(
            nn.Linear(body_in, mlp_wide),
            nn.LayerNorm(mlp_wide),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_wide, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_inner),
        )

        if head_type == "CORAL":
            if self.predict_start:
                self.head1 = CoralHead(mlp_inner, len(buckets1) - 1)
            if self.predict_end:
                self.head2 = CoralHead(mlp_inner, len(buckets2) - 1)
        elif head_type == "MSE":
            if self.predict_start:
                self.head1 = RegressionHead(mlp_inner)
            if self.predict_end:
                self.head2 = RegressionHead(mlp_inner)
        elif head_type == "CE":
            if self.predict_start:
                self.head1 = ClassificationHead(mlp_inner, len(buckets1))
            if self.predict_end:
                self.head2 = ClassificationHead(mlp_inner, len(buckets2))
        else:
            raise ValueError(f"Unknown {head_type=}")

    """
    Internal helpers.
    """

    def _get_context(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Combine image/mask/text embeddings into a single context vector."""
        text_emb = combine_text_embeddings(src_emb, tar_emb)
        return torch.cat([self.img_proj(img_emb), self.mask_proj(mask_emb), self.text_proj(text_emb)], dim=-1)

    def decode_bucket_indices(
        self,
        out1: torch.Tensor | None,
        out2: torch.Tensor | None,
        *,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map raw head outputs to bucket indices for both targets."""
        if batch_size is None:
            if out1 is not None:
                batch_size = out1.size(0)
            elif out2 is not None:
                batch_size = out2.size(0)
            else:
                raise ValueError("batch_size is required when both head outputs are None")
        device = self.buckets1.device
        return decode_head_pair(
            self.head_type,
            out1,
            out2,
            buckets1=self.buckets1,
            buckets2=self.buckets2,
            predict_start=self.predict_start,
            predict_end=self.predict_end,
            batch_size=batch_size,
            device=device,
        )

    def bucket_scores(
        self,
        out1: torch.Tensor | None,
        out2: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Per-bucket additive scores for each head (log probs, except MSE)."""
        def _scores(out, buckets, active):
            if not active:
                return None
            if out is None:
                raise ValueError("head output required for an active head")
            if self.head_type == "CE":
                return bucket_scores_classification(out)
            if self.head_type == "CORAL":
                return bucket_scores_ordinal(out)
            if self.head_type == "MSE":
                return bucket_scores_regression(out, buckets)
            raise ValueError(f"head_type must be 'CORAL', 'MSE', or 'CE', got {self.head_type!r}")

        return _scores(out1, self.buckets1, self.predict_start), _scores(out2, self.buckets2, self.predict_end)

    def decode_cells(
        self,
        out1: torch.Tensor | None,
        out2: torch.Tensor | None,
        cell_start_idx: torch.Tensor,
        cell_end_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Pick one labeled (t_start, t_end) cell per sample.

        The two heads are decoded jointly rather than independently: each cell
        scores as the sum of its two bucket scores, and only the cells listed in
        cell_start_idx / cell_end_idx compete. Independent argmax can name a
        cell that does not exist in the grid (t_end >= t_start, or a bucket that
        never carries a label); restricting the argmax to real cells cannot.

        Returns: (N,) column indices into the cell list.
        """
        s1, s2 = self.bucket_scores(out1, out2)
        cell_scores = 0.0
        if s1 is not None:
            cell_scores = cell_scores + s1[:, cell_start_idx]
        if s2 is not None:
            cell_scores = cell_scores + s2[:, cell_end_idx]
        if not isinstance(cell_scores, torch.Tensor):
            raise ValueError("at least one head must be active to decode a cell")
        return cell_scores.argmax(dim=-1)

    """
    Forward and predict.
    """

    def forward(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Compute raw head outputs for targets with >1 bucket.
        Skipped heads return None.
        """
        features = self.body(self._get_context(img_emb, mask_emb, src_emb, tar_emb))
        out1 = self.head1(features) if self.predict_start else None
        out2 = self.head2(features) if self.predict_end else None
        return out1, out2

    def predict(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference and return bucket values (not indices).
        """
        training = self.training
        self.eval()
        with torch.no_grad():
            l1, l2 = self(img_emb, mask_emb, src_emb, tar_emb)
        self.train(training)

        idx1, idx2 = self.decode_bucket_indices(l1, l2, batch_size=img_emb.shape[0])
        return self.buckets1[idx1], self.buckets2[idx2]
