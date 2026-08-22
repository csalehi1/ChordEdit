# model_m.py

"""
Surrogate model M^.

    Model architecture:
    M(img_emb, src_emb, tar_emb) -> (n_cells, 2) grid of (psnr, clip)
"""

from __future__ import annotations

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

import torch
import torch.nn as nn

from settings import *


def combine_text_embs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concat, difference, and Hadamard product of an embedding pair (4 * dim)."""
    return torch.cat([a, b, a - b, a * b], dim=-1)


def pairwise_ranking_loss(pred: torch.Tensor, true: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Logistic pairwise loss to penalize predictions that disagree with the ground truth."""
    if pred.shape[-1] < 2:
        return pred.new_zeros(())
    diff_true = true.unsqueeze(-1) - true.unsqueeze(-2)
    diff_pred = pred.unsqueeze(-1) - pred.unsqueeze(-2)
    mask = diff_true > 0
    if top_k and top_k < true.shape[-1]:
        idx = true.topk(top_k, dim=-1).indices
        is_top = torch.zeros_like(true, dtype=torch.bool).scatter_(-1, idx, True)
        mask = mask & is_top.unsqueeze(-1)
    if not mask.any():
        return pred.new_zeros(())
    return torch.nn.functional.softplus(-diff_pred[mask]).mean()


"""
Projectors.
"""

class LinearImageProjector(nn.Sequential):
    """Project a flattened VAE latent with a single Linear + LayerNorm + ReLU."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__(
            nn.Linear(n_in, n_out),
            nn.LayerNorm(n_out),
            nn.ReLU(),
        )


class ConvImageProjector(nn.Module):
    """Encode a flattened VAE latent with convolutions instead of one Linear."""

    def __init__(self, n_in: int, n_out: int, channels: int = 4):
        super().__init__()
        spatial_sq = n_in // channels
        side = int(round(spatial_sq ** 0.5))
        if channels * side * side != n_in:
            raise ValueError(f"{n_in=} is not {channels}xSxS for an integer S")
        self.channels, self.side = channels, side

        widths = [channels, 32, 64, 128, 128]
        layers: list[nn.Module] = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Conv2d(a, b, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, b), nn.SiLU()]
        self.stem = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1] * (side // 2 ** (len(widths) - 1)) ** 2, n_out),
            nn.LayerNorm(n_out),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.channels, self.side, self.side)
        return self.head(self.stem(x))


class LinearTextProjector(nn.Sequential):
    """Project combined source/target text features with a single Linear, LayerNorm, and ReLU."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__(
            nn.Linear(n_in, n_out),
            nn.LayerNorm(n_out),
            nn.ReLU(),
        )


"""
MLP.
"""

class MLPBlock(nn.Module):

    def __init__(self, n_in: int, n_out: int, dropout_rate: float):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.norm = nn.LayerNorm(n_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm(self.linear(x)))
        return self.dropout(h)


class MLPBody(nn.Module):

    def __init__(self, n_in: int, n_wide: int, n_hidden: int, n_out: int, dropout_rate: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLPBlock(n_in, n_wide, dropout_rate),
            MLPBlock(n_wide, n_hidden, dropout_rate),
            MLPBlock(n_hidden, n_out, dropout_rate),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


"""
Surrogate Model.
"""

class SurrogateRegressor(nn.Module):
    """PSNR-Unedited and CLIP-Edited MLP towers, each predicting all grid cells at once."""

    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        n_cells: int,
        n_targets: int = len(M_TARGET_COLS),
        img_proj_dim: int = IMG_PROJ_DIM,
        text_proj_dim: int = TEXT_PROJ_DIM,
        n_wide: int = MLP_WIDE,
        n_hidden: int = MLP_HIDDEN,
        n_inner: int = MLP_INNER,
        psnr_dropout_rate: float = MLP_PSNR_DROPOUT,
        clip_dropout_rate: float = MLP_CLIP_DROPOUT,
    ):
        super().__init__()
        if n_targets != 2:
            raise ValueError("SurrogateRegressor expects exactly two targets (PSNR, CLIP)")

        def image_projector(n_in: int, allow_conv: bool = True) -> nn.Module:
            if str(IMG_ENCODER) == "conv":
                if not allow_conv:
                    raise ValueError(f"{IMG_ENCODER=} applies to VAE latents only, not CLIP embeddings")
                return ConvImageProjector(n_in, img_proj_dim)
            if str(IMG_ENCODER) == "linear":
                return LinearImageProjector(n_in, img_proj_dim)
            raise ValueError(f"Unknown {IMG_ENCODER=}")

        # Project the embeddings to the MLP input dimension. Under "vae+clip"
        # the image input arrives as one concatenated tensor (embeddings.py) and
        # each half gets its own projector.
        self.img_emb_source = str(IMG_EMB_SOURCE)
        if self.img_emb_source == "vae+clip":
            from clip_image import CLIP_IMG_DIM

            self.vae_dim = img_dim - CLIP_IMG_DIM
            if self.vae_dim <= 0:
                raise ValueError(f"{img_dim=} is too small to hold a {CLIP_IMG_DIM}-d CLIP embedding")
            self.img_proj = image_projector(self.vae_dim)
            self.clip_img_proj = image_projector(CLIP_IMG_DIM, allow_conv=False)
        else:
            self.vae_dim = img_dim
            self.img_proj = image_projector(img_dim, allow_conv=self.img_emb_source == "vae")

        self.text_proj = LinearTextProjector(text_dim * 4, text_proj_dim)

        # Build the PSNR and CLIP towers.
        n_img_arms = 2 if self.img_emb_source == "vae+clip" else 1
        n_features = n_img_arms * img_proj_dim + text_proj_dim
        self.psnr_body = MLPBody(n_features, n_wide, n_hidden, n_inner, psnr_dropout_rate)
        self.psnr_head = nn.Linear(n_inner, n_cells)
        self.clip_body = MLPBody(n_features, n_wide, n_hidden, n_inner, clip_dropout_rate)
        self.clip_head = nn.Linear(n_inner, n_cells)

        # Register the target mean and std for denormalization.
        self.register_buffer("target_mean", torch.zeros(n_targets))
        self.register_buffer("target_std", torch.ones(n_targets))

    def destandardize(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized * self.target_std + self.target_mean

    def set_target_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.target_mean.copy_(mean.to(self.target_mean))
        self.target_std.copy_(std.to(self.target_std).clamp(min=1e-8))

    def forward(
        self,
        img_emb: torch.Tensor,  # (N, D_img)
        src_emb: torch.Tensor,  # (N, D_txt)
        tar_emb: torch.Tensor,  # (N, D_txt)
    ) -> torch.Tensor:          # (N, n_cells, 2)
        """Return standardized per-cell metric predictions."""
        text_emb = combine_text_embs(src_emb, tar_emb)
        if self.img_emb_source == "vae+clip":
            img_parts = [
                self.img_proj(img_emb[..., : self.vae_dim]),
                self.clip_img_proj(img_emb[..., self.vae_dim :]),
            ]
        else:
            img_parts = [self.img_proj(img_emb)]
        x = torch.cat([*img_parts, self.text_proj(text_emb)], dim=-1)
        psnr = self.psnr_head(self.psnr_body(x))
        clip = self.clip_head(self.clip_body(x))
        return torch.stack([psnr, clip], dim=-1)


class SurrogateModel(nn.Module):
    """Trainable SurrogateRegressor sized from precomputed embedding dims."""

    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        n_cells: int,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.regressor = SurrogateRegressor(
            img_dim=img_dim,
            text_dim=text_dim,
            n_cells=n_cells,
            n_targets=len(M_TARGET_COLS),
        )
        if device is not None:
            self.to(device)

    def pred_emb(
        self,
        img_emb: torch.Tensor,  # (N, D_img)
        src_emb: torch.Tensor,  # (N, D_txt)
        tar_emb: torch.Tensor,  # (N, D_txt)
    ) -> torch.Tensor:          # (N, n_cells, 2)
        """Predict per-cell (psnr, clip) in raw metric units."""
        out = self.regressor(img_emb, src_emb, tar_emb)
        return self.regressor.destandardize(out)
