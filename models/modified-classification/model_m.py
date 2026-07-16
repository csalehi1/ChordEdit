"""
Metric surrogate M.

    M(img_emb, mask_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)

Frozen ChordEdit encoders produce embeddings; a trainable regressor with
separate PSNR and CLIP towers maps embeddings plus projected timesteps to
the two metric values. Image/mask/text bottlenecks balance the input;
timesteps use Fourier features; the CLIP tower is FiLM-conditioned on
timestep embeddings.
"""

from __future__ import annotations

import math
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

import torch
import torch.nn as nn
from PIL import Image
from pipeline_chord import ChordEditPipeline, DEFAULT_COMPUTE_DTYPE
from run_pie_bench import paths_from_model_root

from _helpers import combine_text_embeddings, mean_pool
from settings import *


def fourier_timestep_features(t: torch.Tensor, n_freqs: int = T_FOURIER_FREQS) -> torch.Tensor:
    """Encode (t_start, t_end) with raw values, their product, and sin/cos bands."""
    feats: list[torch.Tensor] = [t, (t[:, 0:1] * t[:, 1:2])]
    for k in range(n_freqs):
        freq = (2.0**k) * math.pi
        feats.append(torch.sin(freq * t))
        feats.append(torch.cos(freq * t))
    return torch.cat(feats, dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning vector."""

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feature_dim)
        self.to_beta = nn.Linear(cond_dim, feature_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.to_gamma(cond) * x + self.to_beta(cond)


class FiLMBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, dropout_rate: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.film = FiLM(out_dim, cond_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.film(self.norm(self.linear(x)), cond)
        h = torch.relu(h)
        return self.dropout(h)


class FiLMMLPBody(nn.Module):
    """MLP body where each hidden layer is modulated by timestep embeddings."""

    def __init__(
        self,
        in_features: int,
        cond_dim: int,
        n_wide: int,
        n_hidden: int,
        n_inner: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                FiLMBlock(in_features, n_wide, cond_dim, dropout_rate),
                FiLMBlock(n_wide, n_hidden, cond_dim, dropout_rate),
                FiLMBlock(n_hidden, n_inner, cond_dim, dropout_rate),
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return x


class TextEncoder(nn.Module):
    """ChordEdit text encoding with mean-pooled hidden states for the MLP."""

    def __init__(self, pipeline: ChordEditPipeline):
        super().__init__()
        self._pipeline = pipeline

    @property
    def hidden_dim(self) -> int:
        return self._pipeline.text_encoder.config.hidden_size

    @torch.no_grad()
    def forward(self, sentences: list[str]) -> torch.Tensor:
        hidden = self._pipeline._encode_text(sentences)
        tokens = self._pipeline.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self._pipeline.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return mean_pool(hidden, tokens.attention_mask.to(hidden.device))


class VaeImageEncoder(nn.Module):
    """ChordEdit VAE image encoding, flattened for the regressor MLP."""

    def __init__(self, pipeline: ChordEditPipeline):
        super().__init__()
        self._pipeline = pipeline
        with torch.no_grad():
            dummy = Image.new("RGB", (pipeline.image_size, pipeline.image_size))
            pixel_values = pipeline._prepare_image_tensor(dummy)
            latents = pipeline._encode_image_to_latent(pixel_values)
            self._hidden_dim = latents.flatten(start_dim=1).shape[-1]

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @torch.no_grad()
    def forward(self, images: list) -> torch.Tensor:
        # stack pixels and run one batched VAE encode instead of
        # encoding images one-at-a-time (much faster for large sample counts).
        if not images:
            raise ValueError("images must be a non-empty list")
        pixel_values = torch.cat(
            [self._pipeline._prepare_image_tensor(image.convert("RGB")) for image in images],
            dim=0,
        )
        encoded = self._pipeline._encode_image_to_latent(pixel_values)
        return encoded.flatten(start_dim=1)


class MetricRegressor(nn.Module):
    """PSNR MLP tower + FiLM-conditioned CLIP tower over bottlenecked embeddings."""

    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        n_targets: int = len(M_TARGET_COLS),
        img_proj_dim: int = IMG_PROJ_DIM,
        text_proj_dim: int = TEXT_PROJ_DIM,
        t_proj_dim: int = T_PROJ_DIM,
        t_fourier_freqs: int = T_FOURIER_FREQS,
        n_wide: int = MLP_WIDE,
        n_hidden: int = MLP_HIDDEN,
        n_inner: int = MLP_INNER,
        dropout_rate: float = MLP_DROPOUT,
        clip_dropout_rate: float = MLP_CLIP_DROPOUT,
    ):
        super().__init__()
        if n_targets != 2:
            raise ValueError("MetricRegressor expects exactly two targets (psnr, clip)")

        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, img_proj_dim),
            nn.LayerNorm(img_proj_dim),
            nn.ReLU(),
        )
        self.mask_proj = nn.Sequential(
            nn.Linear(img_dim, img_proj_dim),
            nn.LayerNorm(img_proj_dim),
            nn.ReLU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim * 4, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.ReLU(),
        )
        t_in = 3 + 4 * t_fourier_freqs
        self.t_encoder = nn.Sequential(
            nn.Linear(t_in, t_proj_dim * 2),
            nn.ReLU(),
            nn.Linear(t_proj_dim * 2, t_proj_dim),
            nn.ReLU(),
        )

        context_dim = img_proj_dim * 2 + text_proj_dim
        psnr_in = context_dim + t_proj_dim
        self.psnr_body = self._make_body(psnr_in, n_wide, n_hidden, n_inner, dropout_rate)
        self.psnr_head = nn.Linear(n_inner, 1)
        self.clip_body = FiLMMLPBody(
            context_dim, t_proj_dim, n_wide, n_hidden, n_inner, clip_dropout_rate
        )
        self.clip_head = nn.Linear(n_inner, 1)

        self.register_buffer("target_mean", torch.zeros(n_targets))
        self.register_buffer("target_std", torch.ones(n_targets))

    @staticmethod
    def _make_body(
        in_features: int,
        n_wide: int,
        n_hidden: int,
        n_inner: int,
        dropout_rate: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, n_wide),
            nn.LayerNorm(n_wide),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_wide, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden, n_inner),
            nn.ReLU(),
        )

    def _context_and_t(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = combine_text_embeddings(src_emb, tar_emb)
        context = torch.cat(
            [self.img_proj(img_emb), self.mask_proj(mask_emb), self.text_proj(text)],
            dim=-1,
        )
        t_feat = self.t_encoder(fourier_timestep_features(t))
        return context, t_feat

    def set_target_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.target_mean.copy_(mean.to(self.target_mean))
        self.target_std.copy_(std.to(self.target_std).clamp(min=1e-8))

    def forward(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return standardized metric predictions, shape (N, 2) — [psnr, clip]."""
        context, t_feat = self._context_and_t(img_emb, mask_emb, src_emb, tar_emb, t)
        psnr = self.psnr_head(self.psnr_body(torch.cat([context, t_feat], dim=-1)))
        clip = self.clip_head(self.clip_body(context, t_feat))
        return torch.cat([psnr, clip], dim=-1)

    def denormalize(self, standardized: torch.Tensor) -> torch.Tensor:
        """Map standardized predictions back to min-max normalized metric units."""
        return standardized * self.target_std + self.target_mean


class MetricPredictor(nn.Module):
    """Bundles ChordEdit encoders with the trainable metric regressor M."""

    def __init__(
        self,
        freeze_encoders: bool = FREEZE_ENCODERS,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.pipeline = ChordEditPipeline.from_local_sd_weights(
            paths_from_model_root(SD_TURBO_ROOT),
            image_size=IMAGE_SIZE,
            use_center_crop=USE_CENTER_CROP,
            compute_dtype=DEFAULT_COMPUTE_DTYPE,
            use_safety_checker=False,
            device=device,
        )
        if freeze_encoders:
            # Freeze the VAE and text encoder weights.
            for param in self.pipeline.vae.parameters():
                param.requires_grad = False
            for param in self.pipeline.text_encoder.parameters():
                param.requires_grad = False

        self.image_encoder = VaeImageEncoder(self.pipeline)
        self.text_encoder = TextEncoder(self.pipeline)
        self._encoder_img_dim = self.image_encoder.hidden_dim
        self._encoder_text_dim = self.text_encoder.hidden_dim
        self.regressor = MetricRegressor(
            img_dim=self._encoder_img_dim,
            text_dim=self._encoder_text_dim,
        )

    @property
    def encoder_img_dim(self) -> int:
        return self._encoder_img_dim

    @property
    def encoder_text_dim(self) -> int:
        return self._encoder_text_dim

    def release_encoders(self) -> None:
        """Free VAE/text pipeline after embeddings are precomputed."""
        # drop frozen SD encoders from GPU once embeddings exist so
        # training only keeps the small regressor on device.
        import gc

        if "image_encoder" in self._modules:
            del self.image_encoder
        if "text_encoder" in self._modules:
            del self.text_encoder
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(
        self, image, mask, src_prompt: str, tar_prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (img_emb, mask_emb, src_emb, tar_emb) each shape (1, D) on regressor device."""
        device = self.regressor.target_mean.device
        img_emb = self.image_encoder([image]).to(device)
        mask_emb = self.image_encoder([mask]).to(device)
        src_emb = self.text_encoder([src_prompt]).to(device)
        tar_emb = self.text_encoder([tar_prompt]).to(device)
        return img_emb, mask_emb, src_emb, tar_emb

    def predict_metrics_from_emb(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict (psnr, clip) in min-max normalized units from precomputed embeddings."""
        out = self.regressor(img_emb, mask_emb, src_emb, tar_emb, t)
        return self.regressor.denormalize(out)

    @torch.no_grad()
    def predict(
        self,
        images: list,
        masks: list,
        src_prompts: list[str],
        tar_prompts: list[str],
        t_start: list[float],
        t_end: list[float],
    ) -> torch.Tensor:
        """Predict (psnr, clip) in min-max normalized units for raw inputs."""
        training = self.training
        self.eval()
        device = self.regressor.target_mean.device
        img_emb = self.image_encoder(images).to(device)
        mask_emb = self.image_encoder(masks).to(device)
        src_emb = self.text_encoder(src_prompts).to(device)
        tar_emb = self.text_encoder(tar_prompts).to(device)
        # Combine the two timestep scalars into a single tensor.
        t = torch.tensor(list(zip(t_start, t_end)), dtype=torch.float, device=device)
        out = self.predict_metrics_from_emb(img_emb, mask_emb, src_emb, tar_emb, t)
        self.train(training)
        return out
