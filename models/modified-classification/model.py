"""
Metric-predictor model.

    M(img_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)

The image and text encoders are frozen and only produce embeddings; the
trainable component is a small MLP regressor that maps those embeddings plus
the two timestep scalars to the two metric values.

    1.  TextEncoder delegates to ChordEditPipeline.encode_prompt, then
        mean-pools token hidden states for the regressor MLP.
    2.  VaeImageEncoder delegates to ChordEditPipeline._prepare_image_tensor
        and _encode_image_to_latent, then flattens latents for the MLP.
    3.  MetricRegressor uses two towers: PSNR from `(t_start, t_end)` only,
        CLIP from image + prompt embeddings (variance structure in the data).
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
from PIL import Image
from pipeline_chord import ChordEditPipeline, DEFAULT_COMPUTE_DTYPE
from run_pie_bench import paths_from_model_root

from models.classification.model import OrdinalPairClassifier, SiameseEncoder
from settings import *


class TextEncoder(nn.Module):
    """
    ChordEdit text encoding with mean-pooled hidden states for the MLP.

    Uses `ChordEditPipeline._encode_text` for the hidden states. Mean-pooling
    (via `SiameseEncoder._mean_pool`) is our addition: ChordEdit keeps the full
    sequence for UNet conditioning.
    """

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
        return SiameseEncoder._mean_pool(hidden, tokens.attention_mask.to(hidden.device))


class VaeImageEncoder(nn.Module):
    """
    ChordEdit VAE image encoding, flattened for the regressor MLP.

    Delegates preprocessing and encoding to ChordEditPipeline. Flattening is
    our addition: ChordEdit keeps (B, 4, H/8, W/8) for UNet editing.
    """

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
        if not images:
            raise ValueError("images must be a non-empty list")
        latents = []
        for image in images:
            pixel_values = self._pipeline._prepare_image_tensor(image.convert("RGB"))
            encoded = self._pipeline._encode_image_to_latent(pixel_values)
            latents.append(encoded.flatten(start_dim=1))
        return torch.cat(latents, dim=0)


class MetricRegressor(nn.Module):
    """MLP that maps precomputed embeddings + timesteps to metric values.

    PSNR and CLIP use separate towers: ~94% of PSNR variance is within the
    (t_start, t_end) grid (embeddings are constant per sample), while ~76% of
    CLIP variance is between samples (image + prompts). A shared MLP lets the
    16k-dim latent drown out the two timestep scalars, so PSNR never learns.
    """

    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        n_targets: int = len(TARGET_COLS),
        n_wide: int = MLP_WIDE,
        n_hidden: int = MLP_HIDDEN,
        n_inner: int = MLP_INNER,
        dropout_rate: float = MLP_DROPOUT,
    ):
        super().__init__()
        if n_targets != 2:
            raise ValueError("split towers expect exactly two targets (psnr, clip)")

        clip_in = img_dim + text_dim * 4
        self.clip_body = self._make_body(clip_in, n_wide, n_hidden, n_inner, dropout_rate)
        self.clip_head = nn.Linear(n_inner, 1)
        self.psnr_body = self._make_body(2, n_wide, n_hidden, n_inner, dropout_rate)
        self.psnr_head = nn.Linear(n_inner, 1)

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

    def set_target_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.target_mean.copy_(mean.to(self.target_mean))
        self.target_std.copy_(std.to(self.target_std).clamp(min=1e-8))

    def forward(
        self,
        img_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return standardized metric predictions, shape (N, 2) — [psnr, clip]."""
        text = OrdinalPairClassifier._combine(src_emb, tar_emb)
        psnr = self.psnr_head(self.psnr_body(t))
        clip = self.clip_head(self.clip_body(torch.cat([img_emb, text], dim=-1)))
        return torch.cat([psnr, clip], dim=-1)

    def denormalize(self, standardized: torch.Tensor) -> torch.Tensor:
        """Map standardized predictions back to raw metric units."""
        return standardized * self.target_std + self.target_mean


class MetricPredictor(nn.Module):
    """Bundles ChordEdit encoders with the trainable regressor for inference."""

    def __init__(
        self,
        freeze_encoders: bool = True,
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
        # Create the regressor MLP with encoder hidden dimensions.
        self.regressor = MetricRegressor(
            img_dim=self.image_encoder.hidden_dim,
            text_dim=self.text_encoder.hidden_dim,
        )

    @torch.no_grad()
    def predict(
        self,
        images: list,
        src_prompts: list[str],
        tar_prompts: list[str],
        t_start: list[float],
        t_end: list[float],
    ) -> torch.Tensor:
        """Predict (psnr, clip) in raw units for raw inputs."""
        training = self.training
        self.eval()
        device = self.regressor.target_mean.device
        img_emb = self.image_encoder(images).to(device)
        src_emb = self.text_encoder(src_prompts).to(device)
        tar_emb = self.text_encoder(tar_prompts).to(device)
        # Combine the two timestep scalars into a single tensor.
        t = torch.tensor(list(zip(t_start, t_end)), dtype=torch.float, device=device)
        out = self.regressor(img_emb, src_emb, tar_emb, t)
        self.train(training)
        # De-standardize the predictions back to raw metric units.
        return self.regressor.denormalize(out)
