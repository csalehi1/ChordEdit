# model_m.py

"""
Surrogate model M^.

    Model architecture:
    M(img_emb, mask_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)
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

from settings import *


def combine_text_embeddings(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concat, difference, and Hadamard product of an embedding pair (4 * dim)."""
    return torch.cat([a, b, a - b, a * b], dim=-1)


# def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     """Mask-weighted mean over the token dimension."""
#     mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
#     return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


# def encode_text_pooled(pipeline: ChordEditPipeline, prompts: list[str]) -> torch.Tensor:
#     """
#     Collapse ChordEdit text-encoder outputs to (N, hidden_dim) for the MLP.
#
#     SD / SD-Turbo: mean-pool CLIP last_hidden_state over tokens (UNet still uses
#     the full sequence). SDXL / SDXL-Turbo: use CLIP's native pooled embeds from
#     text_encoder_2 (text_embeds), matching ChordEdit's SDXL conditioning.
#     """
#     device = pipeline._device
#     dtype = pipeline._compute_dtype
#
#     if getattr(pipeline, "_is_sdxl", False):
#         cond = pipeline._encode_sdxl_text(prompts)
#         if cond.pooled_embeds is None:
#             raise RuntimeError("SDXL text encoding did not produce pooled_embeds")
#         return cond.pooled_embeds.to(device=device, dtype=dtype)
#
#     inputs = pipeline.tokenizer(
#         list(prompts),
#         padding="max_length",
#         truncation=True,
#         max_length=pipeline.tokenizer.model_max_length,
#         return_tensors="pt",
#     )
#     input_ids = inputs.input_ids.to(device)
#     attn_mask = inputs.attention_mask.to(device)
#     encoder_mask = attn_mask if pipeline._use_attention_mask else None
#     outputs = pipeline.text_encoder(input_ids=input_ids, attention_mask=encoder_mask)
#     if hasattr(outputs, "last_hidden_state"):
#         hidden = outputs.last_hidden_state
#     else:
#         hidden = outputs[0]
#     return mean_pool(hidden, attn_mask).to(device=device, dtype=dtype)


def fourier_timestep_features(t: torch.Tensor, n_freqs: int = T_FOURIER_FREQS) -> torch.Tensor:
    """Encode (t_start, t_end) with raw values, their product, and sin/cos bands."""
    # The Fourier bands allow the model to capture both coarse and
    # fine-grained temporal relationships.
    feats: list[torch.Tensor] = [t, (t[:, 0:1] * t[:, 1:2])]
    for k in range(n_freqs):
        freq = (2.0**k) * math.pi
        feats.append(torch.sin(freq * t))
        feats.append(torch.cos(freq * t))
    return torch.cat(feats, dim=-1)


def pairwise_ranking_loss(pred: torch.Tensor, true: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Logistic pairwise loss: penalize pred ordering that disagrees with true."""
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
ChordEdit encoders.
"""

# class TextEncoder(nn.Module):
#     """ChordEdit text encoding collapsed to a fixed vector for the MLP."""

#     def __init__(self, pipeline: ChordEditPipeline):
#         super().__init__()
#         self._pipeline = pipeline
#         with torch.no_grad():
#             # Probe output width (SD mean-pool hidden vs SDXL pooled embeds).
#             self._hidden_dim = int(encode_text_pooled(pipeline, [""]).shape[-1])

#     @property
#     def hidden_dim(self) -> int:
#         return self._hidden_dim

#     @torch.no_grad()
#     def forward(self, sentences: list[str]) -> torch.Tensor:
#         # Reuse ChordEdit's tokenizer and text encoder(s), then collapse to a
#         # fixed vector. UNet conditioning uses the full token sequence; the MLP
#         # needs one embedding per prompt (see encode_text_pooled above).
#         return encode_text_pooled(self._pipeline, sentences)


# class VaeImageEncoder(nn.Module):
#     """ChordEdit VAE image encoding, flattened for the regressor MLP."""

#     def __init__(self, pipeline: ChordEditPipeline):
#         super().__init__()
#         self._pipeline = pipeline
#         with torch.no_grad():
#             # Probe latent width via ChordEdit's own preprocess and VAE encode so
#             # img_dim matches whatever image_size / center-crop the pipeline uses.
#             dummy = Image.new("RGB", (pipeline.image_size, pipeline.image_size))
#             pixel_values = pipeline._prepare_image_tensor(dummy)
#             latents = pipeline._encode_image_to_latent(pixel_values)
#             self._hidden_dim = latents.flatten(start_dim=1).shape[-1]

#     @property
#     def hidden_dim(self) -> int:
#         return self._hidden_dim

#     @torch.no_grad()
#     def forward(self, images: list) -> torch.Tensor:
#         # ChordEdit's _prepare_image_tensor / _encode_image_to_latent keep the
#         # same crop, normalize, and VAE scaling as real edits. Batch the VAE
#         # forward instead of encoding one image at a time.
#         if not images:
#             raise ValueError("images must be a non-empty list")
#         pixel_values = torch.cat(
#             [self._pipeline._prepare_image_tensor(image.convert("RGB")) for image in images],
#             dim=0,
#         )
#         encoded = self._pipeline._encode_image_to_latent(pixel_values)
#         return encoded.flatten(start_dim=1)


class ConvImageProjector(nn.Module):
    """Encode a flattened VAE latent with convolutions instead of one Linear.

    The image and mask embeddings are flattened (C, S, S) VAE latents, so a
    single Linear over 16k inputs discards all spatial structure - including
    how large and where the edit mask is, which is most of what decides
    PSNR-Unedited. This folds the latent back to (C, S, S) and downsamples.
    """

    def __init__(self, flat_dim: int, out_dim: int, channels: int = 4):
        super().__init__()
        spatial_sq = flat_dim // channels
        side = int(round(spatial_sq ** 0.5))
        if channels * side * side != flat_dim:
            raise ValueError(f"{flat_dim=} is not {channels}xSxS for an integer S")
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
CLIP-Edited.
"""

class FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning vector."""

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        # The condition initially has no effect on the feature: forward
        # returns 1*x+0=x. As training progresses, gradients flowing
        # into to_gamma.weight and to_beta.weight gradually teach it how
        # to use the condition. This is for stability.
        self.to_gamma = nn.Linear(cond_dim, feature_dim)
        self.to_beta = nn.Linear(cond_dim, feature_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias) # \gamma starts as all 1s
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias) # \beta starts as all 0s

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Apply the FiLM modulation to the input.
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
        # Three FiLM blocks with ReLU activations and dropout serve as
        # the overall MLP for CLIP-Edited.
        self.blocks = nn.ModuleList(
            [
                FiLMBlock(in_features, n_wide, cond_dim, dropout_rate),
                FiLMBlock(n_wide, n_hidden, cond_dim, dropout_rate),
                FiLMBlock(n_hidden, n_inner, cond_dim, dropout_rate),
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Iterate through the blocks, applying the FiLM modulation to the input.
        for block in self.blocks:
            x = block(x, cond)
        return x


"""
PSNR-Unedited
"""

class MLPBlock(nn.Module):
    """Plain MLP layer (no FiLM): Linear -> LayerNorm -> ReLU -> Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm(self.linear(x)))
        return self.dropout(h)


class MLPBody(nn.Module):
    """MLP body for PSNR-Unedited (timesteps concatenated into the input, not FiLM'd)."""

    def __init__(
        self,
        in_features: int,
        n_wide: int,
        n_hidden: int,
        n_inner: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MLPBlock(in_features, n_wide, dropout_rate),
                MLPBlock(n_wide, n_hidden, dropout_rate),
                MLPBlock(n_hidden, n_inner, dropout_rate),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


"""
Shared metric predictor components.
"""

class SurrogateRegressor(nn.Module):
    """
    PSNR-Unedited MLP tower and CLIP-Edited FiLM-conditioned MLP
    tower over bottlenecked embeddings and timestep conditioning.
    """

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
        psnr_dropout_rate: float = MLP_PSNR_DROPOUT,
        clip_dropout_rate: float = MLP_CLIP_DROPOUT,
    ):
        super().__init__()
        if n_targets != 2:
            raise ValueError("SurrogateRegressor expects exactly two targets (PSNR, CLIP)")

        # Project the embeddings to the MLP input dimension.
        def image_projection() -> nn.Module:
            if str(IMG_ENCODER) == "conv":
                return ConvImageProjector(img_dim, img_proj_dim)
            elif str(IMG_ENCODER) == "linear":
                return nn.Sequential(
                    nn.Linear(img_dim, img_proj_dim),
                    nn.LayerNorm(img_proj_dim),
                    nn.ReLU(),
                )
            else:
                raise ValueError(f"Unsupported {IMG_ENCODER=}")

        self.img_proj = image_projection()
        self.mask_proj = image_projection()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim * 4, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.ReLU(),
        )

        # Timestep encoder takes Fourier features instead of raw t.
        t_in, t_out = 3 + 4 * t_fourier_freqs, t_proj_dim * 2
        self.t_encoder = nn.Sequential(
            nn.Linear(t_in, t_out),
            nn.ReLU(),
            nn.Linear(t_out, t_proj_dim),
            nn.ReLU(),
        )

        # PSNR-Unedited MLP body and head.
        psnr_in = img_proj_dim * 2 + text_proj_dim + t_proj_dim
        self.psnr_body = MLPBody(psnr_in, n_wide, n_hidden, n_inner, psnr_dropout_rate)
        self.psnr_head = nn.Linear(n_inner, 1)

        # CLIP-Edited FiLM-conditioned MLP body and head.
        clip_in = img_proj_dim * 2 + text_proj_dim
        self.clip_body = FiLMMLPBody(clip_in, t_proj_dim, n_wide, n_hidden, n_inner, clip_dropout_rate)
        self.clip_head = nn.Linear(n_inner, 1)

        self.register_buffer("target_mean", torch.zeros(n_targets))
        self.register_buffer("target_std", torch.ones(n_targets))

    def denormalize(self, standardized: torch.Tensor) -> torch.Tensor:
        """Map standardized predictions back to target units."""
        return standardized * self.target_std + self.target_mean

    def _context(
        self,
        img_emb: torch.Tensor,  # (N, D_img)
        mask_emb: torch.Tensor, # (N, D_img)
        src_emb: torch.Tensor,  # (N, D_txt)
        tar_emb: torch.Tensor,  # (N, D_txt)
    ) -> torch.Tensor:          # (N, D_ctx)
        """Timestep-independent part of the input: one vector per sample."""
        text_emb = combine_text_embeddings(src_emb, tar_emb)
        return torch.cat([self.img_proj(img_emb), self.mask_proj(mask_emb), self.text_proj(text_emb)], dim=-1)

    def _get_inputs(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        t: torch.Tensor,        # (N, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # context (N, D_ctx), t_feat (N, D_t)
        context = self._context(img_emb, mask_emb, src_emb, tar_emb)
        # Encode the timestep into a feature vector.
        t_feat = self.t_encoder(fourier_timestep_features(t))
        return context, t_feat

    def set_target_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the mean and standard deviation of the targets."""
        self.target_mean.copy_(mean.to(self.target_mean))
        self.target_std.copy_(std.to(self.target_std).clamp(min=1e-8))

    def forward(
        self,
        img_emb: torch.Tensor,  # (N, D_img)
        mask_emb: torch.Tensor, # (N, D_img)
        src_emb: torch.Tensor,  # (N, D_txt)
        tar_emb: torch.Tensor,  # (N, D_txt)
        t: torch.Tensor,        # (N, 2)
    ) -> torch.Tensor:          # (N, 2)
        """Return standardized metric predictions."""
        context, t_feat = self._get_inputs(img_emb, mask_emb, src_emb, tar_emb, t)
        # Predcit PSNR and CLIP from the context and timestep feature.
        psnr = self.psnr_head(self.psnr_body(torch.cat([context, t_feat], dim=-1)))
        clip = self.clip_head(self.clip_body(context, t_feat))
        return torch.cat([psnr, clip], dim=-1)

    def forward_grid(
        self,
        img_emb: torch.Tensor,  # (G, D_img)
        mask_emb: torch.Tensor, # (G, D_img)
        src_emb: torch.Tensor,  # (G, D_txt)
        tar_emb: torch.Tensor,  # (G, D_txt)
        t: torch.Tensor,        # (G, C, 2)
    ) -> torch.Tensor:          # (G, C, 2)
        """Return standardized metric predictions for a sample's timestep grid."""
        g, c = int(t.shape[0]), int(t.shape[1])
        t_flat = t.reshape(g * c, 2)
        context = self._context(img_emb, mask_emb, src_emb, tar_emb)
        context = context.unsqueeze(1).expand(-1, c, -1).reshape(g * c, -1)
        t_feat = self.t_encoder(fourier_timestep_features(t_flat))
        psnr = self.psnr_head(self.psnr_body(torch.cat([context, t_feat], dim=-1)))
        clip = self.clip_head(self.clip_body(context, t_feat))
        return torch.cat([psnr, clip], dim=-1).reshape(g, c, -1)


class SurrogateModel(nn.Module):
    """Trainable SurrogateRegressor sized from precomputed embedding dims."""

    def __init__(
        self,
        img_dim: int,
        text_dim: int,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self._encoder_img_dim = int(img_dim)
        self._encoder_text_dim = int(text_dim)
        self.regressor = SurrogateRegressor(
            self._encoder_img_dim,
            self._encoder_text_dim,
            n_targets=len(M_TARGET_COLS),
        )
        if device is not None:
            self.to(device)

    @property
    def encoder_img_dim(self) -> int:
        return self._encoder_img_dim

    @property
    def encoder_text_dim(self) -> int:
        return self._encoder_text_dim

    # def release_encoders(self) -> None:
    #     """Free VAE/text pipeline after embeddings are precomputed."""
    #     import gc
    #     if "image_encoder" in self._modules:
    #         del self.image_encoder
    #     if "text_encoder" in self._modules:
    #         del self.text_encoder
    #     self.pipeline = None
    #     gc.collect()
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    # @torch.no_grad()
    # def encode(
    #     self,
    #     images: list,
    #     masks: list,
    #     src_prompts: list[str],
    #     tar_prompts: list[str],
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Return (img_emb, mask_emb, src_emb, tar_emb) each shape (N, D) on regressor device."""
    #     device = self.regressor.target_mean.device
    #     return (
    #         self.image_encoder(images).to(device),
    #         self.image_encoder(masks).to(device),
    #         self.text_encoder(src_prompts).to(device),
    #         self.text_encoder(tar_prompts).to(device),
    #     )

    def predict_emb(
        self,
        img_emb: torch.Tensor,  # (N, D_img)
        mask_emb: torch.Tensor, # (N, D_img)
        src_emb: torch.Tensor,  # (N, D_txt)
        tar_emb: torch.Tensor,  # (N, D_txt)
        t: torch.Tensor,        # (N, 2)
    ) -> torch.Tensor:          # (N, 2)
        """Predict (psnr, clip) in raw metric units from precomputed embeddings."""
        out = self.regressor(img_emb, mask_emb, src_emb, tar_emb, t)
        return self.regressor.denormalize(out)

    # @torch.no_grad()
    # def predict_raw(
    #     self,
    #     images: list,
    #     masks: list,
    #     src_prompts: list[str],
    #     tar_prompts: list[str],
    #     t_start: list[float],
    #     t_end: list[float],
    # ) -> torch.Tensor:
    #     """Predict (psnr, clip) in raw metric units for raw inputs."""
    #     was_training = self.training
    #     self.eval()
    #     img_emb, mask_emb, src_emb, tar_emb = self.encode(images, masks, src_prompts, tar_prompts)
    #     t = torch.tensor(list(zip(t_start, t_end)), dtype=torch.float, device=img_emb.device)
    #     out = self.predict_emb(img_emb, mask_emb, src_emb, tar_emb, t)
    #     self.train(was_training)
    #     return out # shape (N, 2)
