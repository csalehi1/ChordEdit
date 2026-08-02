"""
Surrogate model M_hat.

Paper: M_hat(x_src, c_src, c_tar, t*, t**) -> s = (s_1, s_2), predicting
s_1 = PSNR-Unedited and s_2 = CLIP-Edited. Code's t_start/t_end are the
paper's (t*, t**) on the quantized N x N grid T (N=11); mask is the edit
mask m_obj.

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

# ChordEdit provides the same VAE/text stack used at edit time, so M's
# embeddings stay consistent with the metrics collected from ChordEdit grids.
# paths_from_model_root resolves component dirs under CHORD_EDIT_MODEL_ROOT.
from pipeline_chord import ChordEditPipeline, DEFAULT_COMPUTE_DTYPE
from run_pie_bench import paths_from_model_root

from settings import *


def combine_text_embeddings(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concat, difference, and Hadamard product of an embedding pair (4 * dim)."""
    return torch.cat([a, b, a - b, a * b], dim=-1)


def encode_text_pooled(pipeline: ChordEditPipeline, prompts: list[str]) -> torch.Tensor:
    """
    Collapse ChordEdit text-encoder outputs to (N, hidden_dim) for the MLP.

    SD / SD-Turbo: mean-pool CLIP last_hidden_state over tokens (UNet still uses
    the full sequence). SDXL / SDXL-Turbo: use CLIP's native pooled embeds from
    text_encoder_2 (text_embeds), matching ChordEdit's SDXL conditioning.
    """
    device = pipeline._device
    dtype = pipeline._compute_dtype

    if getattr(pipeline, "_is_sdxl", False):
        cond = pipeline._encode_sdxl_text(prompts)
        if cond.pooled_embeds is None:
            raise RuntimeError("SDXL text encoding did not produce pooled_embeds")
        return cond.pooled_embeds.to(device=device, dtype=dtype)

    inputs = pipeline.tokenizer(
        list(prompts),
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)
    encoder_mask = attn_mask if pipeline._use_attention_mask else None
    outputs = pipeline.text_encoder(input_ids=input_ids, attention_mask=encoder_mask)
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    else:
        hidden = outputs[0]
    return mean_pool(hidden, attn_mask).to(device=device, dtype=dtype)


def format_results(m: dict[str, float]) -> str:
    """Fixed-width metric line so Train/Val columns stay aligned."""
    return f"loss={m['loss']:7.4f}  " + "  ".join(
        f"{col}: MAE={m[f'mae_{col}']:6.3f} R2={m[f'r2_{col}']:7.3f}"
        for col in M_TARGET_COLS
    )


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


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted mean over the token dimension."""
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


"""
Per-model text extraction for packing scattered embeddings.

Scattered text files store whatever the annotation pipeline emitted; collapsing
them to one vector per prompt must match encode_text_pooled for the current
CHORD_EDIT_MODEL so packed caches and on-the-fly encoding agree per model type:
sd stores full last_hidden_state sequences (pooled here with mean_pool), sdxl
must store text_encoder_2's pooled embeds directly (text_embeds is a projection
of the EOS token and cannot be reconstructed from hidden-state sequences),
flux is not implemented.
"""

TEXT_POOLING_BY_PIPELINE = {"sd": "masked_mean", "sdxl": "pooled_embeds"}


def _get_text_pooling() -> str:
    """Text pooling name for CHORD_EDIT_PIPELINE_TYPE; raises for unsupported types."""
    if CHORD_EDIT_PIPELINE_TYPE == "flux":
        raise NotImplementedError(
            "CHORD_EDIT_MODEL='flux' text pooling is not implemented; "
            "use sd_turbo / sdxl_turbo."
        )
    if CHORD_EDIT_PIPELINE_TYPE not in TEXT_POOLING_BY_PIPELINE:
        raise ValueError(f"Unsupported CHORD_EDIT_PIPELINE_TYPE={CHORD_EDIT_PIPELINE_TYPE!r}")
    return TEXT_POOLING_BY_PIPELINE[CHORD_EDIT_PIPELINE_TYPE]


class SdMaskedMeanTextExtractor:
    """SD branch: mask-weighted mean over the stored last_hidden_state sequence.

    Attention masks come from re-tokenizing the prompts (tokenizer only, no
    encoder weights), since the scattered files do not store them. Reuses
    mean_pool so this cannot drift from encode_text_pooled.
    """

    name = "masked_mean"

    def __init__(self, src_prompts: list[str], tar_prompts: list[str]):
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(str(CHORD_EDIT_MODEL_ROOT / "tokenizer"))
        self.seq_len = int(tokenizer.model_max_length)

        def _attn(prompts: list[str]) -> torch.Tensor:
            enc = tokenizer(
                list(prompts),
                padding="max_length",
                truncation=True,
                max_length=self.seq_len,
                return_tensors="pt",
            )
            return enc.attention_mask

        self._attn = {"src": _attn(src_prompts), "tar": _attn(tar_prompts)}

    def text_dim(self, probe: torch.Tensor, path) -> int:
        """Validate a stored text tensor's shape and return the hidden dim."""
        if probe.ndim != 3 or probe.shape[0] != 1:
            raise ValueError(f"Expected text sequence (1, T, D), got {tuple(probe.shape)} in {path}")
        if int(probe.shape[1]) != self.seq_len:
            raise ValueError(
                f"Stored sequence length {int(probe.shape[1])} != tokenizer max length "
                f"{self.seq_len} in {path}; scattered embeddings do not match this model's tokenizer"
            )
        return int(probe.shape[2])

    def __call__(self, t: torch.Tensor, i: int, kind: str, dim: int) -> torch.Tensor:
        """Collapse sample i's stored sequence for kind in {'src', 'tar'} to (dim,)."""
        hidden = t.reshape(1, self.seq_len, dim)
        return mean_pool(hidden, self._attn[kind][i].unsqueeze(0))[0]


class SdxlPooledTextExtractor:
    """SDXL branch: scattered files must already store text_encoder_2's pooled
    embeds (1, D); pooling cannot be redone from sequences without weights."""

    name = "pooled_embeds"

    def __init__(self, src_prompts: list[str], tar_prompts: list[str]):
        pass

    def text_dim(self, probe: torch.Tensor, path) -> int:
        if probe.numel() != probe.shape[-1]:
            raise ValueError(
                f"Expected pooled text embeds (1, D) or (D,), got {tuple(probe.shape)} in {path}. "
                f"SDXL pooled embeds (text_encoder_2 text_embeds) cannot be reconstructed from "
                f"hidden-state sequences; re-run the annotation pipeline storing pooled embeds."
            )
        return int(probe.shape[-1])

    def __call__(self, t: torch.Tensor, i: int, kind: str, dim: int) -> torch.Tensor:
        if t.numel() != dim:
            raise ValueError(f"Pooled text embeds numel {t.numel()} != {dim}")
        return t.reshape(-1)


def make_text_extractor(src_prompts: list[str], tar_prompts: list[str]):
    """Text extractor matching CHORD_EDIT_PIPELINE_TYPE (see expected_text_pooling)."""
    pooling = _get_text_pooling()
    extractor_cls = {
        SdMaskedMeanTextExtractor.name: SdMaskedMeanTextExtractor,
        SdxlPooledTextExtractor.name: SdxlPooledTextExtractor,
    }[pooling]
    return extractor_cls(src_prompts, tar_prompts)


def pairwise_ranking_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Logistic pairwise loss: penalize pred ordering that disagrees with true."""
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    diff_true = true.unsqueeze(1) - true.unsqueeze(0)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    mask = diff_true > 0
    if not mask.any():
        return pred.new_zeros(())
    return torch.nn.functional.softplus(-diff_pred[mask]).mean()


"""
ChordEdit encoders.
"""

class TextEncoder(nn.Module):
    """ChordEdit text encoding collapsed to a fixed vector for the MLP."""

    def __init__(self, pipeline: ChordEditPipeline):
        super().__init__()
        self._pipeline = pipeline
        with torch.no_grad():
            # Probe output width (SD mean-pool hidden vs SDXL pooled embeds).
            self._hidden_dim = int(encode_text_pooled(pipeline, [""]).shape[-1])

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @torch.no_grad()
    def forward(self, sentences: list[str]) -> torch.Tensor:
        # Reuse ChordEdit's tokenizer and text encoder(s), then collapse to a
        # fixed vector. UNet conditioning uses the full token sequence; the MLP
        # needs one embedding per prompt (see encode_text_pooled above).
        return encode_text_pooled(self._pipeline, sentences)


class VaeImageEncoder(nn.Module):
    """ChordEdit VAE image encoding, flattened for the regressor MLP."""

    def __init__(self, pipeline: ChordEditPipeline):
        super().__init__()
        self._pipeline = pipeline
        with torch.no_grad():
            # Probe latent width via ChordEdit's own preprocess and VAE encode so
            # img_dim matches whatever image_size / center-crop the pipeline uses.
            dummy = Image.new("RGB", (pipeline.image_size, pipeline.image_size))
            pixel_values = pipeline._prepare_image_tensor(dummy)
            latents = pipeline._encode_image_to_latent(pixel_values)
            self._hidden_dim = latents.flatten(start_dim=1).shape[-1]

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @torch.no_grad()
    def forward(self, images: list) -> torch.Tensor:
        # ChordEdit's _prepare_image_tensor / _encode_image_to_latent keep the
        # same crop, normalize, and VAE scaling as real edits. Batch the VAE
        # forward instead of encoding one image at a time.
        if not images:
            raise ValueError("images must be a non-empty list")
        pixel_values = torch.cat(
            [self._pipeline._prepare_image_tensor(image.convert("RGB")) for image in images],
            dim=0,
        )
        encoded = self._pipeline._encode_image_to_latent(pixel_values)
        return encoded.flatten(start_dim=1)


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
        dropout_rate: float = MLP_DROPOUT,
        clip_dropout_rate: float = MLP_CLIP_DROPOUT,
    ):
        super().__init__()
        if n_targets != 2:
            raise ValueError("SurrogateRegressor expects exactly two targets (psnr, clip)")

        # Project the embeddings to the MLP input dimension.
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
        self.psnr_body = MLPBody(psnr_in, n_wide, n_hidden, n_inner, dropout_rate)
        self.psnr_head = nn.Linear(n_inner, 1)

        # CLIP-Edited FiLM-conditioned MLP body and head.
        clip_in = img_proj_dim * 2 + text_proj_dim
        self.clip_body = FiLMMLPBody(clip_in, t_proj_dim, n_wide, n_hidden, n_inner, clip_dropout_rate)
        self.clip_head = nn.Linear(n_inner, 1)

        self.register_buffer("target_mean", torch.zeros(n_targets))
        self.register_buffer("target_std", torch.ones(n_targets))

    def _get_inputs(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Combine image/mask/text embeddings into a single context vector.
        text_emb = combine_text_embeddings(src_emb, tar_emb)
        context = torch.cat([self.img_proj(img_emb), self.mask_proj(mask_emb), self.text_proj(text_emb)], dim=-1)
        # Encode the timestep into a feature vector.
        t_feat = self.t_encoder(fourier_timestep_features(t))
        return context, t_feat

    def set_target_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the mean and standard deviation of the targets."""
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
        """Return standardized metric predictions."""
        context, t_feat = self._get_inputs(img_emb, mask_emb, src_emb, tar_emb, t)
        # Predcit PSNR and CLIP from the context and timestep feature.
        psnr = self.psnr_head(self.psnr_body(torch.cat([context, t_feat], dim=-1)))
        clip = self.clip_head(self.clip_body(context, t_feat))
        return torch.cat([psnr, clip], dim=-1) # shape (N, 2)

    def denormalize(self, standardized: torch.Tensor) -> torch.Tensor:
        """Map standardized predictions back to raw metric units (PSNR / CLIP)."""
        return standardized * self.target_std + self.target_mean


class SurrogateModel(nn.Module):
    """Bundles ChordEdit encoders with SurrogateRegressor."""

    def __init__(
        self,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        # We reuse ChordEdit's VAE and text encoder (and their preprocess helpers)
        # so their embeddings match the runs that produced (psnr, clip) labels. Full
        # from_local_* also loads UNet/scheduler; encoder-only loading would
        # require pipeline_chord.py changes, which we avoid here.
        if CHORD_EDIT_PIPELINE_TYPE == "flux":
            raise NotImplementedError(
                "CHORD_EDIT_MODEL='flux' encoder loading is not implemented in "
                "ChordEditPipeline yet; use sd_turbo / sdxl_turbo."
            )
        if CHORD_EDIT_PIPELINE_TYPE not in {"sd", "sdxl"}:
            raise ValueError(f"Unsupported CHORD_EDIT_PIPELINE_TYPE={CHORD_EDIT_PIPELINE_TYPE!r}")

        component_paths = paths_from_model_root(
            CHORD_EDIT_MODEL_ROOT, model_type=CHORD_EDIT_PIPELINE_TYPE
        )
        self.pipeline = ChordEditPipeline.from_local_weights(
            component_paths,
            model_type=CHORD_EDIT_PIPELINE_TYPE,
            image_size=CHORD_EDIT_IMAGE_SIZE,
            use_center_crop=USE_CENTER_CROP,
            compute_dtype=DEFAULT_COMPUTE_DTYPE,
            use_safety_checker=False,
            device=device,
        )
        # VAE-encoding a full EMBED_BATCH_SIZE batch at image_size=1024 in
        # fp32 needs >40 GiB of activations (OOMs on a 48 GiB card). Slicing
        # makes the VAE encode one image at a time with identical outputs.
        self.pipeline.vae.enable_slicing()
        # Encoders are inherited from the ChordEdit pipeline and are never
        # trainable: freeze the VAE and text encoder weights unconditionally.
        for param in self.pipeline.vae.parameters():
            param.requires_grad = False
        for param in self.pipeline.text_encoder.parameters():
            param.requires_grad = False
        if self.pipeline.text_encoder_2 is not None:
            for param in self.pipeline.text_encoder_2.parameters():
                param.requires_grad = False

        # Thin wrappers that call ChordEdit preprocess/encode helpers above.
        self.image_encoder = VaeImageEncoder(self.pipeline)
        self.text_encoder = TextEncoder(self.pipeline)
        self._encoder_img_dim = self.image_encoder.hidden_dim
        self._encoder_text_dim = self.text_encoder.hidden_dim
        self.regressor = SurrogateRegressor(self._encoder_img_dim, self._encoder_text_dim, n_targets=len(M_TARGET_COLS))

    @property
    def encoder_img_dim(self) -> int:
        return self._encoder_img_dim

    @property
    def encoder_text_dim(self) -> int:
        return self._encoder_text_dim

    def release_encoders(self) -> None:
        """Free VAE/text pipeline after embeddings are precomputed."""
        # Drop frozen SD encoders from GPU once embeddings exist so
        # that training only keeps the small regressor on device.
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

    def predict_emb(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict (psnr, clip) in raw metric units from precomputed embeddings."""
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
        """Predict (psnr, clip) in raw metric units for raw inputs."""

        # Set the model to evaluation mode, remembering the current mode.
        was_training = self.training
        self.eval()

        device = self.regressor.target_mean.device
        img_emb = self.image_encoder(images).to(device)
        mask_emb = self.image_encoder(masks).to(device)
        src_emb = self.text_encoder(src_prompts).to(device)
        tar_emb = self.text_encoder(tar_prompts).to(device)
        t = torch.tensor(list(zip(t_start, t_end)), dtype=torch.float, device=device)
        
        out = self.predict_emb(img_emb, mask_emb, src_emb, tar_emb, t)

        # Restore the mode the model was in before predict().
        self.train(was_training)

        return out # shape (N, 2)
