"""
_flux.py

Encoder-only FLUX stand-in for ChordEditPipeline, for --skip-generated
embedding caching. pipeline_chord.py on this branch has no FLUX support
(that lives on zara/flux-schnell-support), and caching only needs the VAE
and text encoders — loading the 12B transformer would be pure waste — so
this wraps a transformer-less diffusers FluxPipeline behind the four
methods run_factorized_grid touches before its skip_generated return:
_prepare_edit_params, _prepare_image_tensor, _encode_image_to_latent, and
encode_prompt.

Semantics mirror zara/flux-schnell-support's ChordEditPipeline exactly:
image preprocessing is the same center-crop + LANCZOS-resize + [-1, 1]
transform, latents are (mode() - shift_factor) * scaling_factor — (16, 64,
64) at 512px — and encode_prompt returns FluxPipeline.encode_prompt's T5
hidden states (max_sequence_length 512, matching that branch) with the CLIP
pooled vector, packed into this branch's _PromptCondition. mask_tokenizer
exposes the T5 tokenizer so _pool_text_embeds masks the sequence axis the
hidden states actually have.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_chord import _CenterSquareCropTransform, _PromptCondition

try:
    from PIL import Image
except ImportError:  # pragma: no cover - PIL ships with the pipeline deps
    Image = None


class FluxEncoderPipeline:
    """VAE + text-encoder subset of a FLUX ChordEditPipeline (no transformer)."""

    def __init__(
        self,
        model_root: str,
        *,
        device: str,
        image_size: int,
        torch_dtype: torch.dtype = torch.float32,
        max_sequence_length: int = 512,
    ) -> None:
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            model_root,
            transformer=None,
            torch_dtype=torch_dtype,
        )
        pipe.to(device)
        self._pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        # _pool_text_embeds masks with this: hidden states come from T5, not CLIP.
        self.mask_tokenizer = pipe.tokenizer_2
        self.image_size = image_size
        self._device = device
        self._compute_dtype = torch_dtype
        self._max_sequence_length = max_sequence_length
        self._use_center_crop = True
        self._vae_transform = transforms.Compose(
            [
                _CenterSquareCropTransform(),
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _prepare_edit_params(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        # Encode-only: the edit params are never consumed before the
        # skip_generated return, so pass the config through untouched.
        return dict(cfg)

    def _prepare_image_tensor(self, image: "Image.Image" | torch.Tensor) -> torch.Tensor:
        if Image is not None and isinstance(image, Image.Image):
            vae_tensor = self._vae_transform(image)
        elif torch.is_tensor(image):
            tensor = image.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            tensor = tensor * 2.0 - 1.0
            vae_tensor = tensor
        else:
            raise TypeError("image must be a PIL.Image or a torch.Tensor.")

        if vae_tensor.ndim == 3:
            vae_tensor = vae_tensor.unsqueeze(0)
        return vae_tensor.to(device=self._device, dtype=self._compute_dtype)

    def _encode_image_to_latent(self, pixel_values: torch.Tensor) -> torch.Tensor:
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
        pixel_values = pixel_values.to(device=self._device, dtype=self._compute_dtype)
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        # FLUX VAE latents use both scaling and shift.
        latents = (latents - shift_factor) * scaling_factor
        return latents.to(device=self._device, dtype=self._compute_dtype)

    def encode_prompt(self, prompts: Sequence[str]) -> _PromptCondition:
        prompt_embeds, pooled_embeds, _txt_ids = self._pipe.encode_prompt(
            prompt=list(prompts),
            prompt_2=None,
            device=self._device,
            max_sequence_length=self._max_sequence_length,
        )
        return _PromptCondition(
            hidden_states=prompt_embeds.to(device=self._device, dtype=self._compute_dtype),
            pooled_embeds=pooled_embeds.to(device=self._device, dtype=self._compute_dtype),
            time_ids=None,
        )
