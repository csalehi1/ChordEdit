"""
Regenerates edit-region masks for every sample in the ReShapeBench export
using CLIPSeg text-prompted segmentation, driven by each row's `foreground`
label.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

LOGGER = logging.getLogger("regenerate_masks")

MODEL_NAME = "CIDAS/clipseg-rd64-refined"


def load_clipseg(device: str):
    LOGGER.info("Loading CLIPSeg (%s) ...", MODEL_NAME)
    processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model


def segment(
    image: Image.Image,
    text_prompt: str,
    processor,
    model,
    device: str,
    threshold: float,
) -> Image.Image:
    """Returns a binary (0/255) L-mode PIL mask the same size as the input image."""
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits: (1, H_small, W_small) -- low-res segmentation logits
    logits = outputs.logits[0]
    probs = torch.sigmoid(logits).cpu().numpy()

    # Resize the low-res probability map up to the original image size.
    prob_img = Image.fromarray((probs * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    prob_arr = np.array(prob_img).astype(np.float32) / 255.0

    binary = (prob_arr >= threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=str, default="reshapebench_export")
    parser.add_argument("--threshold", type=float, default=0.35, help="Sigmoid probability cutoff for binarizing the mask.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    export_root = Path(args.export_root).expanduser().resolve()
    img_dir = export_root / "annotation_images"
    mask_dir = export_root / "annotation_masks"
    mapping_path = export_root / "mapping_file.json"

    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping_file.json not found at {mapping_path}. Run convert_reshapebench.py first.")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    sample_ids = list(mapping.keys())
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    processor, model = load_clipseg(device)

    updated, failed = 0, 0
    for i, sample_id in enumerate(sample_ids, start=1):
        entry = mapping[sample_id]
        foreground_label = entry.get("foreground")
        if not foreground_label:
            LOGGER.warning("Sample %s has no 'foreground' label; skipping.", sample_id)
            failed += 1
            continue

        img_path = img_dir / entry["image_path"]
        if not img_path.exists():
            LOGGER.warning("Sample %s image missing at %s; skipping.", sample_id, img_path)
            failed += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            mask = segment(image, foreground_label, processor, model, device, args.threshold)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to segment %s ('%s'): %s", sample_id, foreground_label, exc)
            failed += 1
            continue

        mask_dest = mask_dir / entry["mask_path"]
        mask_dest.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_dest)
        updated += 1

        if i % 25 == 0 or i == len(sample_ids):
            LOGGER.info("Processed %d/%d (failed=%d)", i, len(sample_ids), failed)

    LOGGER.info("Done. Regenerated %d mask(s), failed %d.", updated, failed)


if __name__ == "__main__":
    main()