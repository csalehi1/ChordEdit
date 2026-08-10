"""
Precompute frozen CLIP image embeddings for the 12k SDXL-Turbo tournament
dataset's source images, once, so training doesn't need to run a vision
transformer forward pass every step. Uses openai/clip-vit-large-patch14,
the same checkpoint already used elsewhere in this project for CLIP
similarity metrics.

Saves a single dict {sample_id: FloatTensor[768]} via torch.save.
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path("/data/home/zarageddes/research/ChordEdit")
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path("/shared/ssd_30T/zarageddes/tournament_12k_sdxlturbo")
MAPPING_JSON = DATA_ROOT / "mapping_file.json"
OUT_PATH = REPO_ROOT / "models" / "classification" / "data" / "sdxlturbo_12k_clip_image_embeddings.pt"

MODEL_ID = "openai/clip-vit-large-patch14"
BATCH_SIZE = 64


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    mapping = json.loads(MAPPING_JSON.read_text())
    sample_ids = sorted(mapping.keys())
    print(f"{len(sample_ids)} source images to embed", flush=True)

    embeddings = {}
    with torch.no_grad():
        for i in range(0, len(sample_ids), BATCH_SIZE):
            batch_ids = sample_ids[i:i + BATCH_SIZE]
            images = [Image.open(DATA_ROOT / mapping[sid]["image_path"]).convert("RGB") for sid in batch_ids]
            inputs = processor(images=images, return_tensors="pt").to(device)
            image_embeds = model.get_image_features(**inputs).pooler_output
            for sid, emb in zip(batch_ids, image_embeds.cpu()):
                embeddings[sid] = emb
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"{i + len(batch_ids)}/{len(sample_ids)} done", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, OUT_PATH)
    print(f"saved {len(embeddings)} embeddings -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
