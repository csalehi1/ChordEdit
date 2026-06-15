import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PIE = Path("~/datasets/PIE-Bench_v1").expanduser()
OUT = PIE / "output"
N = 20

METHODS = {
    "delta015": OUT / "ChordEdit_delta015/annotation_images/chord_default_auto_None_None_0.15",
    "delta000": OUT / "ChordEdit_delta000/annotation_images/chord_default_auto_None_None_0.0",
}


def clean(s):
    return s.replace("[", "").replace("]", "")


def as_tensor(x):
    if hasattr(x, "pooler_output"):
        return x.pooler_output
    return x

def norm(x):
    x = as_tensor(x)
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def img_emb(model, proc, path, device):
    img = Image.open(path).convert("RGB")
    x = proc(images=img, return_tensors="pt").to(device)
    return norm(model.get_image_features(**x))


@torch.no_grad()
def txt_emb(model, proc, text, device):
    x = proc(text=[text], return_tensors="pt", padding=True).to(device)
    return norm(model.get_text_features(**x))


def cos(a, b):
    return float((a * b).sum().item())


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
model.eval()

mapping = json.load(open(PIE / "mapping_file.json"))
keys = list(mapping.keys())[:N]

rows = []

for name, method_dir in METHODS.items():
    vals = []
    print("evaluating", name)

    for key in keys:
        item = mapping[key]
        rel = Path(item["image_path"])

        src_img_path = PIE / "annotation_images" / rel
        out_img_path = method_dir / rel

        src_prompt = clean(item["original_prompt"])
        tgt_prompt = clean(item["editing_prompt"])

        src_img = img_emb(model, proc, src_img_path, device)
        out_img = img_emb(model, proc, out_img_path, device)

        src_txt = txt_emb(model, proc, src_prompt, device)
        tgt_txt = txt_emb(model, proc, tgt_prompt, device)

        clip_src = cos(out_img, src_txt)
        clip_tgt = cos(out_img, tgt_txt)

        img_direction = norm(out_img - src_img)
        txt_direction = norm(tgt_txt - src_txt)
        clip_edit = cos(img_direction, txt_direction)

        vals.append({
            "CLIP Src": clip_src,
            "CLIP Tgt": clip_tgt,
            "CLIP Edit": clip_edit,
        })

    df = pd.DataFrame(vals)

    rows.append({
        "Method": name,
        "Samples": len(df),
        "CLIP Src": df["CLIP Src"].mean(),
        "CLIP Tgt": df["CLIP Tgt"].mean(),
        "CLIP Edit": df["CLIP Edit"].mean(),
    })

summary = pd.DataFrame(rows)
print(summary)

save_path = OUT / "mini_delta_clip_metrics.csv"
summary.to_csv(save_path, index=False)
print("saved:", save_path)