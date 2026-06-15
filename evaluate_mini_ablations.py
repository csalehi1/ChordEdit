import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PIE_ROOT = Path("~/datasets/PIE-Bench_v1").expanduser()
OUTPUT_ROOT = PIE_ROOT / "output"

METHODS = {
    "ChordEdit_delta015": OUTPUT_ROOT / "ChordEdit_delta015/annotation_images/chord_default_auto_None_None_0.15",
    "ChordEdit_delta000": OUTPUT_ROOT / "ChordEdit_delta000/annotation_images/chord_default_auto_None_None_0.0",
    "ChordEdit_tstart100": OUTPUT_ROOT / "ChordEdit_tstart100/annotation_images/chord_default_auto_1.0_None_None",
    "ChordEdit_tstart060": OUTPUT_ROOT / "ChordEdit_tstart060/annotation_images/chord_default_auto_0.6_None_None",
    "ChordEdit_tend020": OUTPUT_ROOT / "ChordEdit_tend020/annotation_images/chord_default_auto_None_0.2_None",
    "ChordEdit_tend050": OUTPUT_ROOT / "ChordEdit_tend050/annotation_images/chord_default_auto_None_0.5_None",
}


def load_rgb(path: Path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10 * math.log10(1.0 / m)


def simple_ssim(a, b):
    # Simple global SSIM over RGB image. Good enough for a first sanity-check table.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()

    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))


mapping = json.load(open(PIE_ROOT / "mapping_file.json"))
keys = list(mapping.keys())[:20]

rows = []

for method_name, method_dir in METHODS.items():
    per_sample = []

    for key in keys:
        item = mapping[key]
        rel_path = Path(item["image_path"])

        src_path = PIE_ROOT / "annotation_images" / rel_path
        out_path = method_dir / rel_path

        if not out_path.exists():
            print(f"Missing output: {out_path}")
            continue

        src = load_rgb(src_path)
        out = load_rgb(out_path, size=(src.shape[1], src.shape[0]))

        per_sample.append({
            "MSE": mse(src, out),
            "PSNR": psnr(src, out),
            "SSIM": simple_ssim(src, out),
        })

    df = pd.DataFrame(per_sample)
    rows.append({
        "Method": method_name,
        "Samples": len(df),
        "MSE": df["MSE"].mean(),
        "PSNR": df["PSNR"].mean(),
        "SSIM": df["SSIM"].mean(),
    })

summary = pd.DataFrame(rows)
out_csv = OUTPUT_ROOT / "mini_ablation_preservation_metrics.csv"
summary.to_csv(out_csv, index=False)

print(summary)
print(f"\nSaved: {out_csv}")
