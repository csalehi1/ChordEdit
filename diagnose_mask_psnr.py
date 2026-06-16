import json, math
from pathlib import Path
import numpy as np
from PIL import Image

PIE_ROOT = Path.home() / "datasets/PIE-Bench_v1"
SRC_ROOT = PIE_ROOT / "annotation_images"
OUT_ROOT = Path("/shared/ssd_30T/zarageddes/chordedit_original_ablations/output")
mapping = json.load(open(PIE_ROOT / "mapping_file.json"))

methods = [
    "table2_naive_delta000_w_prox",
    "table2_ours_delta015_w_prox",
    "table2_naive_delta000_wo_prox",
    "table2_ours_delta015_wo_prox",
]

def decode_rle_mask(rle, h=512, w=512):
    mask = np.zeros(h * w, dtype=bool)
    for start, length in zip(rle[0::2], rle[1::2]):
        mask[int(start):int(start) + int(length)] = True
    return mask.reshape(h, w)

def psnr_from_mse(mse):
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

print("method,num_images,mask_fraction,whole_psnr,mask_psnr,inverse_mask_psnr")

for method in methods:
    gen_paths = sorted((OUT_ROOT / method).rglob("*.jpg"))
    whole_psnrs = []
    mask_psnrs = []
    inv_psnrs = []
    mask_fracs = []

    for gen_path in gen_paths:
        image_id = gen_path.stem
        item = mapping[image_id]
        src_path = SRC_ROOT / item["image_path"]

        src = load_rgb(src_path)
        gen = load_rgb(gen_path)
        mask = decode_rle_mask(item["mask"], h=src.shape[0], w=src.shape[1])
        inv_mask = ~mask

        diff2 = (src - gen) ** 2
        whole_psnrs.append(psnr_from_mse(float(diff2.mean())))
        mask_fracs.append(float(mask.mean()))

        if mask.any():
            mask_psnrs.append(psnr_from_mse(float(diff2[mask].mean())))
        if inv_mask.any():
            inv_psnrs.append(psnr_from_mse(float(diff2[inv_mask].mean())))

    def avg(xs):
        return float(np.mean(xs)) if xs else float("nan")

    print(
        f"{method},{len(gen_paths)},"
        f"{avg(mask_fracs):.4f},"
        f"{avg(whole_psnrs):.4f},"
        f"{avg(mask_psnrs):.4f},"
        f"{avg(inv_psnrs):.4f}"
    )
