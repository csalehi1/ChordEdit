<div align="center">
  <h1>[CVPR 2026 Oral] ChordEdit: One-Step Low-Energy Transport for Image Editing</h1>
  <div class="authors">
    <span><a href="https://luliangsi.github.io">Liangsi Lu</a><sup>1</sup>, <a href="https://cxh.netlify.app/">Xuhang Chen</a><sup>2</sup>, <a href="#">Minzhe Guo</a><sup>1</sup>, <a href="#">Shichu Li</a><sup>3</sup>, <a href="#">Jingchao Wang</a><sup>4</sup>, <a href="https://cnshiyang.github.io">Yang Shi</a><sup>1†</sup></span><br>
    <span style="color: #666; font-size: 0.9em;"><sup>1</sup> Guangdong University of Technology, <sup>2</sup> Huizhou University, <sup>3</sup> Shenzhen University, <sup>4</sup> Peking University <br> <sup>†</sup> Corresponding author</span>
  </div>

  <a href="https://chordedit.github.io"><img src="https://img.shields.io/badge/Project-Page-2b7de9"></a>
  <a href="https://arxiv.org/pdf/2602.19083"><img src="https://img.shields.io/badge/arXiv-2602.19083-b31b1b.svg"></a>

  <img src="chord_show.gif" alt="ChordEdit demo" width="100%" />
</div>

## 1. Environment
- Python 3.12
- PyTorch 2.5.0
- This repository requires the `sd-turbo` weights: https://huggingface.co/stabilityai/sd-turbo
- Model root should contain:
  - `unet/`
  - `scheduler/`
  - `text_encoder/`
  - `tokenizer/`
  - `vae/`

## 2. Install Dependencies
Create and activate a Python 3.12 environment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On macOS, if `python3.12` is not available, install it first:
```bash
brew install python@3.12
```

If `.venv` was already created with a different Python version, recreate it:
```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For CUDA 12.4 Linux machines:
```bash
python -m pip install -r requirements.txt
```

For macOS or non-CUDA machines:
```bash
python -m pip install -r requirements-macos.txt
```

## 3. Run the Web Demo
Launch the interactive demo:
```bash
python app.py --model-root /path/to/sd-turbo --server-port 7860
```

Running `python app.py` from the activated environment launches a local Gradio web app.
- Left panel: upload the original image, set source prompt, target prompt, and tuning parameters.
- Right panel: view the edited output image.
- Bottom section: click built-in examples (image + source prompt + target prompt) to auto-fill inputs.

<img src="chord_app.png" alt="ChordEdit app" width="100%" />

## 4. Run PIE Benchmark Export
Run PIE-Bench export with real local paths:
```bash
python run_pie_bench.py --model-root <sd-turbo-folder> --pie-root <pie-bench-folder>
```

To run ChordEdit on up to 1000 images from the benchmark mapping, run:
```bash
python run_pie_bench.py \
  --model-root <sd-turbo-folder> \
  --pie-root <pie-bench-folder> \
  --max-samples 1000
```

Replace `<sd-turbo-folder>` with the folder containing the local `sd-turbo` weights, and replace `<pie-bench-folder>` with the folder containing `mapping_file.json`. Do not type the angle brackets literally.

Before running, verify the PIE-Bench path:
```bash
ls <pie-bench-folder>/mapping_file.json
ls <pie-bench-folder>/annotation_images
```

`--max-samples` processes the first `N` valid records from `mapping_file.json`. The official PIE-Bench set has 700 samples, so a 1000-image run requires a mapping file and image folder with at least 1000 valid entries.

`--pie-root` should point to a PIE-Bench folder containing at least:

1. `annotation_images/` — original PIE-Bench images (subfolders keep the official naming).
2. `mapping_file.json` — the mapping metadata describing prompts, instructions, and masks.

Example layout:
```
pie_bench
|-annotation_images
|-mapping_file.json
```

For PIE-Bench data preparation and protocol details, please refer to:
https://github.com/cure-lab/PnPInversion


## 5. Reproduced ChordEdit results
This averages the 700 PIE-Bench samples in PIE-Bench.

| Method | Structure Dist. ↓ | PSNR ↑ | LPIPS ↓ | MSE ↓ | SSIM ↑ | CLIP Src. ↑ | CLIP Tgt. ↑ | CLIP Edit ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chord default sd (paper)   | -      | 22.20   | 0.12825 | 0.00684   | -      | - | 25.58   | 22.96 |
| chord default sd   | 0.0295 | 22.64   | 0.1185  | 0.0080 | 0.7675 | 25.4281 | 24.8226 | 22.1554 |
| chord default sdxl | 0.0318 | 22.3838 | 0.1335  | 0.0079 | 0.7736 | 25.4281 | 25.8762 | 23.0175 |
| chord sym sd       | 0.0151 | 26.1456 | 0.0747  | 0.0039 | 0.8148 | 25.4281 | 23.6089 | 20.9540 |
| chord sym sdxl     | 0.0169 | 26.1842 | 0.0859  | 0.0034 | 0.8295 | 25.4281 | 24.7610 | 21.6957 |


# Citation
If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!
```
@article{lu2026chordedit,
  title={ChordEdit: One-Step Low-Energy Transport for Image Editing},
  author={Lu, Liangsi and Chen, Xuhang and Guo, Minzhe and Li, Shichu and Wang, Jingchao and Shi, Yang},
  journal={arXiv preprint arXiv:2602.19083},
  year={2026}
}
```
