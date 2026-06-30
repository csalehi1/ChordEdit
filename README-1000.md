# ImgEdit to ChordEdit Conversion Pipeline

This README documents a portable pipeline for converting ImgEdit parquet/image data into a ChordEdit-ready dataset.

The pipeline is:

1. Read ImgEdit parquet metadata.
2. Flatten ImgEdit records into a PIE-style CSV.
3. Use a local Qwen VLM to generate ChordEdit-ready prompts.
4. Export the labeled CSV into ChordEdit's `mapping_file.json` layout.
5. Run ChordEdit with the exported dataset.

The commands below use placeholder paths so the workflow can be adapted to any machine.

## Required Inputs

You need these components:

```text
ChordEdit repository
ImgEdit parquet files
Extracted ImgEdit source/target images
The conversion scripts from this pipeline
Qwen vision-language model access through Hugging Face
sd-turbo weights for ChordEdit
```

ChordEdit's `run_pie_bench.py` expects an `sd-turbo` model root containing:

```text
unet/
scheduler/
text_encoder/
tokenizer/
vae/
```

## Path Variables

Choose paths for your machine.

### Windows Command Prompt

```bat
set CHORDEDIT_ROOT=C:\path\to\ChordEdit
set IMGEDIT_ROOT=D:\path\to\ImgEdit
set PARQUET_ROOT=%IMGEDIT_ROOT%\Parquet
set IMAGE_ROOT=%IMGEDIT_ROOT%\Singleturn\action_part1
set PIPELINE_TOOLS=%CHORDEDIT_ROOT%\imgedit_pipeline\scripts
set MODEL_ROOT=C:\path\to\sd-turbo
```

### Linux or macOS Bash

```bash
export CHORDEDIT_ROOT=/path/to/ChordEdit
export IMGEDIT_ROOT=/path/to/ImgEdit
export PARQUET_ROOT="$IMGEDIT_ROOT/Parquet"
export IMAGE_ROOT="$IMGEDIT_ROOT/Singleturn/action_part1"
export PIPELINE_TOOLS="$CHORDEDIT_ROOT/imgedit_pipeline/scripts"
export MODEL_ROOT=/path/to/sd-turbo
```

`IMAGE_ROOT` can point to a parent folder. The scripts support recursive image search, so images can be inside subfolders such as:

```text
<IMAGE_ROOT>/part1/*.jpg
```

## Pipeline Scripts

This pipeline uses three main scripts:

```text
convert_imgedit_to_pie_csv.py
label_imgedit_with_qwen.py
export_qwen_csv_to_chordedit.py
```

Recommended layout:

```text
<CHORDEDIT_ROOT>/
  imgedit_pipeline/
    README.md
    scripts/
      convert_imgedit_to_pie_csv.py
      label_imgedit_with_qwen.py
      export_qwen_csv_to_chordedit.py
    docs/
```

Optional/alternative scripts:

```text
enrich_imgedit_prompts_vlm.py
convert_imgedit_benchmark_to_pie_csv.py
```

`enrich_imgedit_prompts_vlm.py` is an API-based prompt enrichment script. It is not required if using local Qwen.

`convert_imgedit_benchmark_to_pie_csv.py` is for ImgEdit benchmark archive data, not the parquet-to-Qwen path.

## Environment Setup

Create or activate a Python environment with PyTorch, Transformers, Qwen dependencies, and ChordEdit dependencies.

Example:

```bat
conda create -n imgedit-qwen python=3.12 -y
conda activate imgedit-qwen
```

Install PyTorch for your GPU. Pick the wheel that matches your hardware and driver.

Example for CUDA 12.8:

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install Qwen and general dependencies:

```bat
python -m pip install transformers accelerate qwen-vl-utils pillow pandas pyarrow
```

Install ChordEdit dependencies without forcing a different PyTorch version:

```bat
python -m pip install diffusers datasets torchmetrics torch-fidelity gradio matplotlib pyyaml seaborn safetensors
```

If your ChordEdit repository has a `requirements.txt`, inspect it before installing. Some repositories pin a specific PyTorch build that may not match your GPU.

Useful optional environment variables:

### Windows Command Prompt

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set HF_HOME=D:\HFCache
set TRANSFORMERS_CACHE=D:\HFCache\transformers
set PIP_CACHE_DIR=D:\PipCache
```

### Linux or macOS Bash

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PIP_CACHE_DIR=/path/to/pip_cache
```

## Step 1: Convert ImgEdit Parquet to PIE-Style CSV

Purpose:

`convert_imgedit_to_pie_csv.py` flattens ImgEdit parquet records into one CSV row per edit. At this stage, `source_prompt` and `target_prompt` are usually blank because ImgEdit parquet metadata does not provide ChordEdit-ready prompts directly.

### Windows

```bat
python "%PIPELINE_TOOLS%\convert_imgedit_to_pie_csv.py" --parquet-root "%PARQUET_ROOT%" --out-csv "%IMGEDIT_ROOT%\imgedit_pie_style_1000.csv" --limit 1000
```

### Linux or macOS

```bash
python "$PIPELINE_TOOLS/convert_imgedit_to_pie_csv.py" --parquet-root "$PARQUET_ROOT" --out-csv "$IMGEDIT_ROOT/imgedit_pie_style_1000.csv" --limit 1000
```

Important output:

```text
<IMGEDIT_ROOT>/imgedit_pie_style_1000.csv
```

Main columns:

```text
id
dataset_name
edit_type
source_image
target_image
mask_path
source_prompt
target_prompt
editing_instruction
clip_score
clip_score_category
turn_index
source_parquet
sample_key
```

## Step 2: Smoke Test Qwen Labeling

Purpose:

`label_imgedit_with_qwen.py` sends the source and target images to Qwen and asks it to produce ChordEdit-ready prompt fields.

Default model used:

```text
Qwen/Qwen3-VL-2B-Instruct
```

### Windows

```bat
python "%PIPELINE_TOOLS%\label_imgedit_with_qwen.py" --input-csv "%IMGEDIT_ROOT%\imgedit_pie_style_1000.csv" --output-csv "%IMGEDIT_ROOT%\imgedit_pie_style_10_qwen_labeled.csv" --image-root "%IMAGE_ROOT%" --model "Qwen/Qwen3-VL-2B-Instruct" --limit 10 --recursive-image-search --dtype bfloat16 --attn-implementation eager --max-new-tokens 160 --max-image-side 512
```

### Linux or macOS

```bash
python "$PIPELINE_TOOLS/label_imgedit_with_qwen.py" --input-csv "$IMGEDIT_ROOT/imgedit_pie_style_1000.csv" --output-csv "$IMGEDIT_ROOT/imgedit_pie_style_10_qwen_labeled.csv" --image-root "$IMAGE_ROOT" --model "Qwen/Qwen3-VL-2B-Instruct" --limit 10 --recursive-image-search --dtype bfloat16 --attn-implementation eager --max-new-tokens 160 --max-image-side 512
```

If GPU memory is tight, lower image size:

```text
--max-image-side 384
```

## Step 3: Label 1000 Samples with Qwen

### Windows

```bat
python "%PIPELINE_TOOLS%\label_imgedit_with_qwen.py" --input-csv "%IMGEDIT_ROOT%\imgedit_pie_style_1000.csv" --output-csv "%IMGEDIT_ROOT%\imgedit_pie_style_1000_qwen_labeled.csv" --image-root "%IMAGE_ROOT%" --model "Qwen/Qwen3-VL-2B-Instruct" --limit 1000 --recursive-image-search --dtype bfloat16 --attn-implementation eager --max-new-tokens 160 --max-image-side 512 --resume
```

### Linux or macOS

```bash
python "$PIPELINE_TOOLS/label_imgedit_with_qwen.py" --input-csv "$IMGEDIT_ROOT/imgedit_pie_style_1000.csv" --output-csv "$IMGEDIT_ROOT/imgedit_pie_style_1000_qwen_labeled.csv" --image-root "$IMAGE_ROOT" --model "Qwen/Qwen3-VL-2B-Instruct" --limit 1000 --recursive-image-search --dtype bfloat16 --attn-implementation eager --max-new-tokens 160 --max-image-side 512 --resume
```

The `--resume` flag lets the script continue from a partially completed output CSV.

Inspect the output:

### Windows PowerShell

```powershell
Import-Csv "$env:IMGEDIT_ROOT\imgedit_pie_style_1000_qwen_labeled.csv" | Select-Object -First 10 id,source_prompt,target_prompt,foreground,foreground_target,qwen_edit_type,vlm_confidence,vlm_error | Format-List
```

### Linux or macOS

```bash
python - <<'PY'
import csv, os
path = os.path.join(os.environ["IMGEDIT_ROOT"], "imgedit_pie_style_1000_qwen_labeled.csv")
with open(path, newline="", encoding="utf-8-sig") as f:
    for idx, row in zip(range(10), csv.DictReader(f)):
        print(row["id"], row["source_prompt"], row["target_prompt"], row.get("vlm_confidence", ""), row.get("vlm_error", ""))
PY
```

Expected successful row shape:

```text
id                : imgedit_00000000_00
source_prompt     : A man with glasses and a beard sits on a couch, holding a music sheet.
target_prompt     : A man with glasses and a beard sits on a couch, holding a music sheet closer to his chest.
foreground        : record
foreground_target : record
qwen_edit_type    : action
vlm_confidence    : high
vlm_error         :
```

## Qwen Labeling Requirements

The Qwen script asks for strict JSON:

```json
{
  "source_prompt": "...",
  "target_prompt": "...",
  "foreground": "...",
  "foreground_target": "...",
  "edit_type": "add|remove|replace|action|style|background|adjust|other",
  "vlm_confidence": "high|medium|low",
  "notes": "..."
}
```

The generated fields are intended for ChordEdit:

```text
source_prompt describes the original scene in one natural sentence.
target_prompt describes the edited scene in one natural sentence.
The edit difference should be clear.
Shared scene context should remain consistent between source and target.
foreground identifies the edited object, person, attribute, action, or region.
foreground_target identifies what it becomes or the new state/action.
```

## Step 4: Export the Qwen CSV to ChordEdit Format

Purpose:

`export_qwen_csv_to_chordedit.py` creates a ChordEdit-compatible dataset folder from the Qwen-labeled CSV.

Choose an export path:

### Windows

```bat
set CHORDEDIT_EXPORT=%CHORDEDIT_ROOT%\imgedit_qwen_export
```

### Linux or macOS

```bash
export CHORDEDIT_EXPORT="$CHORDEDIT_ROOT/imgedit_qwen_export"
```

Run the export:

### Windows

```bat
python "%PIPELINE_TOOLS%\export_qwen_csv_to_chordedit.py" --input-csv "%IMGEDIT_ROOT%\imgedit_pie_style_1000_qwen_labeled.csv" --out-root "%CHORDEDIT_EXPORT%" --image-root "%IMAGE_ROOT%" --recursive-image-search --limit 1000 --min-confidence low --copy-targets --make-full-masks --overwrite
```

### Linux or macOS

```bash
python "$PIPELINE_TOOLS/export_qwen_csv_to_chordedit.py" --input-csv "$IMGEDIT_ROOT/imgedit_pie_style_1000_qwen_labeled.csv" --out-root "$CHORDEDIT_EXPORT" --image-root "$IMAGE_ROOT" --recursive-image-search --limit 1000 --min-confidence low --copy-targets --make-full-masks --overwrite
```

Output structure:

```text
<CHORDEDIT_EXPORT>/
  annotation_images/
  annotation_masks/
  target_images/
  mapping_file.json
  exported_rows.csv
```

Example `mapping_file.json` entry:

```json
{
  "imgedit_00000000_00": {
    "dataset_name": "imgedit",
    "image_path": "imgedit_00000000_00.png",
    "source_prompt": "A man with glasses and a beard sits on a couch, holding a music sheet.",
    "target_prompt": "A man with glasses and a beard sits on a couch, holding a music sheet closer to his chest.",
    "editing_instruction": "The person lowers the record closer to their chest.",
    "foreground": "record",
    "foreground_target": "record",
    "edit_type": "action",
    "clip_score": "",
    "clip_score_category": "missing",
    "vlm_confidence": "high",
    "original_source_image": "hOdCsLRNAyw_segment_21_frame_0.jpg",
    "original_target_image": "hOdCsLRNAyw_segment_21_frame_55.jpg",
    "mask_path": "imgedit_00000000_00.png",
    "target_image_path": "target_images/imgedit_00000000_00.jpg"
  }
}
```

Note:

ChordEdit's current `run_pie_bench.py` reads `image_path`, `source_prompt`, and `target_prompt`. Full-image masks are created for compatibility with other PIE-style tooling.

## Step 5: Run ChordEdit

### Windows

```bat
cd /d "%CHORDEDIT_ROOT%"
conda activate imgedit-qwen
python run_pie_bench.py --pie-root "%CHORDEDIT_EXPORT%" --mapping-file "mapping_file.json" --image-subdir "annotation_images" --method-name "ChordEdit_imgedit_qwen_1000" --model-root "%MODEL_ROOT%" --device cuda:0 --max-samples 1000 --overwrite
```

### Linux or macOS

```bash
cd "$CHORDEDIT_ROOT"
conda activate imgedit-qwen
python run_pie_bench.py --pie-root "$CHORDEDIT_EXPORT" --mapping-file "mapping_file.json" --image-subdir "annotation_images" --method-name "ChordEdit_imgedit_qwen_1000" --model-root "$MODEL_ROOT" --device cuda:0 --max-samples 1000 --overwrite
```

Generated images are written under:

```text
<CHORDEDIT_EXPORT>/output/ChordEdit_imgedit_qwen_1000/
```

The exact nested folder name may include ChordEdit settings such as mode, model type, and timestep values.

## Files Produced by the Pipeline

Intermediate CSV:

```text
<IMGEDIT_ROOT>/imgedit_pie_style_1000.csv
```

Qwen-labeled CSV:

```text
<IMGEDIT_ROOT>/imgedit_pie_style_1000_qwen_labeled.csv
```

ChordEdit export:

```text
<CHORDEDIT_EXPORT>/mapping_file.json
<CHORDEDIT_EXPORT>/annotation_images/
<CHORDEDIT_EXPORT>/annotation_masks/
<CHORDEDIT_EXPORT>/target_images/
<CHORDEDIT_EXPORT>/exported_rows.csv
```

ChordEdit generated outputs:

```text
<CHORDEDIT_EXPORT>/output/<method_name>/
```

## Troubleshooting

### `No module named 'diffusers'`

Install ChordEdit dependencies:

```bat
python -m pip install diffusers accelerate datasets torchmetrics torch-fidelity gradio matplotlib pyyaml seaborn safetensors
```

### CUDA kernel image error

Error:

```text
CUDA error: no kernel image is available for execution on the device
```

Fix:

Install a PyTorch build that supports your GPU and CUDA driver. For newer NVIDIA GPUs, a newer CUDA wheel may be required.

Example:

```bat
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CUDA out of memory during Qwen labeling

Lower the image size:

```text
--max-image-side 384
```

You can also reduce generation length:

```text
--max-new-tokens 96
```

Keep this environment variable enabled when useful:

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Blank prompts or `vlm_error`

The labeling script attempts to salvage partial JSON if Qwen returns truncated output. If many rows still fail, rerun with:

```text
--resume --max-new-tokens 160
```

For low VRAM:

```text
--resume --max-new-tokens 96 --max-image-side 384
```

### Windows command-line line breaks

In Windows Command Prompt, use one-line commands. Do not paste PowerShell-style backtick line continuations into Command Prompt.

## Summary

The final conversion path is:

```text
ImgEdit parquet
  -> convert_imgedit_to_pie_csv.py
  -> imgedit_pie_style_1000.csv
  -> label_imgedit_with_qwen.py
  -> imgedit_pie_style_1000_qwen_labeled.csv
  -> export_qwen_csv_to_chordedit.py
  -> ChordEdit PIE-style export folder
  -> run_pie_bench.py
  -> ChordEdit-generated edited images
```
