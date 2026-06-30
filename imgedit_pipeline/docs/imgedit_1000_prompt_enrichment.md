# ImgEdit 1000-Image Prompt Enrichment Demo

This second-stage script reads a PIE-style ImgEdit CSV and uses a vision model
to generate:

```text
source_prompt
target_prompt
prompt_confidence
prompt_review_status
edit_summary
edit_type_guess
```

It also uses `clip_score` / `clip_score_category` to decide whether each row
should be auto-accepted or reviewed.

## Important

The parquet-only download gives you metadata paths, not the actual image bytes.
The enrichment script needs the real source and target image files to exist on
your PC.

If your dry run says `missing local source_image and target_image`, that means
you need to download/extract the relevant ImgEdit image files before captioning.

For the first 1000 rows produced by the current parquet converter, the rows come
from:

```text
C:\Datasets\ImgEdit\Parquet\action_part1.parquet
```

The matching image data is in the split tar files under:

```text
Singleturn/action_part1.tar.split.*
```

Download only that archive first, not the whole multi-terabyte dataset:

```powershell
huggingface-cli download `
  --repo-type dataset `
  sysuyy/ImgEdit `
  --include "Singleturn/action_part1.tar.split.*" `
  --local-dir "C:\Datasets\ImgEdit"
```

Then merge and extract the split tar:

```powershell
cd "C:\Datasets\ImgEdit\Singleturn"
cmd /c copy /b action_part1.tar.split.* action_part1.tar
mkdir action_part1
tar -xf action_part1.tar -C action_part1
```

## Step 1: Confirm Your 1000-Row CSV

```powershell
Import-Csv "C:\Datasets\ImgEdit\imgedit_pie_style_100.csv" |
  Select-Object -First 5 |
  Format-List
```

For a 1000-row version:

```powershell
python "C:\Users\Caleb\Documents\Codex\2026-06-25\usi\outputs\convert_imgedit_to_pie_csv.py" `
  --parquet-root "C:\Datasets\ImgEdit\Parquet" `
  --out-csv "C:\Datasets\ImgEdit\imgedit_pie_style_1000.csv" `
  --limit 1000
```

## Step 2: Dry Run Image Resolution

This checks whether the image files referenced in the CSV actually exist.

```powershell
python "C:\Users\Caleb\Documents\Codex\2026-06-25\usi\outputs\enrich_imgedit_prompts_vlm.py" `
  --input-csv "C:\Datasets\ImgEdit\imgedit_pie_style_1000.csv" `
  --output-csv "C:\Datasets\ImgEdit\imgedit_pie_style_1000_dryrun.csv" `
  --image-root "C:\Datasets\ImgEdit\Singleturn\action_part1" `
  --limit 1000 `
  --recursive-image-search `
  --dry-run
```

## Step 3: Install API Dependency

```powershell
pip install openai
```

## Step 4: Set Your API Key

```powershell
$env:OPENAI_API_KEY="YOUR_KEY_HERE"
```

## Step 5: Run The 1000-Image Prompt Demo

Start with `--detail low` for a cheaper pilot.

```powershell
python "C:\Users\Caleb\Documents\Codex\2026-06-25\usi\outputs\enrich_imgedit_prompts_vlm.py" `
  --input-csv "C:\Datasets\ImgEdit\imgedit_pie_style_1000.csv" `
  --output-csv "C:\Datasets\ImgEdit\imgedit_pie_style_1000_with_prompts.csv" `
  --image-root "C:\Datasets\ImgEdit\Singleturn\action_part1" `
  --limit 1000 `
  --model "gpt-5.5" `
  --detail low `
  --recursive-image-search `
  --resume
```

If your account does not have access to `gpt-5.5`, rerun with a
vision-capable model your account can access.

## CLIP Score Behavior

The script does not blindly trust every row:

```text
high    -> likely auto_accept
medium  -> usually review_recommended
low     -> usually manual_review
missing -> usually manual_review
```

It still generates prompts for low/missing rows if the images exist, but those
rows should be filtered or manually inspected before becoming benchmark data.
