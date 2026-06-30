# ImgEdit to PIE-Style CSV: First Step

This first-step converter flattens ImgEdit parquet metadata into one CSV row per
edit. It works on either a single `.parquet` file or the whole `ImgEdit/Parquet`
directory.

## Output Columns

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

ImgEdit's final parquet files do not contain PIE-style `source_prompt` and
`target_prompt`, so those columns are created but left blank for now. The
`editing_instruction` column comes from ImgEdit's `prompt` field.

## Basic Conversion

```powershell
python outputs\convert_imgedit_to_pie_csv.py `
  --parquet-root "C:\Datasets\ImgEdit\Parquet" `
  --out-csv "C:\Datasets\ImgEdit\imgedit_pie_style.csv"
```

## Smoke Test On 100 Rows

```powershell
python outputs\convert_imgedit_to_pie_csv.py `
  --parquet-root "C:\Datasets\ImgEdit\Parquet" `
  --out-csv "C:\Datasets\ImgEdit\imgedit_pie_style_100.csv" `
  --limit 100
```

## Add CLIP Score Categories

If you have a preprocessing/score file with `clip_score` values, pass it with
`--clip-score-file`. The converter will try to match scores by full path,
basename, parent sample key, or slash-to-underscore sample key.

```powershell
python outputs\convert_imgedit_to_pie_csv.py `
  --parquet-root "C:\Datasets\ImgEdit\Parquet" `
  --clip-score-file "C:\Datasets\ImgEdit_recap_mask\jsons\some_scores.jsonl" `
  --out-csv "C:\Datasets\ImgEdit\imgedit_pie_style_with_clip.csv"
```

Default CLIP categories:

```text
high    >= 0.90
medium  >= 0.75 and < 0.90
low     < 0.75
missing no matched score
```

You can change the thresholds:

```powershell
python outputs\convert_imgedit_to_pie_csv.py `
  --parquet-root "C:\Datasets\ImgEdit\Parquet" `
  --clip-score-file "C:\Datasets\clip_scores.jsonl" `
  --medium-threshold 0.80 `
  --high-threshold 0.93 `
  --out-csv "C:\Datasets\ImgEdit\imgedit_pie_style_with_clip.csv"
```

## Dependency

The script streams parquet with `pyarrow`. If your environment does not have it:

```powershell
pip install pyarrow
```

