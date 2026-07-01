# LLM Timestep Policy — Results

Results for using a VLM (Qwen3-VL) to predict the best ChordEdit diffusion timestep
bucket for an image edit, given the source image, source prompt, and target prompt.

## Task

Predict `LOW` / `MID` / `HIGH`, corresponding to `t_start` ranges:

- `LOW` = 0.0, 0.1, 0.2, 0.3, 0.4 — weak edit, strongest preservation
- `MID` = 0.5, 0.6 — moderate edit
- `HIGH` = 0.7, 0.8, 0.9, 1.0 — strong edit, more source override

Script: [run_qwen_vl_policy_balanced_short.py](run_qwen_vl_policy_balanced_short.py)

## Setup

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Dataset: full700 oracle (`policy_dataset_full700_sdturbo_wholepsnr_clipedit_perexample.csv`),
  labeled via whole-image PSNR + CLIP Edit with per-example normalized combined scores
- Prompting: 3 fixed in-context examples (one per bucket) embedded in the system message,
  each with a source image, source/target prompt, and a one-sentence reason
- Oracle bucket distribution: `low`=322, `mid`=210, `high`=168 (out of 700)

## Result

Output: `qwen_vl_policy_predictions_full700.csv`

- **Accuracy: 30.7%** (215/700 correct)

Confusion table (rows = oracle, columns = predicted):

| oracle | high | low | mid |
|---|---|---|---|
| high | 116 | 13  | 39 |
| low  | 209 | 33  | 80 |
| mid  | 125 | 19  | 66 |

Accuracy by oracle bucket:

| bucket | accuracy |
|---|---|
| high | 69.0% |
| low  | 10.2% |
| mid  | 31.4% |

Predicted bucket distribution: `high`=450, `mid`=185, `low`=65.

## Takeaways

- The model's predictions concentrate heavily on `HIGH` (450 of 700 predictions, 64%),
  regardless of the true bucket — it gets most `high`-truth examples right almost by
  default, but this same bias badly hurts it on `low` and `mid` examples (209 of 322
  true-`low` examples were predicted `high`).
- Accuracy (30.7%) is below the majority-class baseline (always predict `low`: 46.0%) —
  the model's bias toward `high` costs more than it gains from discriminating on the
  (majority) `low` class.
- Next steps: investigate why the model defaults to `high`, and whether prompt/example
  changes (e.g. more `low`/`mid` few-shot coverage) shift this bias.

## Ablation test: predicting a specific timestep instead of a bucket

Same setup as above, but instead of predicting `LOW`/`MID`/`HIGH`, the model predicts a
specific `t_start` value from the 11 discrete options: `0.0, 0.1, 0.2, ..., 1.0`.

Script: [run_qwen_vl_policy_balanced_short_timestep.py](run_qwen_vl_policy_balanced_short_timestep.py)
Model: `Qwen/Qwen3-VL-4B-Instruct`
Output: `qwen_vl_policy_predictions_full700_timestep.csv`

### Result

- **Exact-match accuracy: 10.0%** (70/700 correct)
- **Mean absolute error: 0.314** (on the 0.0–1.0 scale)

Predicted `t_start` distribution — heavily collapsed toward `0.7`/`0.9`:

| t_start | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| count | 51 | 5 | 39 | 4 | 41 | 33 | 338 | 29 | 159 | 1 |

71% of all 700 predictions were either `0.7` or `0.9`.

Exact-match accuracy by oracle bucket (oracle predicted the exact timestep rather than just the right bucket):

| bucket | accuracy |
|---|---|
| high | 30.4% |
| low  | 1.9% |
| mid  | 6.2% |

Explaination:
high = 30.4% — of the 168 examples whose true t_start is in 0.7-1.0, the model's exact prediction matched 30.4% of the time.
mid = 6.2% — of the 210 examples truly in 0.5-0.6, only 6.2% got an exact hit.
low = 1.9% — of the 322 examples truly in 0.0-0.4, almost none (1.9%) got an exact hit.

### Mapped back to buckets

Mapping the predicted `t_start` into `LOW`/`MID`/`HIGH` (same thresholds as above) for a
direct comparison to the bucket-prediction task:

- **Bucket accuracy: 28.3%** — below both the direct bucket-prediction run (30.7%) and the
  majority-class baseline (46.0%)

| oracle | low | mid | high |
|---|---|---|---|
| low  | 45  | 35  | 242 |
| mid  | 32  | 23  | 155 |
| high | 22  | 16  | 130 |

| bucket | accuracy |
|---|---|
| high | 77.4% |
| low  | 14.0% |
| mid  | 11.0% |

### Notes

- MAE (0.314) means predictions are off by roughly 3 buckets' worth on average, and the
  low/mid exact-match accuracy (1.9% / 6.2%) shows the model essentially never lands on the
  correct value when the true answer isn't already high.