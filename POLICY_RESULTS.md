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

**Balanced accuracy** (unweighted mean of the three per-bucket accuracies above, so
majority-class bias can't inflate the score — see "Balanced accuracy" note below):
**36.9%**.

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

### Note on "balanced accuracy"

Because the oracle bucket distribution is imbalanced (`low`=46.0%, `mid`=30.0%,
`high`=24.0% of 700), raw accuracy rewards models that happen to over-predict whichever
bucket is largest, even with no real discriminative signal. **Balanced accuracy**
corrects for this by weighting each bucket equally regardless of how often it appears.

**How it's calculated:**

1. For each oracle bucket `b`, compute that bucket's accuracy (i.e. recall): out of all
   examples whose *true* bucket is `b`, what fraction did the model predict as `b`?
   `acc_b = (# examples with oracle=b AND pred=b) / (# examples with oracle=b)`
   This is exactly the "Accuracy by oracle bucket" table shown for each model above —
   it only uses one row of the confusion table at a time (ignores off-diagonal mistakes
   between the other two buckets).
2. Average the three per-bucket accuracies **unweighted** — i.e. divide by 3, not by
   the number of examples in each bucket:
   `balanced_accuracy = (acc_low + acc_mid + acc_high) / 3`

**Worked example** (Qwen3-VL-4B-Instruct, original run, from the confusion table and
per-bucket accuracy table above):

- `acc_low`  = 33/322  = 10.25%  (of 322 true-`low` examples, 33 were predicted `low`)
- `acc_mid`  = 66/210  = 31.43%
- `acc_high` = 116/168 = 69.05%
- `balanced_accuracy = (10.25% + 31.43% + 69.05%) / 3 = 36.9%`

Note this is *not* the same as raw accuracy (30.7%), which instead divides the total
number of correct predictions (215) by the total number of examples (700) — so raw
accuracy implicitly weights each bucket by its support (322, 210, 168), while balanced
accuracy weights each bucket equally (1/3 each) regardless of support.

Because there are 3 equally-weighted buckets, a model that guesses uniformly at random
scores 33.3% balanced accuracy (vs. 46.0% for the raw-accuracy majority-class baseline)
— that 33.3% is the meaningful "no real signal" floor to compare against below.

## Model comparison: other VLMs on the same bucket-prediction task

Same setup as above (full700 dataset, 3 fixed in-context examples, same system prompt),
run with a generalized script that supports multiple model families.

Script: [run_vlm_policy_balanced_short.py](run_vlm_policy_balanced_short.py)

For reference, the majority-class baseline (always predict `low`) is **46.0%**; none of
the five models below reach it.

### Qwen3-VL-8B-Instruct

Output: `qwen_vl_8b_policy_predictions_full700_balanced_short.csv`

- **Accuracy: 30.0%** (210/700 correct)

| oracle | high | low | mid |
|---|---|---|---|
| high | 112 | 16 | 40 |
| low  | 210 | 38 | 74 |
| mid  | 124 | 26 | 60 |

| bucket | accuracy |
|---|---|
| high | 66.7% |
| low  | 11.8% |
| mid  | 28.6% |

Predicted bucket distribution: `high`=446, `mid`=174, `low`=80. Nearly identical bias
pattern to the 4B model above — scaling up within the Qwen3-VL family didn't help.

**Balanced accuracy: 35.7%**

### Qwen2.5-VL-7B-Instruct

Output: `qwen25vl_7b_policy_predictions_full700_balanced_short.csv`

- **Accuracy: 32.1%** (225/700 correct)

| oracle | high | low | mid |
|---|---|---|---|
| high | 88  | 28 | 52 |
| low  | 162 | 58 | 102 |
| mid  | 92  | 39 | 79 |

| bucket | accuracy |
|---|---|
| high | 52.4% |
| low  | 18.0% |
| mid  | 37.6% |

Predicted bucket distribution: `high`=342, `mid`=233, `low`=125 — the most balanced
predicted distribution of the five models, though still over-predicts `high`.

**Balanced accuracy: 36.0%**

### LLaVA-OneVision-7B (`llava-hf/llava-onevision-qwen2-7b-ov-hf`)

Output: `llava_ov_7b_policy_predictions_full700_balanced_short.csv`

- **Accuracy: 39.3%** (275/700 correct) — highest raw accuracy of the five models

| oracle | high | low | mid |
|---|---|---|---|
| high | 33 | 106 | 29 |
| low  | 79 | 207 | 36 |
| mid  | 44 | 131 | 35 |

| bucket | accuracy |
|---|---|
| high | 19.6% |
| low  | 64.3% |
| mid  | 16.7% |

Predicted bucket distribution: `low`=444, `high`=156, `mid`=100. This model's accuracy
"win" is mostly explained by over-predicting `low`, which happens to be the majority
oracle class (46%) — its `mid`/`high` recall are actually the worst of all five models.

**Balanced accuracy: 33.5%** — barely above the 33.3% random-guessing baseline, the
lowest of all five models despite having the highest raw accuracy.

### InternVL3-8B (`OpenGVLab/InternVL3-8B-hf`)

Output: `internvl_8b_policy_predictions_full700_balanced_short.csv`

- **Accuracy: 30.4%** (213/700 correct)

| oracle | high | low | mid |
|---|---|---|---|
| high | 37 | 11 | 120 |
| low  | 80 | 31 | 211 |
| mid  | 45 | 20 | 145 |

| bucket | accuracy |
|---|---|
| high | 22.0% |
| low  | 9.6%  |
| mid  | 69.0% |

Predicted bucket distribution: `mid`=476, `high`=162, `low`=62 — heavily over-predicts
`mid` instead.

**Balanced accuracy: 33.6%** — also barely above the 33.3% random-guessing baseline.

### Cross-model takeaways

Sorted by balanced accuracy — see the "Note on balanced accuracy" above for why raw
accuracy alone is misleading here.

| model | accuracy | balanced accuracy | dominant predicted bucket |
|---|---|---|---|
| Qwen3-VL-4B-Instruct (original) | 30.7% | **36.9%** | `high` (64.3%) |
| Qwen2.5-VL-7B-Instruct | 32.1% | **36.0%** | `high` (48.9%) |
| Qwen3-VL-8B-Instruct | 30.0% | **35.7%** | `high` (63.7%) |
| InternVL3-8B | 30.4% | **33.6%** | `mid` (68.0%) |
| LLaVA-OneVision-7B | 39.3% | **33.5%** | `low` (63.4%) |

(random-guessing baseline: 33.3% balanced accuracy, 46.0% raw accuracy)

- **Balanced accuracy flips the raw-accuracy ranking.** LLaVA-OneVision-7B has the best
  raw accuracy (39.3%) but the *worst* balanced accuracy (33.5%, barely above chance) —
  its raw-accuracy lead was entirely a byproduct of over-predicting `low`, the majority
  oracle class. The original Qwen3-VL-4B, which looked mediocre on raw accuracy, is
  actually the best model once each bucket is weighted equally.
- Every model collapses toward over-predicting one dominant bucket (Qwen3-VL 4B/8B and
  Qwen2.5-VL → `high`; LLaVA-OneVision → `low`; InternVL3 → `mid`), and **none beats the
  46.0% majority-class raw-accuracy baseline**.
- By balanced accuracy, InternVL3-8B and LLaVA-OneVision-7B are both barely above the
  33.3% random-guessing baseline — i.e. close to no real discriminative signal — while
  the three Qwen-family models retain modest (~3-4pp) signal above chance.
- Scaling Qwen3-VL from 4B to 8B did not change the bias pattern and slightly *lowered*
  balanced accuracy (36.9% → 35.7%), suggesting the bias comes from the prompting/task
  setup rather than model capacity.

## Ablation test: more in-context examples (10 vs 3)

Same setup as the original run (Qwen3-VL-4B-Instruct, full700 dataset, same system
prompt), but with 10 fixed in-context examples instead of 3 — 7 additional examples
were hand-curated from the full700 oracle (roughly balanced: 4 `LOW`, 3 `MID`, 3
`HIGH` total) and validated against their oracle bucket the same way as the original 3.

Script: [run_vlm_policy_balanced_short.py](run_vlm_policy_balanced_short.py) (`--num_prompt_examples 10`)
Output: `qwen_vl_4b_policy_predictions_full700_10examples.csv`

### Result

- **Accuracy: 31.0%** (217/700 correct)

| oracle | high | low | mid |
|---|---|---|---|
| high | 114 | 23 | 31 |
| low  | 204 | 58 | 60 |
| mid  | 127 | 38 | 45 |

| bucket | accuracy |
|---|---|
| high | 67.9% |
| low  | 18.0% |
| mid  | 21.4% |

Predicted bucket distribution: `high`=63.6%, `mid`=19.4%, `low`=17.0%.

**Balanced accuracy: 35.8%**

### Comparison to the 3-example baseline

| metric | 3 examples | 10 examples |
|---|---|---|
| Accuracy | 30.7% | 31.0% |
| Balanced accuracy | 36.9% | 35.8% |
| acc(low) | 10.2% | 18.0% |
| acc(mid) | 31.4% | 21.4% |
| acc(high) | 69.0% | 67.9% |
| Predicted `high` share | 64.3% | 63.6% |

### Takeaways

- **Adding 7 more in-context examples doesn't fix the bucket-collapse bias.** The model
  still predicts `high` for ~64% of all 700 examples regardless of true bucket, almost
  identical to the 3-example run.
- Raw accuracy is essentially flat (+0.3pp) and balanced accuracy is actually slightly
  *worse* (-1.1pp) — `low` recall improved (10.2% → 18.0%) but `mid` recall got worse
  (31.4% → 21.4%), so the extra examples mostly reshuffled which bucket gets confused
  with which rather than reducing the overall `high` bias.
- This suggests the bias is not a simple few-shot-coverage problem (i.e. not caused by
  only having one example per bucket) — more targeted prompt engineering, or a
  different task framing, is likely needed to shift it.

## Ablation test: in-context examples from the same edit category as the query

Hypothesis: instead of showing every query the same 3 fixed (LOW/MID/HIGH) examples
regardless of what kind of edit is being made, showing 3 examples drawn from the
query's own PIE-Bench edit category (`editing_type_name` — e.g. `change_style`,
`delete_object`) might give the model more relevant signal for that specific type of
edit.

**Approach:** for each query, dynamically pick one `LOW`, one `MID`, and one `HIGH`
example from rows sharing the query's own `editing_type_name` (excluding the query
itself), using the dataset's ground-truth category and bucket labels directly (no LLM
classification yet — that would be the next step if this helps). If a category is
missing examples in some bucket, backfill with additional examples from the category's
other buckets so the count stays at 3 (verified working via a synthetic test where a
bucket was artificially emptied). Reasons are auto-generated from each example's own
`editing_instruction` (e.g. `"Change the animal from a cat to a tiger. This is a
moderate edit within the 'change_object' category, so a mid timestep is
appropriate."`), since bespoke hand-written reasons aren't available for
dynamically-chosen examples. `random` was excluded from the evaluation set — it's a
grab-bag of edit types under one label, so it doesn't have a coherent "same category"
to match against (560 of 700 rows remain).

Script: [run_vlm_policy_balanced_short.py](run_vlm_policy_balanced_short.py)
(`--example_selection same_category --exclude_editing_types random`)
Model: `Qwen/Qwen3-VL-4B-Instruct`
Output: `qwen_vl_4b_policy_predictions_full700_same_category.csv`

### Result

- **Accuracy: 28.8%** (161/560 correct)

| oracle | high | low | mid |
|---|---|---|---|
| high | 81  | 15 | 40 |
| low  | 152 | 30 | 76 |
| mid  | 93  | 23 | 50 |

| bucket | accuracy |
|---|---|
| high | 59.6% |
| low  | 11.6% |
| mid  | 30.1% |

**Balanced accuracy: 33.8%**

### Comparison to the fixed-example baseline, restricted to the same 560 rows

Since this run excludes `random`, the fixed-3-example baseline is recomputed on the same
560-row subset for a fair comparison (rather than comparing against the original 700-row
number, which includes `random` rows this run never sees):

| oracle | high | low | mid |
|---|---|---|---|
| high | 89  | 12 | 35 |
| low  | 165 | 27 | 66 |
| mid  | 95  | 17 | 54 |

| metric | fixed (3 ex, 560 rows) | same-category (3 ex, 560 rows) |
|---|---|---|
| Accuracy | 30.4% (170/560) | 28.8% (161/560) |
| Balanced accuracy | 36.2% | 33.8% |
| acc(low) | 10.5% | 11.6% |
| acc(mid) | 32.5% | 30.1% |
| acc(high) | 65.4% | 59.6% |
| Predicted `high` share | 62.3% | 62.0%* |

*same-category predicted-bucket totals: high=326, mid=166, low=68 (from the confusion table columns above).

### Takeaways

- **Same-category examples performed worse than the generic fixed examples**, on both
  raw accuracy (28.8% vs. 30.4%) and balanced accuracy (33.8% vs. 36.2%) — the opposite
  of the hypothesis. It's also the worst balanced-accuracy result of any variant tried
  so far in this doc (below even the weakest of the 5-model comparison, InternVL3-8B at
  33.6%).
- The `HIGH`-collapse bias persists either way (~62% of predictions are `high` in both
  variants) — matching the query's edit category to the few-shot examples didn't reduce
  it.
- Possible confound: the same-category reasons are auto-generated and generic ("this is
  a moderate edit within the '{category}' category...") rather than the hand-written,
  more specific reasoning in the fixed pool's 3 examples. The drop in performance could
  reflect *weaker reasoning demonstrations* rather than category-matching itself being
  unhelpful — this run doesn't cleanly isolate the two.
- This was a "cheat" test using the ground-truth `editing_type_name` label directly; it
  was checked first because if category-matching doesn't even help with perfect category
  knowledge, there's no reason to build the more complex LLM-classifies-then-matches
  pipeline. Given the result, that next step doesn't look promising as currently
  designed — reason quality would need to be controlled for before concluding
  category-matching itself doesn't help.

## Ablation test: predicting a specific timestep instead of a bucket

Same setup as above, but instead of predicting `LOW`/`MID`/`HIGH`, the model predicts a
specific `t_start` value from the 11 discrete options: `0.0, 0.1, 0.2, ..., 1.0`.

Script: [run_qwen_vl_policy_balanced_short_timestep.py](run_qwen_vl_policy_balanced_short_timestep.py)
Model: `Qwen/Qwen3-VL-4B-Instruct`

Two versions of the system prompt were tried (v2 tweaked wording after seeing v1's
results); both used the same 3 fixed few-shot examples (file_ids `323000000007` /
`621000000004` / `613000000003`, reasons unchanged in meaning between versions, only
minor phrasing tweaks — e.g. "timestep (0.1) is appropriate" → "timestep of 0.1 is
appropriate").

### v1: original prompt

Output: `qwen_vl_policy_predictions_full700_timestep.csv`

```
You are predicting the best ChordEdit diffusion timestep t_start for an image edit.

Choose the t_start that best balances:
1. making the requested edit visible, and
2. preserving the original image.

t_start must be exactly one of: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
LOW range (0.0-0.4): weak edit, strongest preservation.
MID range (0.5-0.6): moderate edit.
HIGH range (0.7-1.0): strong edit, more source override.

Choose a low t_start when preservation likely matters more than forcing a strong edit.
Choose a mid t_start when both a weak and a strong edit seem plausible.
Choose a high t_start only when the edit would likely be weak or absent without strong source override.
```

**Result:**

- **Exact-match accuracy: 10.0%** (70/700 correct)
- **Mean absolute error: 0.314** (on the 0.0–1.0 scale)

Predicted `t_start` distribution — heavily collapsed toward `0.7`/`0.9`:

| t_start | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| count | 51 | 5 | 39 | 4 | 41 | 33 | 338 | 29 | 159 | 1 |

71% of all 700 predictions were either `0.7` or `0.9`.

Exact-match accuracy by oracle bucket (i.e. the model had to predict the exact timestep,
not just the right bucket):

| bucket | accuracy |
|---|---|
| high | 30.4% |
| low  | 1.9% |
| mid  | 6.2% |

- `high` = 30.4% — of the 168 examples whose true `t_start` is in 0.7-1.0, the model's
  exact prediction matched 30.4% of the time.
- `mid` = 6.2% — of the 210 examples truly in 0.5-0.6, only 6.2% got an exact hit.
- `low` = 1.9% — of the 322 examples truly in 0.0-0.4, almost none got an exact hit.

Mapped back to buckets (same LOW/MID/HIGH thresholds as the bucket task, for direct
comparison):

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

### v2: updated prompt

The `LOW`/`MID`/`HIGH` range hints were removed from the system prompt (leaving just the
list of 11 allowed values and the low/mid/high guidance sentences), and the HIGH guidance
sentence was reworded slightly. Also fixed to run pinned to a single GPU (`--gpu_id`)
instead of spreading across all 8 via `device_map="auto"`.

Output: `qwen_vl_policy_predictions_full700_timestep_v2.csv`

```
You are predicting the best ChordEdit diffusion timestep t_start for an image edit.

Choose the t_start that best balances:
1. making the requested edit visible, and
2. preserving the original image.

t_start must be exactly one of: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

Choose a low t_start when preservation likely matters more than forcing a strong edit.
Choose a mid t_start when both a weak and a strong edit seem plausible.
Choose a high t_start only when the edit would likely be too weak without strong source override.
```

**Result:**

- **Exact-match accuracy: 10.9%** (76/700 correct)
- **Mean absolute error: 0.298** (on the 0.0–1.0 scale)

Predicted `t_start` distribution — collapse point shifted from `0.7`/`0.9` (v1) to `0.6`/`0.9`:

| t_start | 0.1 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| count | 86 | 79 | 35 | 260 | 37 | 41 | 162 |

`0.0`, `0.2`, `0.4`, and `1.0` were never predicted at all (fewer distinct values used
than v1).

Exact-match accuracy by oracle bucket:

| bucket | v1 | v2 |
|---|---|---|
| high | 30.4% | 8.9% |
| mid  | 6.2%  | 23.8% |
| low  | 1.9%  | 3.4% |

Mapped back to buckets (same LOW/MID/HIGH thresholds):

- **Bucket accuracy: 33.9%** — above both v1's mapped result (28.3%) and the direct
  bucket-prediction run (30.7%), though still below the majority-class baseline (46.0%)

| oracle | low | mid | high |
|---|---|---|---|
| low  | 80  | 144 | 98 |
| mid  | 47  | 89  | 74 |
| high | 38  | 62  | 68 |

### Balanced accuracy (both versions)

As with the bucket-prediction task, raw exact-match accuracy can be misleading here since
the oracle `t_start` distribution is imbalanced across the 11 values. Balanced accuracy
(unweighted mean of per-class recall) removes that effect. Two granularities:

- **11-way** (over the 11 exact `t_start` classes; random-guess baseline ≈ 9.1%)
- **3-bucket** (over low/mid/high, using the mapped predictions; random-guess baseline ≈ 33.3%)

| metric | v1 | v2 |
|---|---|---|
| Raw exact-match accuracy | 10.0% | 10.9% |
| **Balanced accuracy (11-way)** | **9.4%** | **9.3%** |
| **Balanced accuracy (3-bucket)** | **12.8%** | **12.1%** |

### Takeaways

- Both versions' 11-way balanced accuracy (~9.3-9.4%) is essentially identical to the
  ~9.1% random-guessing baseline — neither prompt gives the model any real ability to
  discriminate the exact timestep from the image/edit content.
- The 3-bucket balanced accuracy (12-13%) is well *below* the 33.3% random-guessing
  baseline for 3 classes — worse than chance, because the model's narrow collapse onto 1-2
  values means it's systematically wrong on whatever bucket it isn't currently fixated on,
  in a way uniform random guessing wouldn't be.
- The prompt edit didn't fix the underlying collapse behavior — it just moved which values
  the model collapses onto (`0.7`/`0.9` → `0.6`/`0.9`), which is why raw accuracy and MAE
  barely moved (10.0%→10.9%, 0.314→0.298) while the per-bucket breakdown flipped (`high`
  accuracy dropped 30.4%→8.9%, `mid` accuracy rose 6.2%→23.8%).
- MAE improved marginally (0.314 → 0.298), consistent with the new collapse point (`0.6`)
  sitting closer to the middle of the 0.0–1.0 range than v1's (`0.7`/`0.9`).
- Fine-grained (11-way) timestep prediction does not appear usable with either prompt as
  currently written. Worth investigating whether this is fixable with prompting at all, or
  whether it requires more few-shot coverage of the underrepresented values, or is a more
  fundamental limitation of asking a 4B-class VLM for fine-grained numeric prediction from
  images.