# Does argmaxing the surrogate beat the fixed default `(0.8, 0.3)`?

## Overview

The pipeline has two stages. The surrogate **M̂** predicts the pair of quality metrics `(PSNR-Unedited, CLIP-Edited)` for an image, a source and target prompt, and a candidate timestep pair `(t_start, t_end)`. The selector **T** has no trainable weights: it queries M̂ at every cell of the quantized timestep grid, scalarizes each cell's predicted metrics into a single objective **phi**, and returns the argmax. This document asks one question — whether that procedure produces better timesteps than the fixed default `(0.8, 0.3)` — for the three objectives at `alpha = 2`.

Every table below is `DIR_NAME = UltraEdit_Region_10000`, `CHORD_EDIT_MODEL = sd_turbo`, and is labeled as such.

**Scope: `PHI_ALPHA = 2.0`.** Naive (alpha-free), LINEX alpha 2 and CARA alpha 2 span the curvature range from linear to strictly concave at the alpha the pipeline ships. An earlier version of this document covered `alpha = 5`; its conclusions do not carry over, and the contrast is discussed below.

**These numbers are computed on the full 121-cell grid.** The annotation pipeline has produced the 66 cells with `t_end ≥ t_start`, so the metrics CSV now covers all 11 × 11 positions rather than the 55 of the strict lower triangle. Two consequences had to be handled in the loader: 14 samples carry metrics *only* for the added cells and one has none at all, and `load_df` now drops samples whose labeled cell count is below the modal count — without that, per-sample deltas are undefined for those samples, `train_t` fails its shared-pair-set check, and the extra sample ids invalidate the packed embedding cache. With the filter the usable set is 9,985 samples split 7,988 / 998 / 999, exactly as before, so this is purely a widening of the candidate set from 55 cells to 121.

The setup is otherwise the recommended configuration from [SUMMARY.md](SUMMARY.md) — anchor on, dropout off, ranking weight 3, 20 epochs, `GRIDS_PER_BATCH = 32`, checkpointing on validation phi Spearman — retrained once per objective at `SEED = 42`, plus a four-seed repeat of each for spreads, with `SPLIT_SEED = 42` holding the split fixed. Selection is the raw argmax of the predicted phi grid: no gate, no shrinkage, no ensemble. Test split, 999 images. Runs are `outputs/UltraEdit_Region_10000/a2_{naive,linex_a2,cara_a2}_a2_s{42,1,2,3}`.

A factorial over the training and evaluation grids is reported below, and it moves the main conclusion: **training on the widened grid is itself what makes the selector inert.** A model trained on the original 55 cells deviates from the default on 42–47% of images and beats the 121-trained model's median regret by 14–35% under both concave objectives — whichever grid the two are then scored on.

**The headline is not that the surrogate picks badly — it is that under the concave objectives it barely picks at all.** Its predicted phi surface peaks at the default cell for 96.5% of images under LINEX alpha 2 and 88.3% under CARA alpha 2, so the selector leaves the default in place and neither helps nor harms by more than a few percent. Under naive it deviates on essentially every image and improves regret by about 60%, yet is still beaten by a single constant cell. The surrogate is therefore either inert or redundant, depending only on curvature.

## Terms

| Term | Definition |
| --- | --- |
| **phi** | The scalar objective T maximizes, computed from per-sample range-normalized metric deltas, `Δ_i = (s_i − s_i⁰) / (max_T s_i − min_T s_i)`, where `s_i⁰` is the metric at the default cell. Phi at the default cell is therefore exactly 0, and phi at any other cell is that cell's gain over the default. |
| **Naive** | `phi = Σ_i w_i Δ_i`. Linear and indifferent to how gains are split between the metrics; `alpha` does not enter. |
| **LINEX** | The average of Naive and CARA. Keeps CARA's superlinear regression penalty without its reward ceiling, at half the curvature. The shipped default. |
| **CARA** | `phi = (1/alpha) Σ_i w_i (1 − exp(−alpha Δ_i))`. Strictly concave: regressions are penalized without bound while each metric's reward saturates at `w_i / alpha` — 0.5 per metric at `alpha = 2`. |
| **regret** | `max_cell true phi − true phi at the selected cell`, per image. Reported as a median, a mean and a 90th percentile over the 999 test images; lower is better. |
| **rho** | Median per-image Spearman correlation between predicted and true phi across that image's grid — whether the whole surface is ordered correctly, not just its peak. |
| **oracle** | Selects each image's true argmax; regret 0 by construction. |
| **best fixed** | The argmax of the *train* split's mean phi surface, applied to every test image: the best strategy available with no model. |
| **deviate rate** | Fraction of images where a strategy picks something other than the default cell. |
| **new cells** | The 66 positions with `t_end ≥ t_start`, unlabeled until the recent annotation pass. |

Regret and phi are objective-specific, so values are comparable only *within* an objective — except the raw-metric table, which is comparable across all of them.

## Headline

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | 121 cells, 999 test images, four seeds per objective, `SPLIT_SEED = 42`:

| Objective | regret median | vs. default | regret mean | vs. default | phi gain mean | deviate rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive (alpha-free) | 0.1861 ± 0.0248 | **−64%** | 0.2206 ± 0.0171 | **−59%** | **+0.322 ± 0.017** | 99.9% |
| LINEX, alpha 2 (the default) | 0.4020 ± 0.0036 | −3% | 0.4161 ± 0.0050 | −4% | +0.017 ± 0.005 | 5.7% ± 1.6% |
| CARA, alpha 2 | 0.3208 ± 0.0052 | −6% | 0.3302 ± 0.0035 | −4% | +0.015 ± 0.004 | 14.3% ± 4.2% |

Unlike the `alpha = 5` results this replaces, every mean gain is positive and reliably so — 3.3 sigma for LINEX alpha 2 and 4.2 sigma for CARA alpha 2. Nothing here is harmful. But the concave rows improve regret by only 3–6%, and the reason is visible in the last column: the selector deviates from the default on 6% and 14% of images respectively. It is not making bad choices; it is declining to make choices.

Two facts frame the rest:

1. **Under naive, the model is redundant.** It improves median regret by 64%, and yet the single fixed cell `(0.1, 0.0)` is better still — 0.1437 against 0.2169 median and 0.1843 against 0.2430 mean at seed 42. Per-image selection is negative value added over one well-chosen constant.
2. **Under LINEX alpha 2 and CARA alpha 2, the model is inert.** A constant is again better — `(0.5, 0.2)` reaches 0.2993 median where the model reaches 0.4054 — because the model almost never leaves the default. When it does leave, it is right 80% and 69% of the time respectively, so the deviations it makes are good ones; there are just very few of them.

## Regret against the model-free references

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | single seed (42):

| Objective | Strategy | Cell | regret median | regret mean | regret p90 | rho |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Naive | default | `(0.8, 0.3)` | 0.5162 | 0.5423 | 0.9177 | — |
| Naive | best fixed | `(0.1, 0.0)` | **0.1437** | **0.1843** | **0.4277** | 0.5947 |
| Naive | surrogate argmax | per image | 0.2169 | 0.2430 | 0.4863 | **0.6392** |
| Naive | oracle | per image | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| LINEX a2 | default | `(0.8, 0.3)` | 0.4134 | 0.4327 | 0.7385 | — |
| LINEX a2 | best fixed | `(0.5, 0.2)` | **0.2993** | **0.3505** | **0.6961** | 0.4955 |
| LINEX a2 | surrogate argmax | per image | 0.4054 | 0.4224 | 0.7331 | **0.5849** |
| LINEX a2 | oracle | per image | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| CARA a2 | default | `(0.8, 0.3)` | 0.3419 | 0.3449 | 0.5715 | — |
| CARA a2 | best fixed | `(0.6, 0.3)` | **0.2619** | 0.3388 | 0.6698 | 0.4671 |
| CARA a2 | surrogate argmax | per image | 0.3263 | **0.3354** | **0.5600** | **0.6007** |
| CARA a2 | oracle | per image | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

The surrogate has the best rank correlation in every objective and the worst-but-one regret in two of them. That is the same tension `SUMMARY.md` documented — rho scores the whole surface, regret only the top cell — in its most extreme form yet: under LINEX alpha 2 the model orders the grid better than any constant possibly could (0.585 against 0.496) while landing within 3% of the default's regret.

CARA alpha 2 is the one place the surrogate wins something real: it has the lowest mean regret (0.3354) and the lowest p90 (0.5600) of all four strategies, beating even the best fixed cell, which pays for its better median (0.2619) with a much worse tail (0.6698).

These model-free rows are **identical** to those in the classification model's `RESULTS.md` — same default and best-fixed cells, same regret to four decimals — which confirms both packages are scoring the same 999-image split with the same phi, and makes the cross-model comparison at the end legitimate.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | surrogate quality behind those rows, seed 42:

| Objective | PSNR R² | CLIP R² | rho | top-1 hit | top-3 hit | median rank of pick (of 121) | cells used | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 0.829 | 0.389 | 0.6392 | 6.0% | 19.7% | 11 | 44 | 13 |
| LINEX a2 | 0.817 | 0.417 | 0.5849 | 1.6% | 4.8% | 40 | 11 | 10 |
| CARA a2 | 0.812 | 0.431 | 0.6007 | 2.6% | 6.7% | 34 | 16 | 9 |

Regression accuracy is essentially identical across objectives, as it must be — the surrogate predicts PSNR and CLIP, not phi, and only the ranking loss changes. Chance top-1 is `1 / 121 = 0.8%`. Note how little the concave rows clear that bar: 1.6% and 2.6%, against the 1.3% and 2.0% of images for which the default cell *is* the true best. Almost every top-1 "hit" under those objectives is an image where staying put happened to be correct.

## What each strategy achieves in real metric units

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | true PSNR-Unedited and CLIP-Edited at whichever cell each strategy picks, averaged over the 999 test images, seed 42. **This is the only table comparable across objectives.**

| Objective | Strategy | PSNR-Unedited | CLIP-Edited | vs. default |
| --- | --- | ---: | ---: | --- |
| any | default `(0.8, 0.3)` | 17.79 | 21.25 | — |
| Naive | best fixed `(0.1, 0.0)` | 31.55 | 16.94 | +13.76 dB, −4.31 CLIP |
| Naive | surrogate argmax | 27.75 | 18.77 | +9.96 dB, −2.48 CLIP |
| Naive | oracle | 26.44 | 22.66 | +8.65 dB, +1.41 CLIP |
| LINEX a2 | best fixed `(0.5, 0.2)` | 23.31 | 19.90 | +5.52 dB, −1.35 CLIP |
| LINEX a2 | surrogate argmax | 18.17 | 21.22 | +0.38 dB, −0.03 CLIP |
| LINEX a2 | oracle | 24.78 | 23.56 | +6.99 dB, +2.31 CLIP |
| CARA a2 | best fixed `(0.6, 0.3)` | 20.74 | 20.72 | +2.95 dB, −0.53 CLIP |
| CARA a2 | surrogate argmax | 19.15 | 21.12 | +1.36 dB, −0.13 CLIP |
| CARA a2 | oracle | 23.78 | 23.88 | +5.99 dB, +2.63 CLIP |

The inertness is unmistakable in real units: under LINEX alpha 2 the surrogate moves the average image by +0.38 dB and −0.03 CLIP — a rounding error — while its own oracle was there to collect +6.99 dB and +2.31 CLIP. Under naive it collects +9.96 dB by giving up 2.48 CLIP, which is what a linear phi with equal weights on range-normalized deltas asks for; the concave objectives refuse that trade, and every oracle row improves both metrics at once, so the trade was never necessary.

## Per-image win rate against the default

Because phi at the default is 0, a strategy's true phi at its picked cell *is* its gain over the default.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42:

| Objective | Strategy | win | tie | loss | gain median | gain mean | gain p10 | share of oracle gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | `(0.1, 0.0)` | 87.4% | 0.0% | 12.6% | +0.338 | +0.358 | −0.020 | **66%** |
| Naive | surrogate | 85.2% | 0.3% | 14.5% | +0.280 | +0.299 | −0.052 | 55% |
| LINEX a2 | `(0.5, 0.2)` | 67.4% | 0.0% | 32.6% | +0.145 | +0.082 | −0.351 | **19%** |
| LINEX a2 | surrogate | 2.9% | 96.5% | 0.6% | 0.000 | +0.010 | 0.000 | 2% |
| CARA a2 | `(0.6, 0.3)` | 62.2% | 0.0% | 37.8% | +0.085 | +0.006 | −0.415 | 2% |
| CARA a2 | surrogate | 7.9% | 88.3% | 3.8% | 0.000 | +0.010 | 0.000 | **3%** |

The tie column is the story. Under the concave objectives the surrogate returns the default cell for 88–97% of images, so its gain distribution is a spike at zero: it cannot lose much and cannot win much. Note that its p10 is exactly 0.000 there, against −0.351 and −0.415 for the best fixed cell — the constant is the risky strategy and the model is the safe one, which is the reverse of the usual framing.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | conditioned on the images the surrogate actually moves, four seeds:

| Objective | images moved (seed 42) | win | loss | gain median | gain mean | gain p10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 997 of 999 | 85.4% ± 0.2% | 14.6% | +0.280 | +0.322 ± 0.017 | −0.052 |
| LINEX a2 | 35 of 999 | 79.9% ± 3.1% | 20.1% | +0.329 | +0.293 ± 0.042 | −0.134 |
| CARA a2 | 117 of 999 | 69.0% ± 1.0% | 31.0% | +0.164 | +0.107 ± 0.031 | −0.427 |

**The deviations the surrogate does make are excellent.** Under LINEX alpha 2 the 35 images it moves gain +0.293 phi on average — three and a half times the best fixed cell's +0.082 across all images — with an 80% win rate. Scaled to the whole split those 35 images contribute the +0.010 mean gain in the table above. The model knows where to go; it will not go there.

## Why the selector stays at the default

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42. The mean phi surface is averaged over the 999 test images, and the metric ranges are per-grid max minus min, averaged over images:

| Objective | mean TRUE surface argmax (value) | mean PRED surface argmax (value) | mean per-image max phi, true | pred | PSNR range, true → pred | CLIP range, true → pred | improves both metrics: model | oracle |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| Naive | `(0.1, 0.0)` +0.358 | `(0.2, 0.0)` +0.690 | +0.4327 | **+0.794** | 20.32 → 14.20 | 12.87 → 3.76 | 25.7% | 58.9% |
| LINEX a2 | `(0.5, 0.2)` +0.082 | **`(0.8, 0.3)` +0.000** | +0.4327 | **+0.004** | 20.32 → 15.37 | 12.87 → 5.59 | 1.3% | 70.0% |
| CARA a2 | `(0.6, 0.2)` +0.007 | **`(0.8, 0.3)` +0.000** | +0.3449 | **+0.017** | 20.32 → 15.98 | 12.87 → 5.95 | 3.6% | 80.9% |

Under both concave objectives the surrogate's *average* predicted phi surface peaks exactly at the default cell, at exactly zero: the model believes no timestep pair beats `(0.8, 0.3)` on average. Per image it believes there is +0.004 (LINEX) or +0.017 (CARA) of phi available where the truth holds +0.433 and +0.345 — it recovers 1% and 5% of the real headroom. There is nothing for an argmax to find.

Part of the mechanism is metric-range compression, and it falls hardest on CLIP. The surrogate reproduces 70–79% of the true PSNR range within a grid but only 29–46% of the true CLIP range, consistent with `R²` of about 0.82 for PSNR against 0.42 for CLIP. Because phi normalizes each metric by its own predicted range, a compressed *and* smoothly monotone predicted surface makes every alternative to the default look like a pure trade of one metric for the other — and a concave phi refuses trades. The true surfaces are not monotone: they have interior cells where both metrics improve together, which the oracle finds for 70% and 81% of images while the surrogate finds them for 1.3% and 3.6%. **Failing to reproduce the both-improve region is what makes the selector inert**, and it is a prediction-side failure, not a selection-rule failure. Compression alone does not explain the magnitude, though: the factorial below shows the same compression ratios in a model that deviates on 43% of images, so the shape of the predicted surface matters more than its scale.

Under naive the same compression is harmless: with no penalty asymmetry the surviving PSNR signal dominates, the predicted surface peaks near the true one, and the model over-estimates the available phi (+0.794 against +0.433) — so it always deviates.

## Training grid versus evaluation grid

The 66 new cells can enter the experiment in two independent places: the cells the surrogate trains on, and the cells T is allowed to pick from. Crossing the two isolates whether the addition helps the model or merely widens the target. `CELL_REGION` selects the region at load time; `analyze_select.py --cell-region` scores a checkpoint over the other one, so conditions B and C share a checkpoint and differ only in what it is asked to rank.

| Condition | Train cells | Test cells | What it isolates |
| --- | --- | --- | --- |
| **A** | 121 | 121 | The configuration this document reports above. |
| **B** | 55 | 121 | Extrapolation: 66 of the cells T may pick were never labeled during training. |
| **C** | 55 | 55 | The pre-annotation world. Reproduces the historical runs bit-for-bit. |
| **D** | 121 | 55 | Whether training on the extra cells helps on the original grid. |

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | four seeds per cell, `SPLIT_SEED = 42`, `PHI_ALPHA = 2.0`:

| Objective | Condition | Train → Test | regret median | regret mean | phi gain mean | share of oracle gain | deviate rate | rho |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | A | 121 → 121 | **0.1861 ± 0.0248** | **0.2206 ± 0.0171** | +0.322 ± 0.017 | 59% | 99.9% | **0.633** |
| Naive | B | 55 → 121 | 0.2205 ± 0.0363 | 0.2675 ± 0.0295 | +0.275 ± 0.030 | 51% | 99.7% | 0.384 |
| Naive | C | 55 → 55 | **0.1794 ± 0.0121** | **0.2143 ± 0.0093** | +0.332 ± 0.009 | 61% | 99.6% | 0.618 |
| Naive | D | 121 → 55 | 0.1845 ± 0.0279 | 0.2176 ± 0.0216 | +0.328 ± 0.022 | 60% | 99.8% | 0.611 |
| LINEX a2 | A | 121 → 121 | 0.4020 ± 0.0036 | 0.4161 ± 0.0050 | +0.017 ± 0.005 | 4% | 5.7% | **0.585** |
| LINEX a2 | B | 55 → 121 | **0.2979 ± 0.0100** | **0.3474 ± 0.0072** | **+0.085 ± 0.007** | **20%** | 61.4% | 0.358 |
| LINEX a2 | C | 55 → 55 | **0.3067 ± 0.0094** | **0.3449 ± 0.0077** | **+0.089 ± 0.008** | **21%** | 47.3% | 0.531 |
| LINEX a2 | D | 121 → 55 | 0.4050 ± 0.0056 | 0.4216 ± 0.0049 | +0.012 ± 0.005 | 3% | 4.8% | 0.522 |
| CARA a2 | A | 121 → 121 | 0.3208 ± 0.0052 | 0.3302 ± 0.0035 | +0.015 ± 0.004 | 4% | 14.3% | **0.584** |
| CARA a2 | B | 55 → 121 | **0.2814 ± 0.0092** | **0.3287 ± 0.0125** | +0.016 ± 0.013 | 5% | 45.9% | 0.379 |
| CARA a2 | C | 55 → 55 | **0.2779 ± 0.0043** | 0.3328 ± 0.0058 | +0.011 ± 0.006 | 3% | 41.1% | 0.516 |
| CARA a2 | D | 121 → 55 | 0.3221 ± 0.0077 | 0.3367 ± 0.0011 | +0.008 ± 0.001 | 2% | 12.9% | 0.511 |

**Read this table down the columns that share an evaluation grid, never across them.** Phi is normalized over whatever candidate set is present and the max in `regret` is taken over that set, so a regret of 0.30 on 55 cells and 0.30 on 121 cells are different quantities. A and B are comparable to each other; C and D are comparable to each other; the achieved metric units below are comparable everywhere.

The model-free references for each evaluation grid, seed 42, put those numbers in context:

| Test cells | Objective | default regret med / mean | best fixed cell | best fixed regret med / mean |
| --- | --- | ---: | --- | ---: |
| 121 | Naive | 0.5162 / 0.5423 | `(0.1, 0.0)` | 0.1437 / 0.1843 |
| 121 | LINEX a2 | 0.4134 / 0.4327 | `(0.5, 0.2)` | 0.2993 / 0.3505 |
| 121 | CARA a2 | 0.3419 / 0.3449 | `(0.6, 0.3)` | 0.2619 / 0.3388 |
| 55 | Naive | 0.5186 / 0.5458 | `(0.1, 0.0)` | 0.1375 / 0.1781 |
| 55 | LINEX a2 | 0.4198 / 0.4338 | `(0.5, 0.2)` | 0.3009 / 0.3663 |
| 55 | CARA a2 | 0.3422 / 0.3442 | `(0.8, 0.3)` = default | 0.3422 / 0.3442 |

### Training on the widened grid is what made the selector inert

Holding the evaluation grid fixed, training on 121 cells is **worse** than training on 55 for both concave objectives, and by a wide margin:

| Objective | Test grid | train 55 | train 121 | change |
| --- | --- | ---: | ---: | ---: |
| LINEX a2 | 121 cells | **0.2979** | 0.4020 | +35% regret from training on 121 |
| LINEX a2 | 55 cells | **0.3067** | 0.4050 | +32% |
| CARA a2 | 121 cells | **0.2814** | 0.3208 | +14% |
| CARA a2 | 55 cells | **0.2779** | 0.3221 | +16% |
| Naive | 121 cells | 0.2205 | **0.1861** | −16% (training on 121 helps) |
| Naive | 55 cells | 0.1794 | 0.1845 | +3% (a wash) |

Under naive, matched train/test wins and everything is within a few percent. Under both concave objectives, training on the extra cells costs 14–35% of median regret **whichever grid the model is then scored on** — so this is not a train/test mismatch effect, it is damage done during training. The deviate rate shows the same thing more directly: on the 55-cell grid the 55-trained model leaves the default on 47.3% of images (LINEX) where the 121-trained model manages 4.8%.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42, each model on its own training region:

| Objective | Train cells | true max phi | pred max phi | pred / true | PSNR range, true → pred | CLIP range, true → pred | deviate rate |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| Naive | 55 | +0.5458 | +0.9782 | 179% | 18.91 → 13.53 (72%) | 11.34 → 3.78 (33%) | 99.8% |
| Naive | 121 | +0.5423 | +0.7943 | 146% | 20.32 → 14.20 (70%) | 12.87 → 3.76 (29%) | 99.7% |
| LINEX a2 | 55 | +0.4338 | **+0.0711** | 16% | 18.91 → 14.37 (76%) | 11.34 → 4.93 (43%) | 42.9% |
| LINEX a2 | 121 | +0.4327 | **+0.0036** | 0.8% | 20.32 → 15.37 (76%) | 12.87 → 5.59 (43%) | 3.5% |
| CARA a2 | 55 | +0.3442 | **+0.0731** | 21% | 18.91 → 15.24 (81%) | 11.34 → 5.39 (48%) | 39.0% |
| CARA a2 | 121 | +0.3449 | **+0.0170** | 4.9% | 20.32 → 15.98 (79%) | 12.87 → 5.95 (46%) | 11.7% |

The true headroom is the same in both worlds — +0.4338 against +0.4327 for LINEX — but the model's belief about it differs twentyfold: trained on 55 cells it predicts +0.0711 of available phi per image, trained on 121 it predicts +0.0036. That, and not any change in the data, is what drops the deviate rate from 42.9% to 3.5%.

**Range compression is not the explanation.** Predicted PSNR spans 76% of the true per-grid range under both training regions, and predicted CLIP 43% under both — the compression documented in the previous section is real but identical across conditions, so it cannot account for a twentyfold difference in predicted phi. Since phi is computed after per-grid normalization, which divides out scale, what must differ is the *shape* of the predicted surface: trained on all 121 cells, the model places the default at or beside its own joint optimum for nearly every image, so every alternative reads as a pure metric trade and a concave phi rejects it. Trained on 55, it does not. The most likely cause is the character of the added region — the `t_end ≥ t_start` cells are the extreme high-PSNR, low-CLIP corner, and a ranking loss that has to order them correctly is being pulled toward reproducing one monotone gradient rather than the interior structure where both metrics improve together.

**Condition B's advantage should be treated with suspicion even so.** Its whole-surface rank correlation is much the worse of the two (0.358 against 0.585), which is what one expects of a model ranking 66 cells it never saw, and it lands 14.9% of its picks in the new region against the 121-trained model's 2.0% despite that. It is not that the 55-trained model understands the new cells; its extrapolation over-values them, and under concave phi being pushed off the default is worth more than being right about where to go. Under naive, where the model already deviates on every image, the same optimism overshoots: B puts 55.8% of its picks in the new region against the oracle's 27.6% and its regret is the worst of the four conditions. **C against D is the trustworthy comparison** — both models scored only on cells they trained on, differing solely in whether the extra cells were present during training — and it points the same way.

## Which cells get picked

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42, of 121 cells:

| Objective | Strategy | distinct cells | deviate rate | most-picked cells |
| --- | --- | ---: | ---: | --- |
| Naive | surrogate | 44 | 99.7% | `(0.3, 0.0)` 20.9%, `(0.3, 0.1)` 14.6%, `(0.2, 0.0)` 8.7%, `(0.1, 0.0)` 7.5% |
| Naive | oracle | 84 | 99.4% | `(0.0, 0.0)` 14.9%, `(0.1, 0.0)` 11.0%, `(0.2, 0.0)` 7.6%, `(0.3, 0.0)` 6.0% |
| LINEX a2 | surrogate | 11 | 3.5% | `(0.8, 0.3)` 96.5%, `(0.0, 0.0)` 0.7%, `(0.0, 0.1)` 0.6% |
| LINEX a2 | oracle | 89 | 98.7% | `(0.0, 0.0)` 9.8%, `(0.1, 0.0)` 6.6%, `(0.3, 0.0)` 4.6%, `(0.2, 0.0)` 4.2% |
| CARA a2 | surrogate | 16 | 11.7% | `(0.8, 0.3)` 88.3%, `(0.3, 0.0)` 3.6%, `(0.0, 0.0)` 2.0%, `(0.2, 0.2)` 1.1% |
| CARA a2 | oracle | 92 | 98.0% | `(0.0, 0.0)` 5.0%, `(0.4, 0.2)` 4.4%, `(0.5, 0.2)` 3.8%, `(0.1, 0.0)` 3.7% |

Under naive the surrogate uses 44 of 121 cells against the oracle's 84 and sits in the same low-`t_start` region, just less deep into it — its modal pick is `(0.3, 0.0)` where the oracle's is `(0.0, 0.0)`. Under the concave objectives it collapses onto the default and uses 11 and 16 cells in total.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | picks landing in the 66 cells with `t_end ≥ t_start`, seed 42:

| Objective | oracle in new cells | surrogate in new cells | oracle on `(0.0, 0.0)` | best new cell's mean phi | as share of oracle gain | best old cell, as share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 27.6% | 11.9% | 14.9% | +0.420 | **77%** | 94% |
| LINEX a2 | 26.7% | 2.0% | 9.8% | +0.287 | 66% | 94% |
| CARA a2 | 25.6% | 4.4% | 5.0% | +0.189 | 55% | 94% |

The new region alone captures 55–77% of the oracle's gain and the original 55 cells alone capture 94%, so the two overlap heavily rather than the addition being a separate prize — but `(0.0, 0.0)`, which did not exist before the annotation pass, is the oracle's single most-picked cell under all three objectives. No objective's best fixed cell lies in the new region. The surrogate reaches it 2–12% of the time against the oracle's 26–28%, so it under-exploits the addition everywhere, most severely where it is most inert.

## The gate has the wrong sign

`NOISE_FLOOR_PHI` keeps the default unless the predicted gain clears a floor, applied as an actual selection rule.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42:

| Floor | Naive deviate | Naive gain mean | LINEX a2 deviate | LINEX a2 gain mean | CARA a2 deviate | CARA a2 gain mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 99.7% | **+0.299** | 3.5% | **+0.010** | 11.7% | **+0.010** |
| 0.05 | 98.9% | +0.298 | 1.9% | +0.007 | 8.8% | +0.007 |
| 0.1 | 97.4% | +0.296 | 1.2% | +0.004 | 6.8% | +0.004 |
| 0.2 | 93.2% | +0.291 | 0.6% | +0.001 | 3.6% | +0.003 |
| 0.3 | 87.0% | +0.284 | 0.4% | +0.000 | 1.3% | +0.002 |
| 0.5 | 71.9% | +0.246 | 0.0% | 0.000 | 0.0% | 0.000 |
| 1.0 | 31.6% | +0.123 | 0.0% | 0.000 | 0.0% | 0.000 |

Every floor above zero makes things worse, monotonically, in all three objectives — it removes deviations that were profitable on average. **Keep `NOISE_FLOOR_PHI = 0.0`.** More usefully, this says the knob the concave objectives need is the opposite one: a *negative* floor, or any recalibration that lets a predicted gain of +0.004 count as a reason to move. The gate as built can only suppress selection, and suppression is not this model's problem.

## Against the classifier on the same question

The classification model's `RESULTS.md` answers the identical question on the identical split, objective and default cell, so these rows are directly comparable — with the caveat that it is a different model class trained on a different signal (one labeled cell per sample against the surrogate's whole grid).

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | four seeds each:

| Objective | classifier gain mean | surrogate gain mean | classifier deviate | surrogate deviate |
| --- | ---: | ---: | ---: | ---: |
| Naive | +0.277 ± 0.010 | **+0.322 ± 0.017** | 99.5% | 99.9% |
| LINEX a2 | **+0.104 ± 0.004** | +0.017 ± 0.005 | 98.8% | 5.7% |
| CARA a2 | −0.014 ± 0.006 | **+0.015 ± 0.004** | 98.9% | 14.3% |

The two models fail in opposite directions. The classifier names a cell for ~99% of images under every objective, which pays under LINEX alpha 2 (+0.104, six times the surrogate's gain) and costs it the sign under CARA alpha 2, where its confident deviations run into the exponential penalty. The surrogate abstains by default, which protects it under CARA alpha 2 — the one place it is the better model — and wastes almost all of the LINEX alpha 2 opportunity. **The classifier cannot abstain; the surrogate cannot commit.** Under naive, where the answer is a corner of the grid and neither abstention nor commitment matters much, they land within 0.045 of each other and both lose to a constant.

## What this says

- **Training on the 66 added cells is what silenced the selector.** Under both concave objectives, a model trained on the original 55 cells has 14–35% lower median regret than one trained on all 121, on either evaluation grid, and deviates from the default an order of magnitude more often (47.3% against 4.8% for LINEX alpha 2 on the 55-cell grid). The true headroom is unchanged between the two worlds; only the model's belief about it collapses, from +0.071 predicted phi per image to +0.004.
- **The surrogate is not misranking the grid; it is under-predicting the prize.** Its phi Spearman is the best of any strategy in all three objectives (0.585–0.639), and its rare deviations win 69–85% of the time. What it gets wrong is the *scale* of the available gain: it recovers 1% of the true headroom under LINEX alpha 2 and 5% under CARA alpha 2, so its argmax has no reason to move.
- **That under-prediction is a CLIP-side regression failure.** Predicted CLIP spans 29–46% of the true per-grid range against PSNR's 70–79%, and the both-improve cells the oracle relies on for 70–81% of images are found by the surrogate for 1–4%. Fixing CLIP is the whole of the remaining headroom; no change to the selection rule can substitute.
- **Under naive, one constant beats the model.** `(0.1, 0.0)` reaches 0.1437 median regret against the surrogate's 0.2169 and captures 66% of the oracle's gain against 55%. When phi is linear enough to want a grid corner, naming that corner once is better than predicting it per image — the same conclusion the classifier reached.
- **CARA alpha 2 is the one row where the surrogate is the best available strategy on the mean** (0.3354, against 0.3388 for the best fixed cell, 0.3449 for the default and −0.014 gain for the classifier). It wins by being cautious, not by being accurate, which is worth having but is not the mechanism anyone intended.
- **Curvature still decides everything, but not in the direction `alpha = 5` suggested.** At `alpha = 5` this model deviated on 29–34% of images and lost on the mean (−0.202 CARA, −0.042 LINEX); at `alpha = 2` it deviates on 6–14% and gains slightly. Higher curvature made it bolder *and* wrong; lower curvature makes it timid and harmlessly right. Both are failures of calibration between predicted and true phi rather than of the argmax rule.

## Caveats

- One split (`SPLIT_SEED = 42`). Headline spreads are four initialization seeds; the reference, breakdown and diagnostic tables are single-seed (42).
- The configuration was tuned in `SUMMARY.md` against LINEX at `alpha = 2` **on the 55-cell grid and against the `(0.9, 0.3)` default**, and is reused unchanged here for all three objectives on the 121-cell grid against `(0.8, 0.3)`. Whether those choices — the ranking weight above all — survive the wider grid is untested and cheap to test.
- `SUMMARY.md` and `CHANGES.md` still describe the 55-cell grid and have not been recomputed. `SUMMARY.md`'s numbers are additionally against the `(0.9, 0.3)` default, so nothing in it is directly comparable to this document.
- These are single models. `SUMMARY.md` found ensembling worth about 0.015 regret and validation-tuned shrinkage another 0.008, neither of which is applied here; shrinkage in particular pulls predictions toward the mean surface, which under the concave objectives would make an already-inert selector more so.
- The factorial's B condition asks a model to rank 66 cells it never saw a label for, and its per-cell anchor has no entry for them, so it falls back to the nearest known cell. Its better regret comes with a much worse whole-surface rho (0.358 against 0.585) and over-selection of the new region, so it should be read as evidence that being pushed off the default helps, not that extrapolation works. C against D is the clean comparison.
- The `alpha = 5` figures quoted for contrast come from the previous version of this document, on the 55-cell grid; only their sign and rough magnitude should be relied on.

## Worth trying next

1. **Calibrate predicted phi against true phi.** The model's ordering is good and its scale is not, which is exactly the failure a monotone recalibration fixes. Fitting a per-objective map from predicted to true phi gain on validation — or simply allowing a negative `NOISE_FLOOR_PHI` — would let the concave objectives act on the deviations they already rank correctly. This is the cheapest experiment with the largest expected effect.
2. **Train on the lower triangle and select over the full grid, or exclude the degenerate corner from training.** This is the largest single effect measured here and it costs nothing to adopt: `CELL_REGION = "lower"` at training time, scored over all 121 cells, gives LINEX alpha 2 a median regret of 0.2979 against the 0.4020 of the shipped configuration. Before adopting it, check whether restricting training to a *sensible* subset — dropping only the degenerate `t_end ≥ t_start` cells rather than everything new — recovers the same effect for a better reason.
3. **Attack the CLIP head.** Predicted CLIP covers under half the true per-grid range, and that compression is what erases the both-improve region. Text conditioning richer than two mean-pooled CLIP vectors is the obvious lever, and `SUMMARY.md` already flagged it.
4. **Train the ranking loss on the objective's own curvature.** Nothing tells the model that a concave phi needs a well-resolved both-improve region; a loss weighted by true phi gap rather than pairwise order would.
5. **Re-run the `SUMMARY.md` sweep on the 121-cell grid against `(0.8, 0.3)`.** Both the grid and the baseline cell have changed since the recommended configuration was chosen; the anchor and ranking weight in particular were tuned to a different surface.

## Summary

`DIR_NAME = UltraEdit_Region_10000`, `CHORD_EDIT_MODEL = sd_turbo`, 121-cell grid, 999 test images, default cell `(0.8, 0.3)`, `PHI_ALPHA = 2.0`.

Argmaxing the surrogate beats the fixed default under all three objectives at `alpha = 2`, and for the first time in this line of work never harms it: mean phi gain is +0.322 under naive, +0.017 under LINEX alpha 2 and +0.015 under CARA alpha 2, the latter two positive at 3–4 sigma. But the two concave objectives improve regret by only 3–6%, because the selector leaves the default in place on 86–94% of images. The diagnostic is unambiguous: the surrogate's mean predicted phi surface peaks at the default cell at exactly zero, and per image it predicts 1–5% of the headroom that actually exists, because it reproduces only 29–46% of the true CLIP range and therefore almost never finds the cells where both metrics improve together — 1.3% of images against the oracle's 70.0% under LINEX alpha 2. Its ranking is the best of any strategy tested (rho 0.585–0.639) and the deviations it does make gain +0.29 phi at an 80% win rate, so the model knows where to go and will not go there. Against a single constant it loses under naive and LINEX alpha 2 and wins on the mean under CARA alpha 2, where caution is worth more than accuracy. The factorial over training and evaluation grids locates the cause more precisely than calibration alone would: training on the 66 newly labeled cells is what collapses the predicted headroom. A model trained on the original 55 cells predicts +0.071 of available phi per image instead of +0.004, deviates on 42–47% of images instead of 4–6%, and has 14–35% lower median regret under both concave objectives no matter which grid it is scored on, while the true headroom is identical in both worlds. Range compression is equal across the two, so what the extra cells change is the shape of the predicted surface, not its scale. The next moves are therefore to train on the lower triangle while selecting over the full grid, to recalibrate predicted phi against true phi, and to fix the CLIP head that compresses it.
