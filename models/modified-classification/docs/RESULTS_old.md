# Trained on the original 55 cells, selecting over all 121

## Overview

This is the companion to [RESULTS.md](RESULTS.md) for one configuration: the surrogate **M̂** trained only on the 55 cells of the strict lower triangle `t_start > t_end` — the grid as the annotation pipeline had it before it filled in the rest — with the selector **T** then allowed to pick from all 121 cells. It is condition **B** of the factorial in `RESULTS.md`, and it exists because that factorial found something unexpected: training on the widened grid is what makes the selector inert, so the model trained on the *old* data is the better selector under both concave objectives.

Every table below is `DIR_NAME = UltraEdit_Region_10000`, `CHORD_EDIT_MODEL = sd_turbo`, and is labeled as such.

**What is being asked of the model here.** Sixty-six of the 121 cells T may choose were never labeled during training. The surrogate takes `(t_start, t_end)` as a continuous input through Fourier features, so it will happily return a prediction for them, but nothing constrained those predictions; and the per-cell anchor — the train split's mean standardized value at each grid cell — has no entry for the new cells and falls back to the nearest cell it does know. This is extrapolation, and it should be read as such: where this configuration wins, the interesting question is *why* a model that never saw those cells outperforms one that did.

Setup is otherwise identical to `RESULTS.md`: `PHI_ALPHA = 2.0`, default cell `(0.8, 0.3)`, the recommended configuration from [SUMMARY.md](SUMMARY.md) (anchor on, dropout off, ranking weight 3, 20 epochs), `SPLIT_SEED = 42`, four initialization seeds per objective, 999 test images, raw argmax with no gate, shrinkage or ensemble. Runs are `outputs/UltraEdit_Region_10000/lo_{naive,linex,cara}_lo_s{42,1,2,3}`, scored with `analyze_select.py --cell-region all`, which writes `t_analysis_evalall.json`.

Training used `CELL_REGION = "lower"`, which keeps 549,175 of 1,209,109 cell rows and reproduces the pre-annotation dataset exactly — 7,988 / 998 / 999 samples at 55 cells each. The naive run at seed 42 reproduces the historical 55-cell numbers bit-for-bit (test regret 0.1826, rho 0.6206, `R²` 0.782 / 0.334), which is the check that this really is the old world.

**The result in one line: on the mean, this is the best configuration measured for either concave objective** — better than the fixed default, better than the best constant cell, and better than the surrogate trained on all 121 cells. On the median it is a close second to the best constant. Under naive it is the worst of the four factorial conditions.

## Terms

Definitions are as in [RESULTS.md](RESULTS.md): **phi** is the scalar objective built from per-sample range-normalized metric deltas against the default cell, so phi at the default is exactly 0 and phi at a picked cell is that cell's gain over the default; **regret** is `max_cell true phi − true phi at the picked cell`; **rho** is the median per-image Spearman correlation between predicted and true phi; **best fixed** is the train split's mean-phi argmax; **oracle** picks each image's true best cell. Two terms matter more here than there:

| Term | Definition |
| --- | --- |
| **new cells** | The 66 positions with `t_end ≥ t_start`. Unlabeled during training in this configuration, but selectable at test time. |
| **deviate rate** | Fraction of images where the model picks something other than the default cell. It is the quantity this configuration changes most. |

All phi values here are normalized over the 121-cell candidate set, so they are comparable to the `test 121` rows of `RESULTS.md` and **not** to anything scored on 55 cells.

## Headline

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | trained on 55 cells, scored over 121, four seeds per objective, `SPLIT_SEED = 42`:

| Objective | regret median | vs. default | regret mean | vs. default | phi gain mean | deviate rate | rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive (alpha-free) | 0.2205 ± 0.0363 | **−57%** | 0.2675 ± 0.0295 | **−51%** | **+0.275 ± 0.030** | 99.7% | 0.384 |
| LINEX, alpha 2 (the default) | 0.2979 ± 0.0100 | **−28%** | 0.3474 ± 0.0072 | **−20%** | **+0.085 ± 0.007** | 61.4% ± 7.2% | 0.358 |
| CARA, alpha 2 | 0.2814 ± 0.0092 | **−18%** | 0.3287 ± 0.0125 | −5% | +0.016 ± 0.013 | 45.9% ± 1.9% | 0.379 |

Against the same rows of the 121-trained model in `RESULTS.md` — 0.4020 median and +0.017 gain for LINEX alpha 2, 0.3208 and +0.015 for CARA alpha 2 — the concave objectives improve by 26% and 12% on median regret, and LINEX alpha 2's mean phi gain rises fivefold. The mechanism is in the deviate-rate column: this model leaves the default on 61% and 46% of images where the 121-trained one manages 6% and 14%.

Two things temper it. The rank correlation is much worse — 0.358 to 0.384 against 0.585 to 0.633 — which is exactly what a model ranking 66 unseen cells should look like. And CARA alpha 2's mean gain, +0.016 ± 0.013, is only 1.3 sigma from zero here, where the 121-trained model's smaller +0.015 ± 0.004 was 3.8 sigma: this configuration buys a better median at the cost of a noisier mean.

## Regret against the model-free references

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | single seed (42), all strategies scored over the same 121 cells:

| Objective | Strategy | Cell | regret median | regret mean | regret p90 | rho |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Naive | default | `(0.8, 0.3)` | 0.5162 | 0.5423 | 0.9177 | — |
| Naive | best fixed | `(0.1, 0.0)` | **0.1437** | **0.1843** | **0.4277** | **0.5947** |
| Naive | surrogate argmax | per image | 0.2283 | 0.2790 | 0.5907 | 0.3667 |
| Naive | oracle | per image | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| LINEX a2 | default | `(0.8, 0.3)` | 0.4134 | 0.4327 | 0.7385 | — |
| LINEX a2 | best fixed | `(0.5, 0.2)` | **0.2993** | 0.3505 | 0.6961 | **0.4955** |
| LINEX a2 | surrogate argmax | per image | 0.3093 | **0.3456** | **0.6765** | 0.3693 |
| LINEX a2 | oracle | per image | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| CARA a2 | default | `(0.8, 0.3)` | 0.3419 | 0.3449 | **0.5715** | — |
| CARA a2 | best fixed | `(0.6, 0.3)` | **0.2619** | 0.3388 | 0.6698 | **0.4671** |
| CARA a2 | surrogate argmax | per image | 0.2714 | **0.3201** | 0.5936 | 0.3679 |
| CARA a2 | oracle | per image | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

Under both concave objectives the surrogate now has the **lowest mean regret of any strategy** — 0.3456 against the best constant's 0.3505 under LINEX alpha 2, and 0.3201 against 0.3388 under CARA alpha 2 — while losing the median to that same constant by 0.010 in both cases. Its p90 is the best of the four under LINEX alpha 2 and worse than the default's under CARA alpha 2.

Under naive it is beaten by the constant `(0.1, 0.0)` on every column, and by a wide margin: 0.2283 against 0.1437 median. That is the same verdict as every other condition tested for naive, only more so.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | surrogate quality, seed 42. The regression figures are measured on the 55 cells it trained on; the selection figures over all 121:

| Objective | PSNR R² | CLIP R² | rho (121 cells) | top-1 hit | top-3 hit | median rank of pick (of 121) | cells used | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 0.782 | 0.334 | 0.3667 | 7.2% | 21.4% | 12 | 28 | 11 |
| LINEX a2 | 0.781 | 0.360 | 0.3693 | 4.1% | 12.6% | 27 | 27 | 12 |
| CARA a2 | 0.792 | 0.366 | 0.3679 | 4.0% | 10.9% | 27 | 24 | 13 |

Chance top-1 is `1 / 121 = 0.8%`. Note the CLIP `R²` here — 0.334 to 0.366 — against 0.389 to 0.431 for the models trained on all 121 cells. **Training on the extra cells makes the surrogate a better regressor and a worse selector.** That is the whole tension of this document in one comparison, and it is the reason a regression metric cannot be used to choose between these configurations.

## What each strategy achieves in real metric units

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | true PSNR-Unedited and CLIP-Edited at the picked cell, averaged over the 999 test images, seed 42:

| Objective | Strategy | PSNR-Unedited | CLIP-Edited | vs. default |
| --- | --- | ---: | ---: | --- |
| any | default `(0.8, 0.3)` | 17.79 | 21.25 | — |
| Naive | best fixed `(0.1, 0.0)` | 31.55 | 16.94 | +13.76 dB, −4.31 CLIP |
| Naive | surrogate argmax | 27.78 | 18.09 | +9.99 dB, −3.16 CLIP |
| Naive | oracle | 26.44 | 22.66 | +8.65 dB, +1.41 CLIP |
| LINEX a2 | best fixed `(0.5, 0.2)` | 23.31 | 19.90 | +5.52 dB, −1.35 CLIP |
| LINEX a2 | surrogate argmax | 22.68 | 20.44 | +4.89 dB, −0.81 CLIP |
| LINEX a2 | oracle | 24.78 | 23.56 | +6.99 dB, +2.31 CLIP |
| CARA a2 | best fixed `(0.6, 0.3)` | 20.74 | 20.72 | +2.95 dB, −0.53 CLIP |
| CARA a2 | surrogate argmax | 21.90 | 20.69 | +4.11 dB, −0.56 CLIP |
| CARA a2 | oracle | 23.78 | 23.88 | +5.99 dB, +2.63 CLIP |

This is where the configuration earns its keep in terms anyone can act on. Under LINEX alpha 2 it delivers +4.89 dB of background preservation for −0.81 CLIP, against the 121-trained model's +0.38 dB for −0.03 CLIP — a real edit-quality change rather than a rounding error. Under CARA alpha 2 it takes +4.11 dB against the best constant's +2.95 dB for essentially the same CLIP cost (−0.56 against −0.53), so it dominates that constant on the PSNR axis without paying for it on the other.

## Per-image win rate against the default

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42. Phi at the default is 0, so a strategy's phi at its picked cell is its gain over the default:

| Objective | Strategy | win | tie | loss | gain median | gain mean | gain p10 | share of oracle gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | `(0.1, 0.0)` | 87.4% | 0.0% | 12.6% | +0.338 | +0.358 | −0.020 | **66%** |
| Naive | surrogate | 78.7% | 0.3% | 21.0% | +0.247 | +0.263 | −0.169 | 49% |
| LINEX a2 | `(0.5, 0.2)` | 67.4% | 0.0% | 32.6% | +0.145 | +0.082 | −0.351 | 19% |
| LINEX a2 | surrogate | 37.6% | 47.2% | 15.1% | 0.000 | **+0.087** | −0.128 | **20%** |
| CARA a2 | `(0.6, 0.3)` | 62.2% | 0.0% | 37.8% | +0.085 | +0.006 | −0.415 | 2% |
| CARA a2 | surrogate | 30.1% | 55.9% | 14.0% | 0.000 | **+0.025** | −0.171 | **7%** |

The shape of the concave rows is what distinguishes this configuration from both alternatives. It abstains on about half the images — 47.2% and 55.9% ties — and when it does move it wins two to one. Compare the constant, which by construction moves on every image and therefore takes a −0.35 to −0.42 p10 for its trouble; this model's p10 is −0.13 and −0.17. It is a partial, self-selecting version of the constant, and the selection is good enough to keep **all** of the constant's mean gain — four times it under CARA alpha 2, +0.025 against +0.006 — at roughly a third of the downside.

Conditioned on the images it actually moves (four seeds):

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo`:

| Objective | images moved (seed 42) | win | loss | gain median | gain mean | gain p10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 996 of 999 | 79.3% ± 3.2% | 20.7% | +0.248 | +0.276 ± 0.030 | −0.169 |
| LINEX a2 | 528 of 999 | 69.8% ± 1.3% | 30.2% | +0.214 | +0.141 ± 0.023 | −0.331 |
| CARA a2 | 441 of 999 | 65.7% ± 2.1% | 34.3% | +0.172 | +0.036 ± 0.028 | −0.508 |

The moved-image win rate is lower than the 121-trained model's (69.8% against 79.9% under LINEX alpha 2), so its individual decisions are worse. It wins overall by making an order of magnitude more of them: 61.4% of images against 5.7%.

## Where it sends images it was never trained on

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42, share of picks landing in the 66 cells with `t_end ≥ t_start`:

| Objective | oracle in new cells | this model in new cells | 121-trained model, for reference | oracle on `(0.0, 0.0)` | this model on `(0.0, 0.0)` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive | 27.6% | **55.8%** | 11.9% | 14.9% | 33.9% |
| LINEX a2 | 26.7% | 14.9% | 2.0% | 9.8% | 0.3% |
| CARA a2 | 25.6% | 13.2% | 4.4% | 5.0% | 0.1% |

**The model that never saw these cells picks them more often than the model that did.** Under naive it sends 55.8% of images into the unseen region against the oracle's 27.6%, and a third of all images to `(0.0, 0.0)` alone — a cell it has no label for — against the oracle's 14.9%. That is over-extrapolation, and it is why naive is this configuration's worst objective: the optimism it produces is unearned, and under a linear phi the model was already deviating on every image, so there is no inertness for it to cure.

Under the concave objectives the same optimism is smaller in absolute terms (14.9% and 13.2% of picks) and lands on useful cells more often than not. The honest reading is that this configuration's advantage under concave phi comes substantially from being *pushed off the default* rather than from knowing where to go. That does not make the measured gain less real, but it does mean the right follow-up is to reproduce the effect deliberately — by calibrating predicted phi, or by excluding only the degenerate corner from training — rather than by shipping a model kept ignorant of a third of its candidate set.

## The gate, again with the wrong sign

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | `NOISE_FLOOR_PHI` as a selection rule, seed 42:

| Floor | Naive deviate | Naive gain mean | LINEX a2 deviate | LINEX a2 gain mean | CARA a2 deviate | CARA a2 gain mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 99.7% | +0.2633 | 52.8% | **+0.0871** | 44.1% | +0.0248 |
| 0.05 | 98.4% | **+0.2638** | 38.2% | +0.0748 | 32.9% | **+0.0249** |
| 0.1 | 97.0% | +0.2637 | 27.4% | +0.0577 | 24.2% | +0.0229 |
| 0.2 | 93.1% | +0.2574 | 14.3% | +0.0365 | 9.8% | +0.0150 |
| 0.3 | 87.0% | +0.2517 | 5.8% | +0.0143 | 2.4% | +0.0011 |
| 0.5 | 66.4% | +0.2138 | 0.1% | +0.0001 | 0.0% | 0.0000 |
| 1.0 | 1.1% | +0.0035 | 0.0% | 0.0000 | 0.0% | 0.0000 |

As in the 121-trained case, every floor above zero costs gain, and the fall is steeper here because the deviations are what this configuration is for: a floor of 0.3 removes nine tenths of them under LINEX alpha 2 and five sixths of the gain. **Keep `NOISE_FLOOR_PHI = 0.0`.** The one flat spot is naive at floor 0.05, where +0.2638 marginally beats +0.2633 by suppressing the least confident 1.3% of an already-saturated deviation rate — noise, not a knob.

## Against the alternatives

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | four seeds, everything scored over the same 121 cells, so all rows are comparable:

| Objective | Strategy | regret median | regret mean | phi gain mean | deviate rate |
| --- | --- | ---: | ---: | ---: | ---: |
| LINEX a2 | default `(0.8, 0.3)` | 0.4134 | 0.4327 | 0.000 | 0% |
| LINEX a2 | best fixed `(0.5, 0.2)` | 0.2993 | 0.3505 | +0.082 | 100% |
| LINEX a2 | surrogate trained on 121 | 0.4020 ± 0.0036 | 0.4161 ± 0.0050 | +0.017 ± 0.005 | 5.7% |
| LINEX a2 | **surrogate trained on 55** | **0.2979 ± 0.0100** | **0.3474 ± 0.0072** | **+0.085 ± 0.007** | 61.4% |
| CARA a2 | default `(0.8, 0.3)` | 0.3419 | 0.3449 | 0.000 | 0% |
| CARA a2 | best fixed `(0.6, 0.3)` | **0.2619** | 0.3388 | +0.006 | 100% |
| CARA a2 | surrogate trained on 121 | 0.3208 ± 0.0052 | 0.3302 ± 0.0035 | +0.015 ± 0.004 | 14.3% |
| CARA a2 | **surrogate trained on 55** | 0.2814 ± 0.0092 | **0.3287 ± 0.0125** | +0.016 ± 0.013 | 45.9% |
| Naive | default `(0.8, 0.3)` | 0.5162 | 0.5423 | 0.000 | 0% |
| Naive | **best fixed `(0.1, 0.0)`** | **0.1437** | **0.1843** | **+0.358** | 100% |
| Naive | surrogate trained on 121 | 0.1861 ± 0.0248 | 0.2206 ± 0.0171 | +0.322 ± 0.017 | 99.9% |
| Naive | surrogate trained on 55 | 0.2205 ± 0.0363 | 0.2675 ± 0.0295 | +0.275 ± 0.030 | 99.7% |

For LINEX alpha 2 this configuration is the best row on both median and mean. For CARA alpha 2 it is the best on the mean and second on the median. For naive it is last among the model rows and well behind the constant. The reference rows are single-seed because they involve no model.

## Caveats

- One split (`SPLIT_SEED = 42`), four initialization seeds, 999 test images. CARA alpha 2's mean gain here is 1.3 sigma from zero, so that particular figure is the least secure in the document.
- Sixty-six of the 121 selectable cells were unlabeled during training and the per-cell anchor has no entry for them, so their predictions are extrapolated and the anchor substitutes its nearest known cell. Every number here is downstream of that.
- The regression metrics are computed on the 55 training cells, and the selection metrics over 121, so the two halves of the surrogate-quality table are not measured on the same candidate set.
- The configuration was tuned in `SUMMARY.md` for LINEX alpha 2 on the 55-cell grid against the `(0.9, 0.3)` default, so this document happens to use it on the grid it was tuned for while measuring it against a different default cell and a wider selection set.
- Nothing here is comparable to numbers scored on 55 cells, including `SUMMARY.md` and the `test 55` rows of `RESULTS.md`: phi is normalized over the candidate set, so widening the set redefines it.

## Summary

`DIR_NAME = UltraEdit_Region_10000`, `CHORD_EDIT_MODEL = sd_turbo`, trained on 55 cells, scored over 121, default `(0.8, 0.3)`, `PHI_ALPHA = 2.0`.

A surrogate trained only on the original lower-triangle cells and then allowed to select from the full grid is the best configuration measured for either concave objective on mean regret: 0.3474 under LINEX alpha 2 against 0.3505 for the best constant cell and 0.4161 for the surrogate trained on all 121 cells, and 0.3287 under CARA alpha 2 against 0.3388 and 0.3302. In real terms it buys +4.89 dB of background preservation for −0.81 CLIP under LINEX alpha 2, where the 121-trained model manages +0.38 dB. It achieves this by deviating from the default on 46–61% of images rather than 6–14%, abstaining on the rest, and winning about two thirds of the moves it makes — a partial version of the best constant that keeps most of the gain at a third of the downside. The costs are real and worth stating plainly: its whole-grid rank correlation is much worse (0.358–0.384 against 0.585–0.633), its CLIP `R²` is lower than the 121-trained model's, and it over-selects the region it never saw — 55.8% of picks under naive against the oracle's 27.6%, which is why naive is its worst objective. The gain under concave phi therefore comes substantially from being pushed off the default rather than from understanding the new cells, and the recommended reading is diagnostic rather than prescriptive: this is evidence that decisiveness is what the concave objectives lack, and the way to buy it should be calibration of predicted phi, not withholding a third of the candidate set from training.
