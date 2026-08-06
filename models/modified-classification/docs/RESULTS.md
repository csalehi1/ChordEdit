# Does the residual-space surrogate beat the fixed default `(0.8, 0.3)`?

## Overview

This document re-asks the question of [RESULTS.md](RESULTS.md) after one change to the training target. The surrogate **M̂** no longer regresses full per-sample normalized deltas: with `M_TARGET_SPACE = "residual"` it regresses each image's *deviation* from the train split's mean true delta surface, and the selector **T** adds that surface back before scalarizing into phi. The surface is a label-only constant saved with the run as `mean_surface.pt`. A shrunk, uninformative prediction now falls back to the population-best cell instead of the default — the failure mode RESULTS.md diagnosed as inertness ("the model knows where to go and will not go there") is structurally removed. The per-cell anchor (`USE_CELL_ANCHOR`) was deleted at the same time: in residual space the per-cell mean target is exactly zero, so the anchor had become a no-op.

Every table below is `DIR_NAME = UltraEdit_Region_10000`, `CHORD_EDIT_MODEL = sd_turbo`, the full 121-cell grid, 999 test images, `SPLIT_SEED = 42` — the same split, default cell and phi as RESULTS.md, verified below by the model-free rows matching to four decimals.

**Scope differs from RESULTS.md in one important way.** These are four seeds (42–45) of a *single* configuration, trained with the shipped objective — ranking loss on LINEX alpha 2 — and checkpointed on validation phi Spearman, 20 epochs, ranking weight 3, `GRIDS_PER_BATCH = 32`. RESULTS.md retrained a model per objective; here the naive and CARA alpha 2 rows re-score the *same* predicted metric grids under those objectives. LINEX alpha 2 is the like-for-like comparison; the other two show how one model transfers across curvatures, not what a model trained for them would do. Selection is the raw argmax of predicted phi: no gate, no shrinkage, no ensemble. Runs are `runs/UltraEdit_Region_10000/test_sdturbo_s{42,43,44,45}`; numbers were computed by `analyze_results2.py` against each run's artifacts.

**The headline is that the selector now selects.** Under LINEX alpha 2 it deviates from the default on 92% of images (against 5.7% in RESULTS.md), cuts median regret by 45% against the default (against 3%), collects a mean phi gain of +0.141 against +0.017, and — for the first time in this line of work — **beats the best fixed cell**, 0.2263 median regret against `(0.5, 0.2)`'s 0.2993. The mechanism RESULTS.md asked for is exactly what happened: predicted per-image headroom went from 1% of the true headroom to 93%. The cost is visible in the tails and in real metric units, and CLIP is now almost entirely carried by the population prior rather than the per-image model — both discussed below.

## Terms

Terms are as in [RESULTS.md](RESULTS.md#terms) (phi, Naive/LINEX/CARA, regret, rho, oracle, best fixed, deviate rate, new cells), plus:

| Term | Definition |
| --- | --- |
| **mean surface** | The train split's per-cell mean of true normalized deltas, `mean_surface.pt`. Its phi surface peaks at the best fixed cell by construction. |
| **residual** | The model's regression target: full delta minus the mean surface at that cell. T adds the surface back before phi, so predicted phi = phi(residual + surface). |

## Headline

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | 121 cells, 999 test images, four seeds, `SPLIT_SEED = 42`. One model per seed (trained on LINEX alpha 2), re-scored per objective:

| Objective | regret median | vs. default | regret mean | vs. default | phi gain mean | deviate rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive (re-scored) | 0.1476 ± 0.0079 | **−71%** | 0.2010 ± 0.0088 | **−63%** | **+0.341 ± 0.009** | 95.6% ± 1.0% |
| LINEX, alpha 2 (trained) | 0.2263 ± 0.0035 | **−45%** | 0.2918 ± 0.0039 | **−33%** | **+0.141 ± 0.004** | 92.4% ± 1.8% |
| CARA, alpha 2 (re-scored) | 0.2383 ± 0.0047 | **−30%** | 0.3362 ± 0.0070 | −3% | +0.009 ± 0.007 | 89.8% ± 2.2% |

Against the same rows of RESULTS.md (per-objective retrained delta-space models):

| Objective | regret median, old → new | phi gain mean, old → new | deviate rate, old → new |
| --- | --- | --- | --- |
| Naive | 0.1861 → **0.1476** | +0.322 → **+0.341** | 99.9% → 95.6% |
| LINEX a2 | 0.4020 → **0.2263** | +0.017 → **+0.141** | 5.7% → 92.4% |
| CARA a2 | 0.3208 → **0.2383** | +0.015 → +0.009 | 14.3% → 89.8% |

Two facts frame the rest:

1. **Under LINEX alpha 2 the model finally beats every model-free strategy.** Median regret 0.2263 against the best fixed cell's 0.2993 and the default's 0.4134; mean gain +0.141 against the constant's +0.082; share of oracle gain 33% against 19%. RESULTS.md's factorial found that even training on the 55-cell lower triangle only reached 0.2979 — the residual target beats that workaround by 24% *while training on all 121 cells*, so the widened-grid pathology is fixed rather than avoided.
2. **The model can now lose, and under CARA alpha 2 it mostly breaks even.** The old model's protection was abstention — a gain distribution spiked at zero with p10 = 0.000. The new one commits on ~90% of images, and its p10 is −0.32 (LINEX) and −0.53 (CARA). Under CARA's exponential regression penalty those tails eat almost the whole median improvement: gain mean +0.009 ± 0.007, barely positive and below the old model's +0.015. Caution was the old model's only virtue; boldness is the new model's, and CARA is where boldness is taxed.

## Regret against the model-free references

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | single seed (42). Default, best fixed and oracle rows are identical to RESULTS.md to four decimals, confirming the same split and phi:

| Objective | Strategy | Cell | regret median | regret mean | regret p90 | rho |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Naive | default | `(0.8, 0.3)` | 0.5162 | 0.5423 | 0.9177 | — |
| Naive | best fixed | `(0.1, 0.0)` | 0.1437 | **0.1843** | **0.4277** | 0.5947 |
| Naive | surrogate argmax | per image | **0.1381** | 0.1924 | 0.4530 | **0.6273** |
| Naive | oracle | per image | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| LINEX a2 | default | `(0.8, 0.3)` | 0.4134 | 0.4327 | 0.7385 | — |
| LINEX a2 | best fixed | `(0.5, 0.2)` | 0.2993 | 0.3505 | 0.6961 | 0.4955 |
| LINEX a2 | surrogate argmax | per image | **0.2248** | **0.2861** | **0.6472** | **0.5937** |
| LINEX a2 | oracle | per image | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| CARA a2 | default | `(0.8, 0.3)` | 0.3419 | 0.3449 | **0.5715** | — |
| CARA a2 | best fixed | `(0.6, 0.3)` | 0.2619 | 0.3388 | 0.6698 | 0.4671 |
| CARA a2 | surrogate argmax | per image | **0.2316** | **0.3317** | 0.7984 | **0.5896** |
| CARA a2 | oracle | per image | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

The surrogate now has the best median regret in all three objectives and the best rank correlation in all three — the rho/regret tension that ran through RESULTS.md and SUMMARY.md is resolved under LINEX alpha 2, where it wins every column. The two re-scored objectives keep a caveat each: under naive the best fixed cell still wins the mean and the tail (the constant `(0.1, 0.0)` remains extremely strong when phi is linear), and under CARA the default's p90 is untouchable because never moving cannot have a bad tail.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | surrogate quality behind those rows, four seeds. R² and ranges are in full-delta units (predicted residual + surface against true delta), not the raw-metric units RESULTS.md reported, because that is the space the model now works in; the true per-grid delta range is exactly 1 by construction:

| Quantity | value (4 seeds) |
| --- | ---: |
| PSNR delta R² | 0.889–0.892 |
| CLIP delta R² | **0.011–0.039** |
| predicted PSNR range per grid (true = 1) | 1.01 |
| predicted CLIP range per grid (true = 1) | 0.62–0.71 |
| best epoch | 9–12 |

| Objective | rho | top-1 hit | oracle in top-3 | median rank of true best (of 121) | cells used |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive | 0.626 ± 0.002 | 11.8% ± 0.6% | 30.5% ± 1.0% | 9.8 | 29 ± 5 |
| LINEX a2 | 0.588 ± 0.005 | 8.1% ± 0.8% | 20.7% ± 1.3% | 14.3 | 36 ± 5 |
| CARA a2 | 0.588 ± 0.003 | 5.8% ± 0.5% | 14.8% ± 1.2% | 17.0 | 39 ± 4 |

Chance top-1 is 0.8%. The old model's concave-objective top-1 rates (1.6%, 2.6%) were almost entirely images where the default happened to be the true best; the new rates (8.1%, 5.8%) are genuine hits, and the true best cell now sits at median rank 14–17 in the predicted ordering instead of 34–40. Rank correlation is unchanged from RESULTS.md (0.585–0.639 there) — **the residual target did not improve the ordering; it fixed the level, which is what the argmax needed.**

The CLIP row is the number to worry about. In full-delta space the per-image CLIP signal is nearly zero (R² ≈ 0.03): what remained of CLIP prediction in the old model was mostly the population surface, and once the surface is supplied as a constant the residual head learns almost nothing image-specific about CLIP. The selector's CLIP awareness is therefore the prior, not the model — consequences below.

## What each strategy achieves in real metric units

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | true PSNR-Unedited and CLIP-Edited at whichever cell each strategy picks, averaged over the 999 test images, seed 42. **This is the only table comparable across objectives.**

| Objective | Strategy | PSNR-Unedited | CLIP-Edited | vs. default |
| --- | --- | ---: | ---: | --- |
| any | default `(0.8, 0.3)` | 17.79 | 21.25 | — |
| Naive | best fixed `(0.1, 0.0)` | 31.55 | 16.94 | +13.76 dB, −4.31 CLIP |
| Naive | surrogate argmax | 29.00 | 18.63 | +11.21 dB, −2.62 CLIP |
| Naive | oracle | 26.44 | 22.66 | +8.65 dB, +1.41 CLIP |
| LINEX a2 | best fixed `(0.5, 0.2)` | 23.31 | 19.90 | +5.52 dB, −1.35 CLIP |
| LINEX a2 | surrogate argmax | 27.38 | 19.24 | +9.59 dB, −2.01 CLIP |
| LINEX a2 | oracle | 24.78 | 23.56 | +6.99 dB, +2.31 CLIP |
| CARA a2 | best fixed `(0.6, 0.3)` | 20.74 | 20.72 | +2.95 dB, −0.53 CLIP |
| CARA a2 | surrogate argmax | 25.67 | 19.82 | +5.99 dB, −1.43 CLIP |
| CARA a2 | oracle | 23.78 | 23.88 | +5.99 dB, +2.63 CLIP |

The old LINEX row read +0.38 dB, −0.03 CLIP — a rounding error. The new one reads +9.59 dB, −2.01 CLIP: real movement, but movement of a particular kind. Every oracle row improves both metrics at once; every surrogate row trades CLIP for PSNR. With essentially no per-image CLIP signal (R² ≈ 0.03), the model's phi differences are driven by PSNR residuals plus the fixed prior, so it chases PSNR into the low-`t_start` corner and pays the CLIP toll the concave objectives were designed to refuse. It wins phi anyway because LINEX's linear half prices +9.6 dB above −2.0 CLIP at these ranges — but the both-improve region the oracle lives in (70% of images) is found on only 26% of picks. Fixing per-image CLIP remains the whole of the remaining headroom, exactly as RESULTS.md concluded, and the residual reformulation has made that deficit undisguisable.

## Per-image win rate against the default

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42. Phi at the default is 0, so true phi at the pick *is* the gain:

| Objective | Strategy | win | tie | loss | gain median | gain mean | gain p10 | share of oracle gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | `(0.1, 0.0)` | 87.4% | 0.0% | 12.6% | +0.338 | +0.358 | −0.020 | **66%** |
| Naive | surrogate | 85.2% | 3.4% | 11.4% | +0.330 | +0.350 | −0.016 | 65% |
| LINEX a2 | `(0.5, 0.2)` | 67.4% | 0.0% | 32.6% | +0.145 | +0.082 | −0.351 | 19% |
| LINEX a2 | surrogate | 65.6% | 5.9% | 28.5% | +0.162 | **+0.147** | −0.317 | **34%** |
| CARA a2 | `(0.6, 0.3)` | 62.2% | 0.0% | 37.8% | +0.085 | +0.006 | −0.415 | 2% |
| CARA a2 | surrogate | 56.8% | 8.1% | 35.1% | +0.079 | +0.013 | −0.555 | **4%** |

In RESULTS.md this table was "the tie column is the story" — 88–97% ties under the concave objectives. The ties are gone (6–8%), and the surrogate's profile now looks like the best fixed cell's, with one real edge: under LINEX it converts nearly the same win rate into 79% more mean gain, because its wins are per-image peaks rather than one compromise cell. Its p10 is no longer 0.000 — the safe strategy became a risky one, slightly less risky than the constant under LINEX (−0.317 against −0.351) and more risky under CARA (−0.555 against −0.415).

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | conditioned on the images the surrogate actually moves, four seeds:

| Objective | images moved (of 999) | win | gain median | gain mean | gain p10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive | 955 ± 10 | 87.1% ± 1.1% | +0.336 | +0.357 ± 0.009 | −0.031 |
| LINEX a2 | 924 ± 18 | 69.4% ± 0.7% | +0.184 | +0.152 ± 0.005 | −0.345 |
| CARA a2 | 897 ± 22 | 62.2% ± 1.0% | +0.110 | +0.010 ± 0.008 | −0.570 |

RESULTS.md's version of this table was the "the deviations it does make are excellent" result: 35 moved images at +0.293 each. The new model moves 924 and averages +0.152 — per deviation it is half as selective, in aggregate it collects fourteen times the gain (+0.141 against +0.010 across the whole split). The win rate on moves is 69% against the old 80%, which is the price of moving on everything instead of only the easiest calls.

## Why the selector moves now

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42:

| Objective | mean TRUE surface argmax (value) | mean PRED surface argmax (value) | mean per-image max phi, true | pred | pred / true |
| --- | --- | --- | ---: | ---: | ---: |
| Naive | `(0.1, 0.0)` +0.358 | `(0.1, 0.0)` +0.437 | +0.542 | +0.524 | 97% |
| LINEX a2 | `(0.5, 0.2)` +0.082 | `(0.5, 0.2)` +0.218 | +0.433 | +0.402 | **93%** |
| CARA a2 | `(0.6, 0.2)` +0.007 | `(0.5, 0.2)` +0.163 | +0.345 | +0.306 | 89% |

This is the table that showed the disease in RESULTS.md and now shows the cure. The old model's mean predicted phi surface peaked at the default cell at exactly zero and its per-image predicted headroom was 0.8–4.9% of the truth. The new model's mean predicted surface peaks at the population-best cell — partly by construction, since a zero residual reproduces the mean surface — and its per-image headroom estimate is 89–97% of the truth. There is finally something for an argmax to find. The predicted level now *overshoots* the mean surface's own value (+0.218 against the true +0.082 at `(0.5, 0.2)` under LINEX), meaning the residual head adds systematic optimism on top of the prior; that optimism is what drives the 92% deviate rate, and its miscalibration is the natural next target.

Range compression, RESULTS.md's suspect, is partially relieved and partially recast: predicted PSNR now spans the full true per-grid delta range (1.01 of 1) and predicted CLIP spans 0.62–0.71 — up from the old raw-space 29–46% — but almost all of the CLIP span is the mean surface itself, not per-image signal (CLIP delta R² ≈ 0.03). Both-improve picks confirm it: the model lands on cells where both metrics truly improve for 26% of images (old: 1.3%) against the oracle's 70% — an order of magnitude better, still driven by the prior knowing where the both-improve region tends to sit rather than the model seeing it per image.

## Which cells get picked

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42, of 121 cells:

| Objective | Strategy | distinct cells | deviate rate | most-picked cells |
| --- | --- | ---: | ---: | --- |
| Naive | surrogate | 35 | 96.6% | `(0.0, 0.0)` 29.3%, `(0.1, 0.0)` 28.0%, `(0.2, 0.0)` 9.5%, `(0.5, 0.2)` 7.0% |
| Naive | oracle | 84 | 99.4% | `(0.0, 0.0)` 14.9%, `(0.1, 0.0)` 11.0%, `(0.2, 0.0)` 7.6%, `(0.3, 0.0)` 6.0% |
| LINEX a2 | surrogate | 40 | 94.1% | `(0.0, 0.0)` 23.0%, `(0.1, 0.0)` 15.6%, `(0.2, 0.0)` 11.0%, `(0.5, 0.2)` 10.0% |
| LINEX a2 | oracle | 89 | 98.7% | `(0.0, 0.0)` 9.8%, `(0.1, 0.0)` 6.6%, `(0.3, 0.0)` 4.6%, `(0.2, 0.0)` 4.2% |
| CARA a2 | surrogate | 42 | 91.9% | `(0.0, 0.0)` 16.8%, `(0.5, 0.2)` 10.9%, `(0.3, 0.0)` 9.1%, `(0.8, 0.3)` 8.1% |
| CARA a2 | oracle | 92 | 98.0% | `(0.0, 0.0)` 5.0%, `(0.4, 0.2)` 4.4%, `(0.5, 0.2)` 3.8%, `(0.1, 0.0)` 3.7% |

The old concave-objective rows were 11 and 16 distinct cells with 88–97% of mass on the default; the new ones use 40 and 42 cells, put 8% or less on the default, and their modal picks coincide with the oracle's modal pick `(0.0, 0.0)` — the cell that did not exist before the annotation pass and that RESULTS.md flagged as the oracle's favorite everywhere. Picks landing in the 66 new cells (`t_end ≥ t_start`): 29–33% for the surrogate against the oracle's 26–28% — the old model reached 2–12%. The addition is finally being exploited, mildly over-exploited if anything, consistent with the optimism noted above.

## The gate, revisited

`NOISE_FLOOR_PHI` keeps the default unless predicted gain clears a floor. `DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42:

| Floor | Naive deviate | Naive gain mean | LINEX a2 deviate | LINEX a2 gain mean | CARA a2 deviate | CARA a2 gain mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 96.6% | **+0.350** | 94.1% | +0.147 | 91.9% | +0.013 |
| 0.05 | 93.8% | +0.347 | 89.2% | +0.147 | 85.5% | +0.016 |
| 0.1 | 89.8% | +0.340 | 84.6% | **+0.148** | 77.7% | +0.026 |
| 0.2 | 82.2% | +0.327 | 72.8% | +0.141 | 60.9% | +0.022 |
| 0.3 | 72.7% | +0.303 | 56.9% | +0.123 | 42.5% | **+0.027** |
| 0.5 | 46.2% | +0.216 | 27.7% | +0.078 | 10.7% | +0.012 |
| 1.0 | 3.0% | +0.019 | 0.5% | +0.002 | 0.0% | 0.000 |

In RESULTS.md every positive floor was monotonically harmful and the recommendation was that the model needed the *opposite* knob — a negative floor. That is no longer true. Under LINEX a2 the curve is flat to slightly rising through 0.1 (the floor prunes moves whose predicted gain was noise), and under CARA a2 a floor of 0.3 roughly doubles the mean gain by suppressing the reckless tail — turning the barely-positive +0.013 into +0.027 at a 42.5% deviate rate. The gate finally has work to do in the direction it was built for. `NOISE_FLOOR_PHI = 0.0` remains right for LINEX; a tuned positive floor is the cheapest fix for CARA.

## What this says

- **Recentering the target is what un-froze the selector.** Same data, same architecture minus a no-op anchor, same ranking loss, same checkpoint metric — only the regression target changed, and the deviate rate under the trained objective went from 5.7% to 92.4% while median regret fell 44%. RESULTS.md located the inertness in a predicted-level failure that no loss term constrained; supplying the level as a constant and asking the model only for deviations removed it.
- **It beats the constant now, and beats the lower-triangle workaround.** 0.2263 median regret against the best fixed cell's 0.2993 and against 0.2979 for RESULTS.md's 55-cell-trained model scored on the same grid — while training on all 121 cells. The widened grid stopped being harmful once the model no longer had to represent the absolute surface the degenerate corner distorts.
- **Ranking did not improve; the level did.** Rho is statistically unchanged (0.588 against 0.585 under LINEX). The whole gain came from predicted headroom going from 1% of truth to 93%. This cleanly confirms RESULTS.md's diagnosis that scale, not ordering, was the binding constraint.
- **The model's risk profile inverted.** The old model could not lose (p10 gain exactly 0.000, 96.5% ties); the new one loses on 28–35% of images with p10 −0.32 to −0.56. Under LINEX the wins price the losses in; under CARA they barely do (+0.009 ± 0.007), and the old cautious model's +0.015 is not clearly beaten. A model *trained* under CARA's own curvature, or the positive gate floor above, are the obvious responses.
- **CLIP is now openly the weak head.** Per-image CLIP signal in delta space is R² ≈ 0.03; the selector's CLIP awareness is the population prior. Every surrogate row in the real-units table trades CLIP away for PSNR, against oracle rows that improve both. This was RESULTS.md's third conclusion and it survives the reformulation intact — now without the inertness masking it.

## Caveats

- One split (`SPLIT_SEED = 42`), four initialization seeds. Reference, real-units, cell-usage and gate tables are single-seed (42).
- **Naive and CARA rows are re-scorings of the LINEX-trained model**, not per-objective retrainings as in RESULTS.md. The LINEX comparison is like-for-like; the CARA shortfall in particular may just mean the ranking loss taught LINEX preferences.
- The configuration changed in more than the target space relative to RESULTS.md's runs: the per-cell anchor was removed (a no-op in residual space, not in the old delta space) and `M_TARGET_SPACE = "raw"` no longer exists. The attribution to recentering rests on the anchor's residual-space equivalence, which the smoke tests confirmed (bit-identical training with and without it), not on a delta-space ablation with the new code.
- R² and range figures are in full-delta units and are not directly comparable to RESULTS.md's raw-metric R²; the CLIP R² collapse partly reflects that the easy (population-level) variance moved into the supplied surface.
- The predicted phi level systematically overshoots (+0.218 against +0.082 at the mean-surface argmax); the gains reported here are what the overshoot delivers under argmax, and any consumer of predicted phi *values* (rather than picks) should recalibrate first.
- No gate, shrinkage or ensembling anywhere; `SUMMARY.md`'s tuned additions have not been revisited under the new target space.

## Worth trying next

1. **A tuned positive `NOISE_FLOOR_PHI` for the concave objectives.** The gate finally suppresses the right thing: floor 0.3 doubles CARA's mean gain in the sweep above. Tune per objective on validation — one number, no retraining.
2. **Retrain under CARA's own curvature before concluding it lost.** The CARA row re-scores a LINEX-trained model; RESULTS.md's per-objective protocol would say whether the tail losses are the objective's nature or the training's mismatch.
3. **Calibrate the predicted phi level.** Predictions overshoot the prior by ~2.7x at the mean-surface argmax. A monotone map from predicted to true gain on validation — the recalibration RESULTS.md proposed for the opposite problem — would make the gate floors principled instead of swept.
4. **Attack the CLIP head, again.** Unchanged from RESULTS.md and now unmasked: per-image CLIP is the difference between the current trade-PSNR-for-CLIP behavior and the oracle's both-improve picks, worth roughly the remaining two-thirds of oracle gain under LINEX.
5. **Re-run the RESULTS.md factorial and the classifier comparison under residual targets.** The lower-triangle advantage should now be gone (this document shows 121-trained beating the old 55-trained numbers, but the direct A–D factorial has not been rerun), and the classifier comparison should be refreshed since the surrogate can now commit.

## Summary

`DIR_NAME = UltraEdit_Region_10000`, `CHORD_EDIT_MODEL = sd_turbo`, 121-cell grid, 999 test images, default cell `(0.8, 0.3)`, `PHI_ALPHA = 2.0`, four seeds, residual target space.

Retargeting the surrogate at deviations from the train split's mean delta surface — with the surface added back at selection — turns the inert selector of RESULTS.md into one that deviates on 92% of images and, under the objective it was trained for, beats every strategy short of the oracle: median regret 0.2263 against 0.2993 for the best fixed cell, 0.4020 for the old delta-space model and 0.4134 for the default, with mean phi gain +0.141 ± 0.004 against the old +0.017. Rank correlation is unchanged, predicted per-image headroom went from 1% of the truth to 93%, and the model now trains on the full 121-cell grid without the degradation that made RESULTS.md recommend retreating to the lower triangle — the recentering fixes what that retreat only avoided. The costs are a real loss tail where none existed (p10 gain −0.32 under LINEX, −0.56 under CARA, where the mean gain of +0.009 roughly ties the old cautious model), predicted phi levels that overshoot and want recalibration, and a CLIP head whose per-image signal has collapsed to R² ≈ 0.03 in delta space, leaving CLIP awareness to the population prior and every surrogate pick trading CLIP for PSNR while the oracle improves both. The gate now works in its intended direction — a positive floor is flat-to-helpful, roughly doubling CARA's mean gain at 0.3 — so the next moves are a validation-tuned floor, a CARA-trained model before judging that objective, phi-level recalibration, and the same CLIP fix every document in this line has asked for.
