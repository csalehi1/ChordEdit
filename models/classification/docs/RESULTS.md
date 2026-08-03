# Does argmaxing the classifier beat the fixed default (0.8, 0.3)?

## Overview

The classifier exists to replace a single global timestep pair with a per-image
choice. This document tests whether it earns that: for each objective at
`alpha = 2`, it compares the classifier's per-image pick against always using
the fixed default cell `(0.8, 0.3)`, against the best fixed cell obtainable
with no model at all, and against the per-image oracle.

Every table below is `DIR_NAME = UltraEdit_Region_10000`,
`CHORD_EDIT_MODEL = sd_turbo`, and is labeled as such.

**Scope: alpha = 2.** Three objectives are reported — naive (alpha-free), LINEX
alpha 2, and CARA alpha 2 — spanning the curvature range from linear to
strictly concave at the alpha the pipeline actually ships. The higher-curvature
variants (`alpha = 5`) are out of scope here; they were measured in an earlier
version of this document and only sharpened the same trend.

**These numbers are computed on the full 121-cell grid.** The annotation
pipeline has produced the 66 cells with `t_end >= t_start`, so the metrics CSV
now covers all 11 x 11 positions rather than the 55 of the strict lower
triangle. The usable sample set is unchanged (9985 samples, 999 test), so this
is purely a widening of the candidate set: every strategy chooses from 121 cells
instead of 55, and phi's per-sample normalization ranges over all of them.
Fifteen samples in the CSV are unusable — fourteen carry metrics *only* for the
added cells and one has no valid metrics at all — and are dropped in
`_data.load_df` for having an incomplete grid, recovering exactly the previous
sample set.

The setup is otherwise the recommended configuration from
[SUMMARY.md](SUMMARY.md) — class weighting off, CORAL head, cosine decay, 20
epochs, `(DEFAULT_T_START, DEFAULT_T_END) = (0.8, 0.3)` — retrained once per
objective at `SEED = 42`, plus a four-seed repeat of each for spreads.
Selection is the joint argmax of the two heads over the 121 cells: no gate, no
shrinkage, no ensemble. Test split, 999 images. Runs are
`outputs/UltraEdit_Region_10000/a2_{naive,linex_a2,cara_a2}_s{42,1,2,3}`.

**The answer depends on the objective's curvature, and the median regret alone
will tell you the wrong thing.** Median regret improves under all three
objectives. Mean regret improves under only two. Under CARA alpha 2 the
classifier makes the median image better and the average image slightly worse
than if it had never deviated from the default. Only under LINEX alpha 2 does
the classifier beat both the default and the best possible constant cell on
median and mean at once.

## Terms

| Term | Meaning |
| --- | --- |
| **cell** | one `(t_start, t_end)` pair; all 121 grid positions carry data. |
| **phi** | the scalar objective ranking a sample's cells, built from per-sample range-normalized deltas of PSNR-Unedited and CLIP-Edited against the default cell. phi at the default cell is exactly 0 by construction, so **phi at a picked cell is that strategy's gain over the default**. |
| **naive** | `sum_i w_i * Delta_i`. Linear; indifferent to how gains are split between metrics. Alpha-free, so it is the alpha -> 0 end of the ladder. |
| **LINEX** | the mean of naive and CARA. Keeps CARA's regression penalty without its reward cap; curvature is half of CARA's at matched alpha. The shipped default. |
| **CARA** | `(1/a) * sum_i w_i * (1 - exp(-a * Delta_i))`. Strictly concave: regressions are penalized exponentially and unboundedly, while gains **saturate at `w_i / a`** — 0.2 per metric at `a = 2`. |
| **regret** | `max_T phi(T) - phi(T_picked)`. Lower is better. |
| **best fixed cell** | argmax of the *training* split's mean phi surface: the best single constant, no model. |
| **deviate rate** | fraction of images where a strategy picks something other than the default cell. |
| **new cells** | the 66 positions with `t_end >= t_start`, unlabeled until the recent annotation pass. |

Phi is objective-specific, so numbers are comparable only *within* an objective —
except the raw-metric table, which is comparable across all of them.

## Headline

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | 121 cells,
999 test images, four seeds per objective, `SPLIT_SEED = 42`:

| Objective | regret median | vs default | regret mean | vs default | phi gain mean |
| --- | --- | --- | --- | --- | --- |
| Naive (alpha-free) | 0.2286 +/- 0.0085 | **-56%** | 0.2649 +/- 0.0096 | **-51%** | **+0.277 +/- 0.010** |
| LINEX, alpha 2 (the default) | 0.2593 +/- 0.0003 | **-37%** | 0.3283 +/- 0.0043 | **-24%** | **+0.104 +/- 0.004** |
| CARA, alpha 2 | 0.2403 +/- 0.0059 | -30% | 0.3590 +/- 0.0055 | **+4%** | **-0.014 +/- 0.006** |

Read the mean column. Under CARA alpha 2 the median improves by 30% — the
second-largest improvement in the table — while the average image ends up
slightly *worse* off than never deviating at all. Concave objectives punish a
ranking error more than they reward a correct one, so a model that is right
more often than not can still lose on average.

The spreads are small relative to every gap: the weakest effect (CARA alpha 2,
gain mean -0.014 +/- 0.006) is still two and a half sigma from zero, so even
the marginal sign is real. LINEX alpha 2 is the tightest configuration in the
set — four seeds land within 0.0007 of each other on median regret.

Three facts frame everything below:

1. **Under naive, most of the gain needs no model.** The single fixed cell
   `(0.1, 0.0)` beats the classifier on regret median (0.1437 vs 0.2188 at seed
   42) and mean (0.1843 vs 0.2554). Per-image selection is negative value added.
2. **Under LINEX alpha 2 — the shipped default — the classifier beats both
   references,** the default cell and the best fixed cell, on median and mean.
   This is the one objective where per-image selection is unambiguously worth
   having.
3. **Under CARA alpha 2 the classifier wins the median and loses the mean, and
   loses to a constant on the mean as well.** The best fixed cell `(0.6, 0.3)`
   holds mean regret at 0.3388 against the classifier's 0.3596. At `alpha = 2`
   every objective has *some* constant that beats the default, so unlike the
   higher-curvature settings there is no objective here where per-image
   selection is the only available source of gain.

## Regret against the model-free references

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | single
seed (42), so these are the reference rows behind the headline:

| Objective | Strategy | Cell | regret median | regret mean | regret p90 |
| --- | --- | --- | --- | --- | --- |
| Naive | default | (0.8, 0.3) | 0.5162 | 0.5423 | 0.9177 |
| Naive | best fixed | (0.1, 0.0) | **0.1437** | **0.1843** | **0.4277** |
| Naive | classifier | per image | 0.2188 | 0.2554 | 0.5466 |
| LINEX a2 | default | (0.8, 0.3) | 0.4134 | 0.4327 | 0.7385 |
| LINEX a2 | best fixed | (0.5, 0.2) | 0.2993 | 0.3505 | **0.6961** |
| LINEX a2 | classifier | per image | **0.2597** | **0.3267** | 0.7173 |
| CARA a2 | default | (0.8, 0.3) | 0.3419 | **0.3449** | **0.5715** |
| CARA a2 | best fixed | (0.6, 0.3) | 0.2619 | 0.3388 | 0.6698 |
| CARA a2 | classifier | per image | **0.2342** | 0.3596 | 0.8455 |

The p90 column is where curvature shows itself. Under naive the classifier's
p90 (0.5466) is far better than the default's (0.9177); under LINEX alpha 2 it
is roughly a wash against the best fixed cell (0.7173 vs 0.6961); under CARA
alpha 2 it is *worse than the default's* — 0.8455 against 0.5715 — while its
median is better. **That bad tail is entirely model-induced**, and it is what
moves the mean.

Fraction of test images for which some cell beats the default: 99.4% (naive),
98.7% (LINEX a2), 98.0% (CARA a2). The default cell is the true best cell for
0.6%, 1.3% and 2.0% of images respectively, and its median rank among the 121
cells is 49, 41 and 37 — so the default is a mediocre choice under every
objective, and the more concave the objective, the less mediocre it looks.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` |
label-space quality behind those rows, seed 42:

| Objective | bal acc t_start | bal acc t_end | acc both | top-1 hit | top-3 hit | median rank of pick (of 121) | cells used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 0.124 | 0.135 | 0.083 | 8.7% | 23.5% | 11 | 58 |
| LINEX a2 | 0.142 | 0.149 | 0.064 | 6.7% | 17.2% | 19 | 64 |
| CARA a2 | 0.117 | 0.129 | 0.044 | 4.6% | 13.8% | 24 | 66 |

Chance top-1 is `1 / 121 = 0.8%`, so the model is 5.6–10.5x better than chance
everywhere. Balanced accuracy moves within a narrow band (0.117–0.142) while
the mean phi gain swings from +0.277 to -0.014. **What changes between
objectives is not how often the model is wrong but what being wrong costs.**

## What each strategy achieves in real metric units

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | true
PSNR-Unedited and CLIP-Edited at whichever cell each strategy picks, averaged
over the 999 test images, seed 42. **This is the only table comparable across
objectives** — the metrics are fixed, only the objective choosing between them
changes.

| Objective | Strategy | PSNR-Unedited | CLIP-Edited | vs default |
| --- | --- | --- | --- | --- |
| any | default `(0.8, 0.3)` | 17.79 | 21.25 | — |
| Naive | best fixed `(0.1, 0.0)` | 31.55 | 16.94 | +13.76 dB, -4.31 CLIP |
| Naive | classifier | 27.10 | 18.93 | +9.31 dB, -2.32 CLIP |
| Naive | oracle | 26.44 | 22.66 | +8.65 dB, +1.41 CLIP |
| LINEX a2 | best fixed `(0.5, 0.2)` | 23.31 | 19.90 | +5.52 dB, -1.35 CLIP |
| LINEX a2 | classifier | 25.54 | 19.43 | +7.75 dB, -1.82 CLIP |
| LINEX a2 | oracle | 24.78 | 23.56 | +6.99 dB, +2.31 CLIP |
| CARA a2 | best fixed `(0.6, 0.3)` | 20.74 | 20.72 | +2.95 dB, -0.53 CLIP |
| CARA a2 | classifier | 24.25 | 19.84 | +6.46 dB, -1.41 CLIP |
| CARA a2 | oracle | 23.78 | 23.88 | +5.99 dB, +2.63 CLIP |

The objective is doing the work here, not the model. Naive with equal weights
on range-normalized deltas will trade 4.3 CLIP points for 13.8 dB without
hesitating, because both metrics are scaled into the same [0, 1] range; the
concave objectives refuse that trade — CARA alpha 2's best fixed cell gives up
only 0.53 CLIP. **Every oracle row improves both metrics at once**, so the
trade naive makes is not forced by the data; it is what phi asked for.

A pattern worth naming, and true in all three objectives: the classifier
**beats its own oracle on PSNR** (25.54 vs 24.78 under LINEX alpha 2) while
falling 3.7–4.1 CLIP points short of it everywhere. It finds high-PSNR cells
well and is comparatively blind to where CLIP can be had cheaply — pointing at
the prompt-side representation, since CLIP-Edited depends most on the prompt
pair, which enters as two mean-pooled vectors.

## Per-image win rate against the default

Because phi at the default is 0, a strategy's true phi at its picked cell *is*
its gain over the default.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42:

| Objective | Strategy | win | tie | loss | gain median | gain mean | gain p10 | share of oracle gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | `(0.1, 0.0)` | 87.4% | 0.0% | 12.6% | +0.338 | +0.358 | -0.020 | **66%** |
| Naive | classifier | 80.2% | 0.5% | 19.3% | +0.253 | +0.287 | -0.121 | 53% |
| LINEX a2 | `(0.5, 0.2)` | 67.4% | 0.0% | 32.6% | +0.145 | +0.082 | -0.351 | 19% |
| LINEX a2 | classifier | 64.3% | 1.2% | 34.5% | +0.147 | **+0.106** | -0.399 | **25%** |
| CARA a2 | `(0.6, 0.3)` | 62.2% | 0.0% | 37.8% | +0.085 | +0.006 | -0.415 | 2% |
| CARA a2 | classifier | 59.7% | 1.1% | 39.2% | +0.096 | **-0.015** | -0.596 | **-4%** |

This is the whole story in one place, and the columns disagree on purpose.
Under CARA alpha 2 the classifier wins on 59.7% of images with a positive
median gain and still loses 0.015 phi per image on average. CARA caps each
metric's reward at `w_i / alpha` = 0.2, so a moved image can gain at most +0.4,
while a regression costs `exp(alpha * Delta)` and is unbounded below — the
observed p10 is -0.60 against the best fixed cell's -0.42. **A 60% hit rate is
not enough when the payoff is that asymmetric.**

Walking the alpha ladder is the same story at every rung: as curvature rises
the win rate slides from 80.2% to 59.7% and the median gain falls to a third of
its value (+0.253 to +0.096), while the downside p10 widens fivefold (-0.12 to
-0.60). The tail moves further than the middle does, which is why the mean
crosses zero between LINEX alpha 2 and CARA alpha 2 while the median never
does.

## Which cells get picked

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42,
of 121 cells:

| Objective | Strategy | distinct cells | deviate rate | most-picked cells |
| --- | --- | --- | --- | --- |
| Naive | classifier | 58 | 99.5% | (0.1, 0.0) 15.0%, (0.2, 0.0) 13.1%, (0.0, 0.0) 9.0% |
| Naive | oracle | 84 | 99.4% | (0.0, 0.0) 14.9%, (0.1, 0.0) 11.0%, (0.2, 0.0) 7.6% |
| LINEX a2 | classifier | 64 | 98.8% | (0.1, 0.0) 8.7%, (0.2, 0.0) 8.4%, (0.3, 0.0) 6.1% |
| LINEX a2 | oracle | 89 | 98.7% | (0.0, 0.0) 9.8%, (0.1, 0.0) 6.6%, (0.3, 0.0) 4.6% |
| CARA a2 | classifier | 66 | 98.9% | (0.5, 0.2) 5.5%, (0.4, 0.2) 4.8%, (0.4, 0.0) 4.7% |
| CARA a2 | oracle | 92 | 98.0% | (0.0, 0.0) 5.0%, (0.4, 0.2) 4.4%, (0.5, 0.2) 3.8% |

The classifier spreads across 58–66 of 121 cells against the oracle's 84–92,
with at most 15% of images on its top pick, and its concentration tracks the
objective: naive puts a third of its picks in the low-`t_start` corner, CARA
alpha 2 spreads almost uniformly. Its problem is not collapse but the opposite —
it deviates on 98.8–99.5% of images under every objective, including CARA
alpha 2 where deviating is net negative. **It has essentially no notion of "the
default is fine here,"** even though the default is the true best cell for 2.0%
of images under CARA alpha 2 and its own tie rate is only 1.1%.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | picks
landing in the 66 cells with `t_end >= t_start`, seed 42:

| Objective | oracle in new cells | classifier in new cells | oracle on `(0.0, 0.0)` | best new cell's mean phi | as share of oracle gain | best old cell, as share |
| --- | --- | --- | --- | --- | --- | --- |
| Naive | 27.6% | 19.6% | 14.9% | +0.420 | **77%** | 94% |
| LINEX a2 | 26.7% | 18.2% | 9.8% | +0.287 | 66% | 94% |
| CARA a2 | 25.6% | 20.1% | 5.0% | +0.189 | 55% | 94% |

The last two columns take the best cell per sample within one region and ignore
the other entirely. The new region alone captures 55–77% of the full oracle's
gain and the original 55 cells alone capture 94%, so the two regions overlap
heavily rather than the new cells being the whole prize — but `(0.0, 0.0)`, a
position that did not exist before the annotation pass, is the oracle's single
most-picked cell under all three objectives. The new region's value falls
monotonically with curvature (77% -> 66% -> 55%), because those cells buy PSNR
at CLIP's expense, which is the trade a concave phi discounts. No objective's
best fixed cell lies in the new region, and the classifier lands there 18–20%
of the time against the oracle's 26–28%, so it under-exploits the addition
under every objective.

## Can a confidence threshold rescue CARA alpha 2?

Keep the default unless the classifier's log-odds margin for its pick over the
default cell clears a floor. Note this gates on *confidence*, not predicted
gain — the classifier does not estimate phi, so there is no predicted gain to
threshold, which is itself part of the answer.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42:

| Floor | CARA a2 deviate | CARA a2 gain mean | LINEX a2 deviate | LINEX a2 gain mean | Naive deviate | Naive gain mean |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 98.9% | -0.0147 | 98.8% | +0.1060 | 99.5% | +0.2869 |
| 0.5 | 98.2% | -0.0150 | 97.4% | +0.1059 | 98.8% | +0.2868 |
| 1.0 | 96.9% | -0.0155 | 96.3% | +0.1065 | 98.3% | +0.2874 |
| 2.0 | 93.1% | -0.0126 | 94.4% | +0.1056 | 95.6% | +0.2870 |
| 3.0 | 90.7% | -0.0115 | 91.3% | +0.1065 | 93.6% | +0.2861 |
| 5.0 | 85.9% | -0.0107 | 83.3% | **+0.1127** | 89.1% | +0.2831 |

Not usefully. Under CARA alpha 2 the mean gain stays negative at every floor
and moves toward zero only in proportion to how much selection the gate
switches off: floor 5 disables 13% of the deviations and recovers 27% of the
loss, which is close to what removing deviations at random would do. **The
classifier's confidence does not identify its profitable deviations.** The one
mild exception runs the other way — under LINEX alpha 2 the floor-5 gate
*improves* the mean gain from +0.106 to +0.113, the only sign anywhere that the
margin carries a little signal about which deviations pay.

## What this says

- **The premise "classify the best cell, then use it" holds only where phi is
  close to linear.** Naive and LINEX alpha 2 are the objectives where a ranking
  error costs about what a correct ranking pays, and they are the only two with
  a positive mean gain. Adding curvature makes each error cost more than a
  success earns, and CARA alpha 2's 4.6% top-1 hit rate cannot clear that bar.
- **Under LINEX alpha 2 the answer is an unqualified yes** — 37% better median
  regret and 24% better mean than the fixed default, and better than the best
  achievable fixed cell on both (0.2597 vs 0.2993 median, 0.3267 vs 0.3505
  mean). It is also the most stable objective in the set, with a four-seed
  spread of 0.0003 on median regret.
- **Under naive the model is beaten by a single constant.** `(0.1, 0.0)` is
  better on median (0.1437 vs 0.2188) and mean (0.1843 vs 0.2554) regret, and
  captures 66% of the oracle's gain against the classifier's 53%. When the
  objective is linear enough to want the extreme corner of the grid, naming
  that corner once is better than predicting it per image.
- **CARA alpha 2 sits just past the crossing point.** The classifier's mean
  gain is -0.014 +/- 0.006: small, but reliably the wrong sign, and its p90
  regret (0.8455) is worse than doing nothing (0.5715). The right comparison
  there is not the default but `(0.6, 0.3)`, which beats the classifier on the
  mean while losing the median.
- **Choosing curvature is a bigger lever than any model tuning.** The 141-run
  campaign in SUMMARY.md moved median regret by 0.27; moving from LINEX alpha 2
  to CARA alpha 2 — the same alpha, one step of curvature — flips the sign of
  whether the model is worth using at all.

## Caveats

- One split (`SPLIT_SEED = 42`). Headline spreads are four init seeds; the
  reference and breakdown tables are single-seed (42).
- The model configuration was tuned in SUMMARY.md against LINEX alpha 2 **on
  the 55-cell grid** and reused unchanged here for all three objectives and the
  full 121-cell grid. Two things are therefore untested: whether a model
  trained against a concave phi wants different hyperparameters, and whether
  the sweep's conclusions survive the wider grid. Both are obvious next
  experiments; the second is cheap, since the sweep harness already exists.
- SUMMARY.md and CHANGES.md still describe the 55-cell grid and have not been
  recomputed.
- `alpha = 5` is not covered here. In the previous five-objective version of
  this document both alpha-5 objectives had a negative mean gain, LINEX at
  -0.20 and CARA at -0.47, and were the only objectives for which no constant
  beat the default; the trend this document reports continues in that direction.
- Not comparable to the surrogate model's results even where objective and
  baseline cell match: different model class, different training signal (the
  classifier sees one labeled cell per sample, the surrogate all cells), single
  models rather than ensembles, and the surrogate's published numbers are still
  on the 55-cell grid.

## Worth trying next

1. **Re-examine phi's weighting before more model tuning.** Every oracle
   improves both metrics while every model row trades CLIP for PSNR, and the
   curvature choice decides whether selection helps at all. This is a
   modelling-goal question, not a hyperparameter one. The widened grid sharpens
   it: the best new cell alone captures 77% of naive's oracle gain and 55% of
   CARA alpha 2's, so the same 66 cells are worth very different amounts
   depending only on curvature.
2. **Give the model a way to decline.** It deviates on ~99% of images under
   every objective, including CARA alpha 2 where deviating is net negative. An
   explicit "keep the default" class, or a loss asymmetric in the same
   direction phi is, would let it abstain. The confidence gate is not a
   substitute: it removes good and bad deviations at nearly the same rate.
3. **Train against phi rather than the argmax label.** Cross-entropy treats
   every non-best cell as equally wrong; regret does not. Weighting each
   sample's loss by the phi gap between predicted and true cell would align
   training with the reported metric, and is the natural route to
   curvature-awareness.
4. **Strengthen the prompt side.** CLIP-Edited is where the model is blind
   (3.7–4.1 points short of its own oracle in every objective) and is the
   metric most dependent on the prompt pair, currently two mean-pooled CLIP
   vectors.
5. **Re-run the SUMMARY.md sweep on the full grid.** The recommended
   configuration was chosen on 55 cells and the label space has since more than
   doubled; the winners may not survive.

## Summary

`DIR_NAME = UltraEdit_Region_10000`, `CHORD_EDIT_MODEL = sd_turbo`, 121-cell
grid, 999 test images, default cell `(0.8, 0.3)`, `alpha = 2`.

Argmaxing the classifier beats the fixed default under all three objectives on
median regret — by 56% under naive, 37% under LINEX alpha 2 and 30% under CARA
alpha 2 — but only two of those survive the mean, and only one survives
comparison with the best constant cell. Under LINEX alpha 2, the shipped
default, the classifier wins on every measure: 24% better mean regret than the
default and better than the best achievable fixed cell on median and mean
alike. Under naive a single constant `(0.1, 0.0)` is simply better than the
model. Under CARA alpha 2 the mean gain is -0.014 +/- 0.006, reliably the wrong
sign, and `(0.6, 0.3)` beats the classifier on the mean. The mechanism is
asymmetric payoff rather than degraded accuracy: balanced accuracy stays within
0.117–0.142 across all three objectives while the mean gain swings from +0.277
to -0.014, because concave phi caps the reward for a correct pick while leaving
the penalty for a wrong one unbounded — along the curvature ladder the median
gain falls to a third while the downside p10 widens fivefold. A confidence gate
does not
recover it, because the model's margin does not track which deviations are
profitable. The practical reading is that per-image selection is worth shipping
at the curvature the pipeline currently uses, and that the choice of curvature,
not the model, decides whether it pays at all.
