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

**These numbers are computed on the full 121-cell grid, and the model
configuration was re-tuned on that grid.** The annotation pipeline has produced
the 66 cells with `t_end >= t_start`, so the metrics CSV now covers all 11 x 11
positions rather than the 55 of the strict lower triangle. The usable sample set
is unchanged (9985 samples, 999 test), so this is purely a widening of the
candidate set: every strategy chooses from 121 cells instead of 55, and phi's
per-sample normalization ranges over all of them. Fifteen samples in the CSV are
unusable — fourteen carry metrics *only* for the added cells and one has no valid
metrics at all — and are dropped in `_data.load_df` for having an incomplete
grid, recovering exactly the previous sample set.

A previous version of this document inherited its hyperparameters from the
[SUMMARY.md](SUMMARY.md) campaign, which was run on the 55-cell grid. That gap
is now closed: a 63-run campaign on the full grid
([below](#re-tuning-on-the-full-grid)) **re-selects the same configuration**, so
the numbers are unchanged and are now tuned-on-121 rather than inherited. The
setup is class weighting off, CORAL head, cosine decay, 20 epochs,
`(DEFAULT_T_START, DEFAULT_T_END) = (0.8, 0.3)`, retrained once per objective at
`SEED = 42`, plus a four-seed repeat of each for spreads. Selection is the joint
argmax of the two heads over the 121 cells: no gate, no shrinkage, no ensemble.
Test split, 999 images. Runs are
`outputs/UltraEdit_Region_10000/a2_{naive,linex_a2,cara_a2}_s{42,1,2,3}`.

**The answer depends on the objective's curvature, and the median regret alone
will tell you the wrong thing.** Median regret improves under all three
objectives. Mean regret improves under only two. Under CARA alpha 2 the
classifier makes the median image better and the average image slightly worse
than if it had never deviated from the default. Only under LINEX alpha 2 does
the classifier beat both the default and the best possible constant cell on
median and mean at once.

**And the strongest single lever found here is not a hyperparameter but which
cells the training labels are drawn from.** Restricting the *label* space to the
55-cell lower triangle while still letting the model choose from all 121 cells at
inference flips CARA alpha 2's mean gain from reliably negative to reliably
positive, at 4.1 sigma. See
[which grid the model is tuned and tested on](#which-grid-the-model-is-tuned-and-tested-on).

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
| **`CELL_SUBSET`** | new settings key selecting the candidate set: `"all"` (121) or `"lower"` (the 55 with `t_end < t_start`). It changes the labels and phi's normalization range, not merely the argmax domain. |

Phi is objective-specific, so numbers are comparable only *within* an objective —
except the raw-metric table, which is comparable across all of them. Phi is also
**grid-specific**: see the [scale warning](#a-warning-about-scale) before reading
55-cell numbers against 121-cell ones.

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

## Re-tuning on the full grid

The configuration in `settings.json` was chosen by the 141-run campaign in
SUMMARY.md, entirely on the 55-cell lower triangle. The label space has since
more than doubled, so the campaign was repeated on the full grid: a 31-config
one-factor-at-a-time round off the current defaults, then eight survivors over
four init seeds each. Runs are `runs/sweep_g1_*` and `runs/sweep_g1f_*`, all at
LINEX alpha 2 with `CELL_SUBSET = all`.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | 121 cells,
LINEX alpha 2, four init seeds per row, ordered by regret mean:

| Configuration (4 seeds) | val regret median | regret median | regret mean | gain mean | p90 | best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| **base — the shipped config** | 0.2530 | **0.2593 +/- 0.0003** | **0.3283 +/- 0.0043** | **+0.1044** | **0.7131** | 12.5 |
| LR 3e-5 | 0.2375 | 0.2644 +/- 0.0127 | 0.3287 +/- 0.0054 | +0.1040 | 0.7114 | 14.8 |
| batch size 256 | 0.2411 | 0.2671 +/- 0.0097 | 0.3305 +/- 0.0092 | +0.1021 | 0.7259 | 15.2 |
| wider MLP (1024/512/256) | 0.2474 | 0.2621 +/- 0.0016 | 0.3306 +/- 0.0046 | +0.1021 | 0.7235 | 14.2 |
| narrow MLP (128/64/32) | 0.2477 | 0.2616 +/- 0.0041 | 0.3308 +/- 0.0066 | +0.1018 | 0.7239 | 15.8 |
| checkpoint on regret | 0.2402 | 0.2687 +/- 0.0045 | 0.3332 +/- 0.0081 | +0.0994 | 0.7198 | 6.2 |
| batch size 16 | 0.2490 | 0.2668 +/- 0.0074 | 0.3337 +/- 0.0062 | +0.0990 | 0.7229 | 14.2 |
| checkpoint on regret + bs 256 | 0.2314 | 0.2660 +/- 0.0138 | 0.3384 +/- 0.0141 | +0.0942 | 0.7478 | 10.8 |

**Nothing beats the incumbent, and the incumbent is not measurably better than
the field either.** It leads on all four test columns, but the gap to the
runner-up on regret mean is 0.0004 against a pooled sigma of 0.0048 — **0.08
sigma**. The whole eight-config span on regret mean is 0.010, about two single-run
noise bars. The honest reading is that the configuration is re-selected because
nothing displaces it, and its one real distinction is stability: a four-seed
median spread of 0.0003 against 0.0016–0.0138 for everything else.

The one-factor round tells the same story and reproduces SUMMARY.md's negative
findings on the wider grid. Twenty-eight of 31 configs land between 0.2355 and
0.2687 validation regret median. Only three choices are clearly bad, and they are
the same three as before: checkpointing on validation loss (0.3793, far worst, and
it collapses selection to 7 distinct cells), LR 1e-5 (0.2959, 16 cells — it
never converges inside the budget), and class weighting on (0.2773). Validation
and test also disagree on the ranking at this effect size — `bs16` has the best
validation regret of all 31 (0.2355) and a near-worst test median (0.2752) —
which is exactly why the finals use seeded repeats.

**So the sweep's conclusions survive the wider grid.** This closes the second
half of the previous version's main caveat. The first half — whether a model
trained against a *concave* phi wants different hyperparameters — is still open:
this campaign, like SUMMARY.md's, tuned at LINEX alpha 2 only.

## Which grid the model is tuned and tested on

Three (tune, test) combinations, holding the sample split fixed. "Tuned on" names
the candidate grid used to select hyperparameters *and* to build the training
labels; "tested on" names the grid the model is argmaxed over and scored against.
Because the two heads span the full 11 x 11 bucket space regardless, a model
trained on the 55-cell lower triangle can still be evaluated over all 121 cells,
which is the middle row.

### A warning about scale

phi normalizes each metric over the candidate cells *of that grid*, so
**55-cell phi and 121-cell phi are different scales** and the third row's
absolute numbers are not comparable to the first two. Empirically the two scales
land close — the default cell's mean regret is 0.4338 on 55 cells against 0.4327
on 121 under LINEX alpha 2, and the oracle's mean phi is 0.4291 against 0.4333 —
so the columns look comparable, but only the `vs default` columns actually are.
Read down the first two rows of each block for a like-for-like comparison, and
use `vs default` for the third.

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | 999 test
images, four init seeds per row, `SPLIT_SEED = 42`:

| Objective | Tuned on | Tested on | cells | regret median | vs default | regret mean | vs default | gain mean | p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 121 cells | 121 cells | 121 | **0.2286 +/- 0.0085** | **-56%** | **0.2649 +/- 0.0096** | **-51%** | **+0.2773 +/- 0.0096** | **0.5601** |
| Naive | 55 cells | 121 cells | 121 | 0.2492 +/- 0.0100 | -52% | 0.2888 +/- 0.0087 | -47% | +0.2535 +/- 0.0087 | 0.5967 |
| Naive | 55 cells | 55 cells | 55 | 0.2495 +/- 0.0085 | -52% | 0.2817 +/- 0.0105 | -48% | +0.2641 +/- 0.0105 | 0.5979 |
| LINEX a2 | 121 cells | 121 cells | 121 | **0.2593 +/- 0.0003** | -37% | 0.3283 +/- 0.0043 | -24% | +0.1044 +/- 0.0043 | 0.7131 |
| LINEX a2 | 55 cells | 121 cells | 121 | 0.2613 +/- 0.0048 | -37% | **0.3174 +/- 0.0042** | **-27%** | **+0.1153 +/- 0.0042** | **0.6719** |
| LINEX a2 | 55 cells | 55 cells | 55 | 0.2639 +/- 0.0042 | -37% | 0.3295 +/- 0.0037 | -24% | +0.1043 +/- 0.0037 | 0.7162 |
| CARA a2 | 121 cells | 121 cells | 121 | 0.2403 +/- 0.0059 | -30% | 0.3590 +/- 0.0055 | +4% | -0.0141 +/- 0.0055 | 0.8385 |
| CARA a2 | 55 cells | 121 cells | 121 | **0.2403 +/- 0.0074** | **-30%** | **0.3349 +/- 0.0064** | **-3%** | **+0.0100 +/- 0.0064** | **0.7426** |
| CARA a2 | 55 cells | 55 cells | 55 | 0.2436 +/- 0.0072 | -29% | 0.3617 +/- 0.0067 | +5% | -0.0175 +/- 0.0067 | 0.8390 |

Note that the *hyperparameter* half of "tuned on" makes no difference at all,
because the 121-cell campaign re-selected the configuration the 55-cell campaign
had already chosen. So the first row of each block is simultaneously
tuned-on-121 and hyperparameters-tuned-on-55; the runs are config-identical and
reproduce each other seed for seed. **Everything that separates the rows is
which cells the training labels came from.**

**Training on the 55-cell labels and testing on 121 is the best configuration in
this table for both concave objectives.** Under CARA alpha 2 it moves the mean
gain from -0.0141 +/- 0.0055 to +0.0100 +/- 0.0064 — a **4.1 sigma** swing that
flips the sign of the document's headline conclusion for that objective — and
cuts p90 regret from 0.8385 to 0.7426. Under LINEX alpha 2 it is worth +0.0109
mean gain at **2.6 sigma** and 0.041 of p90. Under naive it *costs* 0.0239 at
**-2.6 sigma**. The effect is monotone in curvature and changes sign between
naive and LINEX alpha 2.

The mechanism is that the restricted label space acts as a prior that keeps the
model out of the added region, and at the alpha the pipeline ships that region is
a trap:

`DIR_NAME = UltraEdit_Region_10000` | `CHORD_EDIT_MODEL = sd_turbo` | seed 42,
picks landing in the 66 cells with `t_end >= t_start`, both models free to choose
from all 121:

| Objective | oracle in new cells | 121-trained model in new | 55-trained model in new | 121-trained on `(0.0, 0.0)` | 55-trained on `(0.0, 0.0)` | cells used, 121- vs 55-trained |
| --- | --- | --- | --- | --- | --- | --- |
| Naive | 27.6% | 19.6% | **0.0%** | 9.0% | 0.0% | 58 vs 34 |
| LINEX a2 | 26.7% | 18.2% | **0.5%** | 4.7% | 0.0% | 64 vs 43 |
| CARA a2 | 25.6% | 20.1% | **0.5%** | 2.2% | 0.0% | 66 vs 42 |

The 55-trained model is *allowed* to pick any of the 121 cells and essentially
never does — 0.0–0.5% against the 121-trained model's 18–20% — and never lands on
`(0.0, 0.0)` at all. So the comparison is not "can it reach the new cells" but
"does going there pay," and the answer tracks curvature exactly. The added cells
buy PSNR at CLIP's expense (see the [raw-metric table](#what-each-strategy-achieves-in-real-metric-units));
a linear phi is happy to make that trade and a concave one is not, so the same
66 cells are an opportunity for naive and a liability for CARA alpha 2.

This is a cheap and immediately usable finding: **at the shipped curvature,
training on the 55-cell labels while selecting over all 121 cells beats training
on all 121 on mean and tail at no cost in median** — 0.2613 +/- 0.0048 against
0.2593 +/- 0.0003 is a 0.002 difference inside the wider of the two spreads —
and it costs one settings key and nothing at inference. It is also a sharper
version of "give the model a way to decline":
the restriction does not teach it to abstain, it removes the region where its
mistakes are most expensive.

Two smaller structural differences the third row exposes:

- **On the 55-cell grid under CARA alpha 2, no constant beats the default.** The
  best fixed cell *is* `(0.8, 0.3)` itself, so on that grid the classifier's
  negative mean gain (-0.0175) has no constant-cell alternative to lose to,
  and per-image selection is the only available source of gain. Widening to 121
  cells is what gives `(0.6, 0.3)` its 0.3388 mean regret. The widened grid
  therefore *raises* the bar the model has to clear, not just its own ceiling.
- **Widening the grid makes the default look worse, but only slightly, and by
  less than the cell count does.** Its median rank goes from 26 of 55 to 49 of
  121 under naive, 22/55 to 41/121 under LINEX alpha 2, and 19/55 to 37/121 under
  CARA alpha 2 — i.e. from the 47th/40th/35th percentile of cells to the
  40th/34th/31st. The fraction of images for which some cell beats it rises from
  99.2%/97.9%/97.4% on 55 cells to 99.4%/98.7%/98.0% on 121. So the added cells
  do contain better choices for most images; the model just cannot reliably tell
  which.

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
moves the mean. Restricting the training labels to the 55-cell region recovers
much of it (0.7445 at seed 42, from 0.8455).

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
| Naive | classifier, 55-cell labels | 26.10 | 19.34 | +8.31 dB, -1.91 CLIP |
| Naive | oracle | 26.44 | 22.66 | +8.65 dB, +1.41 CLIP |
| LINEX a2 | best fixed `(0.5, 0.2)` | 23.31 | 19.90 | +5.52 dB, -1.35 CLIP |
| LINEX a2 | classifier | 25.54 | 19.43 | +7.75 dB, -1.82 CLIP |
| LINEX a2 | classifier, 55-cell labels | 24.22 | 19.97 | +6.43 dB, -1.29 CLIP |
| LINEX a2 | oracle | 24.78 | 23.56 | +6.99 dB, +2.31 CLIP |
| CARA a2 | best fixed `(0.6, 0.3)` | 20.74 | 20.72 | +2.95 dB, -0.53 CLIP |
| CARA a2 | classifier | 24.25 | 19.84 | +6.46 dB, -1.41 CLIP |
| CARA a2 | classifier, 55-cell labels | 23.48 | 20.22 | +5.69 dB, -1.03 CLIP |
| CARA a2 | oracle | 23.78 | 23.88 | +5.99 dB, +2.63 CLIP |

The objective is doing the work here, not the model. Naive with equal weights
on range-normalized deltas will trade 4.3 CLIP points for 13.8 dB without
hesitating, because both metrics are scaled into the same [0, 1] range; the
concave objectives refuse that trade — CARA alpha 2's best fixed cell gives up
only 0.53 CLIP. **Every oracle row improves both metrics at once**, so the
trade naive makes is not forced by the data; it is what phi asked for.

The 55-cell-label rows show the same thing from the other side: under every
objective, restricting the labels moves the model 0.8–1.3 dB down and 0.4–0.5
CLIP up, i.e. **toward** the oracle's balance and away from the PSNR corner. That is
why it helps the concave objectives and hurts the linear one.

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
| Naive | classifier, 55-cell labels | 80.6% | 0.6% | 18.8% | +0.242 | +0.266 | -0.114 | 49% |
| LINEX a2 | `(0.5, 0.2)` | 67.4% | 0.0% | 32.6% | +0.145 | +0.082 | -0.351 | 19% |
| LINEX a2 | classifier | 64.3% | 1.2% | 34.5% | +0.147 | **+0.106** | -0.399 | **25%** |
| LINEX a2 | classifier, 55-cell labels | 65.3% | 1.8% | 32.9% | +0.135 | **+0.115** | -0.312 | **27%** |
| CARA a2 | `(0.6, 0.3)` | 62.2% | 0.0% | 37.8% | +0.085 | +0.006 | -0.415 | 2% |
| CARA a2 | classifier | 59.7% | 1.1% | 39.2% | +0.096 | **-0.015** | -0.596 | **-4%** |
| CARA a2 | classifier, 55-cell labels | 62.2% | 1.7% | 36.1% | +0.095 | **+0.017** | -0.482 | **+5%** |

This is the whole story in one place, and the columns disagree on purpose.
Under CARA alpha 2 the classifier wins on 59.7% of images with a positive
median gain and still loses 0.015 phi per image on average. CARA caps each
metric's reward at `w_i / alpha` = 0.2, so a moved image can gain at most +0.4,
while a regression costs `exp(alpha * Delta)` and is unbounded below — the
observed p10 is -0.60 against the best fixed cell's -0.42. **A 60% hit rate is
not enough when the payoff is that asymmetric.**

The 55-cell-label rows isolate where the fix comes from: the win rate barely
moves (59.7% -> 62.2% under CARA alpha 2) while **p10 improves from -0.596 to
-0.482**. The restriction does not make the model right more often; it makes
being wrong cheaper, which is exactly the quantity a concave phi is sensitive
to.

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
| Naive | classifier, 55-cell labels | 34 | 99.4% | (0.3, 0.0) 15.3%, (0.2, 0.0) 10.1%, (0.4, 0.1) 9.4% |
| Naive | oracle | 84 | 99.4% | (0.0, 0.0) 14.9%, (0.1, 0.0) 11.0%, (0.2, 0.0) 7.6% |
| LINEX a2 | classifier | 64 | 98.8% | (0.1, 0.0) 8.7%, (0.2, 0.0) 8.4%, (0.3, 0.0) 6.1% |
| LINEX a2 | classifier, 55-cell labels | 43 | 98.2% | (0.5, 0.2) 7.3%, (0.3, 0.1) 6.7%, (0.2, 0.0) 6.5% |
| LINEX a2 | oracle | 89 | 98.7% | (0.0, 0.0) 9.8%, (0.1, 0.0) 6.6%, (0.3, 0.0) 4.6% |
| CARA a2 | classifier | 66 | 98.9% | (0.5, 0.2) 5.5%, (0.4, 0.2) 4.8%, (0.4, 0.0) 4.7% |
| CARA a2 | classifier, 55-cell labels | 42 | 98.3% | (0.5, 0.2) 10.0%, (0.5, 0.3) 6.0%, (0.4, 0.2) 5.9% |
| CARA a2 | oracle | 92 | 98.0% | (0.0, 0.0) 5.0%, (0.4, 0.2) 4.4%, (0.5, 0.2) 3.8% |

The classifier spreads across 58–66 of 121 cells against the oracle's 84–92,
with at most 15% of images on its top pick, and its concentration tracks the
objective: naive puts a third of its picks in the low-`t_start` corner, CARA
alpha 2 spreads almost uniformly. Its problem is not collapse but the opposite —
it deviates on 98.8–99.5% of images under every objective, including CARA
alpha 2 where deviating is net negative. **It has essentially no notion of "the
default is fine here,"** even though the default is the true best cell for 2.0%
of images under CARA alpha 2 and its own tie rate is only 1.1%.

The 55-cell-label variant uses barely half as many distinct cells (34–43) and
deviates just as often, so it is not abstaining either — it is deviating within a
smaller and safer set.

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

That last sentence needs a caveat the previous version of this document did not
have: under-exploiting the new region is only a defect for naive. The 55-cell-label
runs show that going there **at all** is net negative for both concave
objectives, so for LINEX alpha 2 and CARA alpha 2 the right description is that
the model over-exploits it. The oracle's 26–28% is not a target the model should
be trying to hit, because the oracle knows which of those cells are safe and the
model does not.

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

Restricting the training labels is a far better lever than gating: it buys
+0.024 mean gain under CARA alpha 2 where the entire gate sweep buys +0.004,
and it does so without switching any selection off.

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
- **CARA alpha 2 sits just past the crossing point, and the label space is what
  pushes it over.** Trained on all 121 cells its mean gain is -0.014 +/- 0.006:
  small, but reliably the wrong sign. Trained on the 55-cell labels and still
  selecting over 121 it is +0.010 +/- 0.006 — reliably the right sign. The
  crossing point is a property of the training label space, not only of the
  objective.
- **Where the labels come from is a bigger lever than any hyperparameter.** The
  63-run campaign on the full grid moved nothing outside the noise bar; changing
  `CELL_SUBSET` from `all` to `lower` for the labels alone is worth 4.1 sigma of
  mean gain under CARA alpha 2 and 2.6 sigma under LINEX alpha 2. Two
  independent tuning campaigns now agree that this model's hyperparameters are
  not where its remaining headroom is.
- **Choosing curvature is a bigger lever still.** The 141-run campaign in
  SUMMARY.md moved median regret by 0.27; moving from LINEX alpha 2 to CARA
  alpha 2 — the same alpha, one step of curvature — flips the sign of whether
  the model is worth using at all.

## Caveats

- One split (`SPLIT_SEED = 42`) throughout. Headline and comparison spreads are
  four init seeds; the reference and breakdown tables are single-seed (42).
- **Both tuning campaigns tuned at LINEX alpha 2 only.** Whether a model trained
  against a concave phi wants different hyperparameters is still untested; that
  is now the only surviving half of the previous version's tuning caveat.
- The 55-cell-label result is a four-seed effect on one split, and it was found
  by evaluating a configuration built for a different purpose (reproducing the
  55-cell world) on the 121-cell grid. It should be confirmed on a second split
  before it is shipped, even though 4.1 sigma on the headline metric is a strong
  starting point.
- **phi is grid-relative.** 55-cell and 121-cell phi are different scales, so
  only the `vs default` columns of the comparison table are strictly comparable
  across the two test grids. The two scales happen to land within about 1% of
  each other on the default and oracle references, which is why the absolute
  columns look comparable, but that is an empirical coincidence of this dataset.
- SUMMARY.md and CHANGES.md still describe the 55-cell grid and have not been
  recomputed. SUMMARY.md's tuned-classifier row is reproduced exactly by the
  `g55_linex_a2_*` runs here (0.2639 +/- 0.0042 median, 0.3295 +/- 0.0037 mean,
  p90 0.7162), so its numbers remain correct as 55-cell numbers.
- `alpha = 5` is not covered here. In the previous five-objective version of
  this document both alpha-5 objectives had a negative mean gain, LINEX at
  -0.20 and CARA at -0.47, and were the only objectives for which no constant
  beat the default; the trend this document reports continues in that direction.
  Given the 55-cell-label result, the alpha-5 objectives are the most likely to
  be rescued by it and are the obvious next place to look.
- Not comparable to the surrogate model's results even where objective and
  baseline cell match: different model class, different training signal (the
  classifier sees one labeled cell per sample, the surrogate all cells), single
  models rather than ensembles, and the surrogate's published numbers are still
  on the 55-cell grid.

## Worth trying next

1. **Ship the restricted label space at the current curvature, and find the
   right restriction.** `CELL_SUBSET = "lower"` for the labels with selection
   over all 121 cells is better than the shipped configuration on mean and tail
   under both concave objectives, at no cost in median and no cost at inference.
   The lower triangle is almost certainly not the *optimal* restriction — it was
   chosen by what the annotation pipeline happened to label first. Selecting the
   label region by its measured phi risk, per objective, is the obvious
   generalization.
2. **Re-examine phi's weighting before more model tuning.** Every oracle
   improves both metrics while every model row trades CLIP for PSNR, and the
   curvature choice decides whether selection helps at all. This is a
   modelling-goal question, not a hyperparameter one. The widened grid sharpens
   it: the best new cell alone captures 77% of naive's oracle gain and 55% of
   CARA alpha 2's, so the same 66 cells are worth very different amounts
   depending only on curvature — and are net harmful to a concave phi in the
   model's hands.
3. **Give the model a way to decline.** It deviates on ~98–99% of images under
   every objective and every label space, including CARA alpha 2 where deviating
   is net negative. An explicit "keep the default" class, or a loss asymmetric in
   the same direction phi is, would let it abstain. Neither the confidence gate
   nor the label restriction is a substitute: the gate removes good and bad
   deviations at nearly the same rate, and the restriction changes *where* it
   deviates rather than *whether* it does.
4. **Train against phi rather than the argmax label.** Cross-entropy treats
   every non-best cell as equally wrong; regret does not. Weighting each
   sample's loss by the phi gap between predicted and true cell would align
   training with the reported metric, and is the natural route to
   curvature-awareness. The label-restriction result is a crude version of this
   done by hand — it removes the highest-variance region instead of pricing it —
   which suggests the principled version has more to give.
5. **Strengthen the prompt side.** CLIP-Edited is where the model is blind
   (3.7–4.1 points short of its own oracle in every objective) and is the
   metric most dependent on the prompt pair, currently two mean-pooled CLIP
   vectors.
6. **Stop sweeping hyperparameters.** Two campaigns, 204 runs, two different
   candidate grids, one conclusion: outside three known-bad choices
   (checkpointing on loss, LR 1e-5, class weighting on) the landscape is flat to
   within the seed noise.

## Summary

`DIR_NAME = UltraEdit_Region_10000`, `CHORD_EDIT_MODEL = sd_turbo`, 121-cell
grid, 999 test images, default cell `(0.8, 0.3)`, `alpha = 2`, hyperparameters
tuned on the 121-cell grid.

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
does not recover it, because the model's margin does not track which deviations
are profitable.

Re-tuning the model on the full 121-cell grid changes none of this: a 63-run
campaign re-selects the configuration the 55-cell campaign had already chosen,
and the eight seeded finalists span 0.010 on regret mean against sigmas near
0.005. What *does* change the conclusion is the training label space. Drawing
labels from the 55-cell lower triangle while still selecting over all 121 cells
is better on mean and tail under both concave objectives — under CARA alpha 2 it
flips the mean gain from -0.014 +/- 0.006 to +0.010 +/- 0.006 at 4.1 sigma, and
cuts p90 regret from 0.839 to 0.743 — because it keeps the model out of the 66
added cells, which buy PSNR at CLIP's expense and are therefore an opportunity
for a linear phi and a liability for a concave one. It works by making the
model's mistakes cheaper (gain p10 -0.60 to -0.48) rather than rarer (win rate
59.7% to 62.2%). The practical reading is that per-image selection is worth
shipping at the curvature the pipeline currently uses, that the choice of
curvature and of label space decide whether it pays at all, and that the
model's hyperparameters — now swept twice, on two grids — are not where the
remaining headroom is.
