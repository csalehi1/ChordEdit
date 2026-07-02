# Grid Pruning Strategy Comparison

Investigates whether the full 11×11 (121-cell) `t_start`×`t_end` grid search
needed per image can be replaced with a cheaper search that still finds a
near-optimal cell, to speed up scaling the oracle-labeling pipeline from
~1,000 to ~10,000 images.

Script: [scripts/grid_pruning_comparison.py](scripts/grid_pruning_comparison.py)

## Setup

- Real 700-image dense grid data (11×11 `t_start`×`t_end`, `t_delta=0.0`):
  `/shared/ssd_30T/salehi/id_to_metrics_sdturbo_fullgrid_unclamped_tdelta0_TRUE_paper_matching_format.csv`
- Scoring: the real `compute_softplus_score` function from this branch
  (`models/classification/utils.py`), imported directly (not reimplemented),
  using this branch's active settings: baseline `t_start=0.9, t_end=0.3`,
  `alpha=1.0`, `beta=2.0`, per-image normalization.
- Per-cell cost basis for time projections: 197.1ms GPU generation (using
  Daniel's factorized/optimized pipeline) + 177.0ms CLIP+PSNR scoring + 2.0ms
  JPEG save = 376.1ms/cell, measured directly on this machine's GPU.

## Strategies tested

- **Coordinate descent** (~21 cells): fix `t_end=0.3` (the app's default),
  sweep all `t_start` values for the best one; fix that `t_start`, sweep all
  `t_end` values for the best one.
- **Diagonal restriction** (66 cells): only search cells where `t_end <= t_start`.
- **Coarse-to-fine** (40 cells): sweep a coarse 6×6 grid (step 0.2), take the
  best cell, refine it into its 4 neighboring sub-cells at ±0.1, take the best
  of those.

## Headline results (n=700 images)

| Strategy | Cells used | % of full grid | Exact match | Avg. score percentile | Avg. score gap | Time @ 10,000 images | Time saved vs. full grid |
|---|---|---|---|---|---|---|---|
| Full 11×11 grid | 121 | 100% | 100% (reference) | 100.0 | 0.0000 | 5.27 days | — |
| **Diagonal restriction** | 66 | 55% | **69.3%** | **99.7** | **0.0055** | 2.87 days | 2.39 days (45%) |
| Coarse-to-fine | 40 | 33% | 56.4% | 99.2 | 0.0142 | 1.74 days | 3.53 days (67%) |
| Coordinate descent | 21 | 17% | 42.1% | 98.6 | 0.0174 | 0.91 days | 4.35 days (83%) |

Score gaps are small relative to the typical per-image score spread
(mean ≈0.632 across the 121 cells): even coordinate descent's mean gap is
only ≈2.8% of that range.

## How bad are the misses, specifically?

Stats computed only over images each strategy got wrong (excludes exact
matches, which trivially contribute zero):

| Strategy | # misses | Median gap | Mean gap | Worst-case gap | Worst as % of typical range | Worst percentile |
|---|---|---|---|---|---|---|
| Diagonal restriction | 215 | 0.0063 | 0.0179 | 0.1439 | 22.8% | 95.8% |
| Coarse-to-fine | 305 | 0.0188 | 0.0326 | 0.1894 | 30.0% | 84.2% |
| Coordinate descent | 405 | 0.0169 | 0.0300 | 0.2529 | 40.0% | 86.7% |

Miss severity distribution (score percentile of the chosen cell, when wrong):

| Strategy | ≥99th %ile (essentially tied) | 95th-99th (very close) | 90th-95th (close) | <90th (real miss) |
|---|---|---|---|---|
| Diagonal restriction | 74.9% | 25.1% | 0.0% | **0.0%** |
| Coarse-to-fine | 46.9% | 48.5% | 3.9% | 0.7% (2 images) |
| Coordinate descent | 39.8% | 51.6% | 7.9% | 0.7% (3 images) |

Diagonal restriction never produces a genuinely bad choice across all 700
images. Coordinate descent and coarse-to-fine each have a small number of
real failures (2-3 images out of 700), and even their worst case is
22-40% of a typical score range — noticeable but not severe.

## What a "miss" looks like in real metric terms

| Strategy | Mean PSNR — true best | Mean PSNR — found | PSNR diff (best − found) | Mean CLIP-Edit — true best | Mean CLIP-Edit — found | CLIP diff (best − found) |
|---|---|---|---|---|---|---|
| Coordinate descent | 26.093 | 26.953 | -0.860 | 24.656 | 23.999 | 0.657 |
| Diagonal restriction | 26.093 | 26.370 | -0.277 | 24.656 | 24.425 | 0.231 |
| Coarse-to-fine | 26.093 | 26.969 | -0.876 | 24.656 | 24.017 | 0.640 |

Negative PSNR diff means the found cell has *higher* PSNR than the true
best; positive CLIP diff means the found cell has *lower* CLIP-Edit. This
pattern (majority but not universal — see below) is consistent across all
three strategies.

Restricted to misses only, and checking per-image direction rather than
just the aggregate mean:

| Strategy | % of misses with found-PSNR higher | % of misses with found-CLIP lower |
|---|---|---|
| Coordinate descent | 57.0% | 81.5% |
| Diagonal restriction | 57.2% | 80.9% |
| Coarse-to-fine | 70.8% | 85.6% |

Not universal — 30-43% of misses go the opposite direction — but a real
majority tendency, especially on the CLIP side.

### Why this happens

`t_start` and `t_end` each independently trade PSNR for CLIP-Edit:

- corr(`t_start`, PSNR) = -0.53, corr(`t_start`, CLIP) = +0.15
- corr(`t_end`, PSNR) = -0.52, corr(`t_end`, CLIP) = +0.13

Turning either parameter up means more edit strength: lower PSNR (further
from source), higher CLIP (better target alignment). Each strategy's miss
mechanism nudges one of these two parameters toward "weaker than optimal":

- **Coordinate descent / coarse-to-fine**: when they miss, the found cell's
  `t_start` is on average lower than the true best's (-0.093 / -0.135),
  correlating strongly with the PSNR/CLIP pattern (r≈0.74-0.77 with PSNR
  diff, r≈-0.68 to -0.71 with CLIP diff).
- **Diagonal restriction**: its misses only happen when the true optimum
  sits in the excluded region (`t_end > t_start`) — 30.7% of images (215/700),
  exactly matching its miss rate. In those cases its found cell is forced to
  a lower `t_end` than the true best's (-0.227 on average), and that forced
  reduction correlates just as strongly with the same PSNR/CLIP pattern
  (r≈0.66, r≈-0.64) — same effect, different parameter.

## Conclusion

**Diagonal restriction is the strongest candidate**: 45% fewer cells than
the full grid, 69.3% exact match, and its failure mode is fully understood,
bounded (only ever misses when the true optimum requires `t_end > t_start`),
and never produces a genuinely bad cell (0% of misses fall below the 90th
percentile). Coarse-to-fine is a reasonable cheaper alternative (67% fewer
cells) with weaker guarantees. Coordinate descent, despite being cheapest,
relies on an independence assumption between `t_start` and `t_end` that does
not hold in this data and has the least explainable failure pattern.

## Caveats

- All numbers here use this branch's *current* `compute_softplus_score`
  settings (`alpha=1.0`, `beta=2.0`, baseline `t_start=0.9`/`t_end=0.3`).
  Results would need to be recomputed if those settings change.
- Based on 700 images from one dataset run; worth re-validating on the next
  batch of newly-labeled images once available, rather than assuming these
  exact percentages transfer unchanged.
