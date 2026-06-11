# Noise Sample Ablation

Dataset: PIE-Bench (700 samples)

Model: SD-Turbo

Fixed settings:

* t_start = 0.90
* t_end = 0.30
* t_delta = 0.15
* step_scale = 1.0
* n_steps = 1
* seed = 42

## Results

| Noise Samples | Structure Distance ↓ |    PSNR ↑ |  LPIPS ↓ |    MSE ↓ |   SSIM ↑ | CLIP Target ↑ | CLIP Edited ↑ |
| ------------: | -------------------: | --------: | -------: | -------: | -------: | ------------: | ------------: |
|             1 |             0.028794 | 22.632562 | 0.122644 | 0.007985 | 0.761944 |     24.814136 |     22.130849 |
|             2 |             0.021944 | 23.632401 | 0.106055 | 0.006313 | 0.777093 |     24.227493 |     21.445531 |
|             3 |             0.020424 | 24.013761 | 0.101176 | 0.005806 | 0.782262 |     23.972224 |     21.230913 |
|             4 |             0.019962 | 24.172914 | 0.099098 | 0.005613 | 0.784511 |     23.824532 |     21.064311 |

## Observations

Increasing the number of noise samples improves preservation-oriented metrics, like structure distance, PSNR, and LPIPS. Smaller improvements are found in the MSE and SSIM. This contradicts what the paper said, which was that increasing the number of noise samples was not necessary and did not provide substantial improvement to these metrics.

Also, increasing the number of noise samples reduces semantic editing strength, as found in the decreases in CLIP Target and Edited scores.

Therefore, there appears to be a tradeoff between image preservation and edit strength. More Monte Carlo samples improve structural fidelity, but don't align with the target edit prompt as much.
