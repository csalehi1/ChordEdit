# Chord Control Field Ablation: `t_delta`

Dataset: PIE-Bench, 700 samples  
Model: SD-Turbo  
Seed: 42  
Fixed settings: `t_start=0.90`, `t_end=0.30`, `noise_samples=1`, `n_steps=1`, `step_scale=1.0`

| t_delta | Rows | Structure Distance ↓ | PSNR ↑ | LPIPS ↓ | MSE ↓ | SSIM ↑ | CLIP Target ↑ | CLIP Edited ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 700 | 0.045552 | 20.131124 | 0.159816 | 0.013343 | 0.727029 | 25.114840 | 22.567368 |
| 0.05 | 700 | 0.038898 | 21.020055 | 0.144566 | 0.011158 | 0.740500 | 25.071978 | 22.500243 |
| 0.10 | 700 | 0.033988 | 21.812557 | 0.133982 | 0.009532 | 0.751256 | 24.994363 | 22.306170 |
| 0.15 | 700 | 0.028794 | 22.632562 | 0.122644 | 0.007985 | 0.761944 | 24.814136 | 22.130849 |
| 0.20 | 700 | 0.025622 | 23.166263 | 0.115325 | 0.007074 | 0.768772 | 24.656040 | 21.996045 |

## Observations

Increasing `t_delta` improves background and structural preservation:
- PSNR rises from 20.13 at `t_delta=0.00` to 23.17 at `t_delta=0.20`.
- LPIPS decreases from 0.1598 to 0.1153.
- Structure distance decreases from 0.0456 to 0.0256.
- SSIM increases from 0.7270 to 0.7688.

The tradeoff is that CLIP target/edit similarity gradually decreases as `t_delta` increases, suggesting stronger smoothing preserves structure better but slightly weakens edit strength.
