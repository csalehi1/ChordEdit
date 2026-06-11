# Transport and Proximal Refinement Ablation

Dataset: PIE-Bench (700 samples)

| Method | Prox | PSNR ↑ | CLIP Edited ↑ | LPIPS ↓ | SSIM ↑ |
|---------|---------|---------|---------|---------|---------|
| Naive δ=0 | w/o prox | 20.202212 | 21.133865 | 0.146186 | 0.748485 |
| Naive δ=0 | w/ prox | 20.131124 | 22.567368 | 0.159816 | 0.727029 |
| Ours δ=0.15 | w/o prox | 23.887469 | 20.836789 | 0.088374 | 0.808144 |
| Ours δ=0.15 | w/ prox | 22.632562 | 22.130849 | 0.122644 | 0.761944 |

## Observations

The proximal refinement step consistently increases CLIP Edited scores, indicating stronger semantic edits. However, this comes at the cost of lower PSNR and reduced preservation of the original image.

The Chord Control Field (δ=0.15) significantly improves preservation compared to the naive baseline (δ=0). The highest PSNR is achieved by the Chord transport field without proximal refinement, while the highest semantic alignment is achieved when proximal refinement is enabled.
