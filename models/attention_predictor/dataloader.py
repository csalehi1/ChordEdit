# dataloader.py

"""
Batching over SplitDataset grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dataset import SplitDataset
from settings import *


"""
SampleBatch and SplitDataloader classes.
"""


@dataclass(frozen=True)
class SampleBatch:
    """One batch of samples, N samples of n_cells cells each."""

    image_tokens: torch.Tensor          # (N, C, S, S), (N, 1, D_clip), (N, N_v+1, D)
    source_tokens: torch.Tensor         # (N, T, D), (N, 1, D)
    target_tokens: torch.Tensor         # (N, T, D), (N, 1, D)
    source_mask: torch.Tensor           # (N, N_t) bool; ones when pooled
    target_mask: torch.Tensor           # (N, N_t) bool; ones when pooled

    y: torch.Tensor                     # (N, n_cells, C)
    default_cell: torch.Tensor          # (N,)


class SplitDatasetLoader:
    """Re-iterable loader yielding SampleBatch objects of whole samples."""

    def __init__(self, dataset: SplitDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.samples_per_batch = max(1, int(SAMPLES_PER_BATCH))

    def _batch(self, sel: torch.Tensor) -> SampleBatch:
        """Gather one SampleBatch for the sample indices in sel."""
        image_tokens, source_tokens, target_tokens, source_mask, target_mask, y = self.dataset.gather(sel)
        return SampleBatch(
            image_tokens=image_tokens,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            source_mask=source_mask,
            target_mask=target_mask,
            y=y,
            default_cell=torch.full((len(sel),), self.dataset.default_cell, dtype=torch.long, device=y.device),
        )

    def __iter__(self):
        """Yield SampleBatch objects covering the split once."""
        n = self.dataset.n_samples
        device = self.dataset.y.device
        order = torch.randperm(n, device=device) if self.shuffle else torch.arange(n, device=device)
        for k in range(0, n, self.samples_per_batch):
            yield self._batch(order[k : k + self.samples_per_batch])

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = self.dataset.n_samples
        return (n + self.samples_per_batch - 1) // self.samples_per_batch


"""
Public entry point.
"""


def get_dataloader(
    dataset: SplitDataset,
    shuffle: bool = True,
) -> SplitDatasetLoader:
    """Loader over samples. Batch size is SAMPLES_PER_BATCH from settings."""
    return SplitDatasetLoader(dataset, shuffle=shuffle)
