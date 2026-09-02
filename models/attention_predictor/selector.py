# selector.py

"""
Select (t_start, t_end) for every sample in a run and write the selections CSV.
"""

from __future__ import annotations

import argparse
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _helpers import *

BATCH_SIZE = 256


# Parse command line arguments.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--settings-path", default=None)
    return parser.parse_args()


# Bind the run's settings.json before importing modules that read settings at
# import time. Live settings are only used to locate the newest run.
_ARGS = parse_args()
RUN_DIR = resolve_run_dir(load_live_settings().RUNS_DIR if _ARGS.run_dir is None else None, _ARGS.run_dir)
load_run_settings(RUN_DIR)

from dataset import ID_TO_SPLIT_NAME, get_dataset
from model import AttentionModel, preds_to_deltas
from settings import *


class SelectorModel:
    """Select (t_start, t_end) from predicted cells; no trainable weights."""

    def __init__(
        self,
        model: AttentionModel,
        cell_t_pairs: np.ndarray,
        mean_surface: torch.Tensor,
    ):
        self.model = model
        self.cell_t_pairs = np.asarray(cell_t_pairs, dtype=np.float64)
        self.mean_surface = mean_surface

    @classmethod
    def load(
        cls,
        weights_path: Path | str,
        mean_surface: torch.Tensor,
        device: torch.device | str | None = None,
        gpu: int | str | None = None,
    ) -> SelectorModel:
        """Build AttentionModel from regressor_weights.pt and wrap it."""
        weights_path = Path(weights_path)
        if device is None:
            device = resolve_device(gpu)
        
        # Load the save checkpoint.
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)

        # Load metadata from the checkpoint.
        cell_t_pairs = ckpt["cell_t_pairs"].cpu().numpy()
        image_shape = tuple(int(v) for v in ckpt["image_shape"])
        text_dim = int(tuple(ckpt["source_shape"])[-1])
        t_start_values = np.asarray(ckpt["t_start_values"].cpu(), dtype=np.float64)
        t_end_values = np.asarray(ckpt["t_end_values"].cpu(), dtype=np.float64)
        default_cell = get_default_cell(cell_t_pairs, t_start_values, t_end_values)

        # Build the model.
        model = AttentionModel(
            image_shape, 
            text_dim, 
            int(cell_t_pairs.shape[0]),
            device=device,
            default_cell=default_cell,
        )
        model.regressor.load_state_dict(ckpt["regressor_state_dict"])
        model.regressor.set_target_standardization(ckpt["target_mean"], ckpt["target_std"])
        model.regressor.to(device).eval()

        print(f"Selector predicts in {PREDICTION_SPACE!r} space over {cell_t_pairs.shape[0]} cells.")
        return cls(model, cell_t_pairs, mean_surface)

    @torch.no_grad()
    def select_batch(
        self,
        image_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (t_start, t_end) arrays of shape (N,)."""
        
        # Predict the cells.
        preds = self.model.pred_cells(
            image_tokens, 
            source_tokens, 
            target_tokens, 
            source_mask, 
            target_mask,
        )

        # Calculate the raw phi scores.
        default_cell = self.model.regressor.default_cell
        deltas = preds_to_deltas(preds, default_cell, self.mean_surface)
        phi = calc_phi(deltas)

        # Reweight the phi scores per-column if requested.
        if SELECTOR_DELTA_WEIGHTS is not None:
            phi = calc_phi(deltas, weights=deltas.new_tensor(SELECTOR_DELTA_WEIGHTS))
        
        # Restrict the argmax to cells clearing per-column floors if requested.
        if SELECTOR_DELTA_FLOORS is not None:
            floors = deltas.new_tensor([float("-inf") if f is None else f for f in SELECTOR_DELTA_FLOORS])
            eligible = (deltas >= floors).all(dim=-1)
            keep = eligible | ~eligible.any(dim=-1, keepdim=True)
            phi = phi.masked_fill(~keep, -float("inf"))

        # Select the best cell for each sample.
        chosen = phi.argmax(dim=-1)

        # Stay on the default cell unless the chosen cell clears a phi floor if requested.
        if SELECTOR_PHI_FLOOR is not None:
            n = chosen.shape[0]
            rows = torch.arange(n, device=chosen.device)
            gain = phi[rows, chosen] - phi[:, default_cell]
            chosen = torch.where(gain > SELECTOR_PHI_FLOOR, chosen, chosen.new_full((n,), default_cell))

        pairs = self.cell_t_pairs[chosen.detach().cpu().numpy()]
        return pairs[:, 0], pairs[:, 1]


def eval(run_dir: Path) -> Path:
    """Select (t_start, t_end) for every sample in the run and write the CSV."""
    print(f"Using settings from {run_dir / 'settings.json'}")

    splits_path = run_dir / ID_TO_SPLIT_NAME
    weights_path = run_dir / "regressor_weights.pt"
    
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path=}.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path=}.")

    # Load the selector model and the dataset.
    device = resolve_device()
    print(f"Device: {device}.")
    bundle = get_dataset(device, run_dir)
    selector = SelectorModel.load(weights_path, device=device, mean_surface=bundle.train.mean_surface)
    sample_ids = sorted(sid for split in bundle.splits.values() for sid in split.sample_ids)
    embs = bundle.train.embs
    sample_idxs = embs.sample_idx(sample_ids)

    # Select (t_start, t_end) for every sample in the run.
    t_starts, t_ends = [], []
    for k in range(0, len(sample_idxs), BATCH_SIZE):
        sel = sample_idxs[k:k + BATCH_SIZE]
        t_start, t_end = selector.select_batch(
            embs.image_tokens[sel], 
            embs.source_tokens[sel], 
            embs.target_tokens[sel],
            embs.source_mask[sel],
            embs.target_mask[sel],
        )
        t_starts.append(t_start)
        t_ends.append(t_end)

    # Write the selections CSV.
    out = run_dir / f"id_to_selections_{DIR_NAME.replace('_', '').lower()}.csv"
    pd.DataFrame({
        "sample_id": sample_ids,
        "t_start": np.concatenate(t_starts),
        "t_end": np.concatenate(t_ends),
    }).to_csv(out, index=False)
    
    print(f"Saved {out}.")
    return out


def main() -> None:

    # Parse the command line arguments.
    args = parse_args()

    # Set the random seeds.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Evaluate the timestep selector.
    eval(RUN_DIR)


if __name__ == "__main__":
    main()
