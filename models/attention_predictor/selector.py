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


# When run as a script, pin the run's settings before dataset/model import.
# When imported from train.py those modules are already loaded, so this must
# not run: load_run_settings would refuse, and parse_args would reject
# train.py's flags.
if __name__ == "__main__":
    _ARGS = parse_args()
    RUN_DIR = resolve_run_dir(load_live_settings().RUNS_DIR if _ARGS.run_dir is None else None, _ARGS.run_dir)
    load_run_settings(RUN_DIR)

from dataset import ID_TO_SPLIT_NAME, get_dataset
from model import AttentionModel
from settings import *


def get_neighbor_map(t_pairs: np.ndarray) -> torch.Tensor:
    """Map each cell to itself and its 8 neighbors."""
    t_start_values = np.unique(t_pairs[:, 0])
    t_end_values = np.unique(t_pairs[:, 1])
    n_start, n_end = len(t_start_values), len(t_end_values)

    # Grid position of every cell, and the inverse map back to cell ids.
    i = np.searchsorted(t_start_values, t_pairs[:, 0])
    j = np.searchsorted(t_end_values, t_pairs[:, 1])
    cell_of = np.empty((n_start, n_end), dtype=np.int64)
    cell_of[i, j] = np.arange(len(t_pairs))

    # Clipping replicates at the edges, so border cells keep 9 neighbors.
    neighbors = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii = np.clip(i + di, 0, n_start - 1)
            jj = np.clip(j + dj, 0, n_end - 1)
            neighbors.append(cell_of[ii, jj])
    return torch.as_tensor(np.stack(neighbors, axis=-1))


class SelectorModel:
    """Select (t_start, t_end) from predicted cells; no trainable weights."""

    def __init__(
        self,
        model: AttentionModel | list[AttentionModel],
        t_pairs: np.ndarray,
    ):
        # A single model is the length-1 case, so select_batch has one path.
        self.models = list(model) if isinstance(model, (list, tuple)) else [model]
        self.model = self.models[0]
        self.t_pairs = np.asarray(t_pairs, dtype=np.float64)
        self.cell_mask = torch.as_tensor(self.t_pairs[:, 0] > self.t_pairs[:, 1], dtype=torch.bool) if USE_DIAGONAL_MASK else None

        # Calculate the neighbor map here and reuse it for every batch, if requested.
        if TEMPERATURE is not None:
            self.neighbor_map = get_neighbor_map(self.t_pairs)

    @staticmethod
    def _build(ckpt: dict, device: torch.device | str) -> AttentionModel:
        """Rebuild one predictor from its checkpoint's own sizing contract."""
        cell_t_pairs = ckpt["cell_t_pairs"].cpu().numpy()
        t_start_values = np.asarray(ckpt["t_start_values"].cpu(), dtype=np.float64)
        t_end_values = np.asarray(ckpt["t_end_values"].cpu(), dtype=np.float64)

        model = AttentionModel(
            tuple(int(v) for v in ckpt["image_shape"]),
            int(tuple(ckpt["source_shape"])[-1]),
            int(cell_t_pairs.shape[0]),
            device=device,
            default_cell=get_default_cell(cell_t_pairs, t_start_values, t_end_values),
            feat_dim=int(tuple(ckpt["feature_shape"])[-1]),
            mask_dim=int(tuple(ckpt["mask_shape"])[-1]),
        )
        model.regressor.load_state_dict(ckpt["regressor_state_dict"])
        model.regressor.to(device).eval()
        return model

    @classmethod
    def load(
        cls,
        weights_path: Path | str,
        device: torch.device | str | None = None,
        gpu: int | str | None = None,
    ) -> SelectorModel:
        """Build an AttentionModel from regressor_weights.pt and wrap it."""
        weights_path = Path(weights_path)
        if device is None:
            device = resolve_device(gpu)
        
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        t_pairs = ckpt["cell_t_pairs"].cpu().numpy()
        print("Loaded model.")
        return cls(cls._build(ckpt, device), t_pairs)

    def calc_phi(self, deltas: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Score phi on normalized deltas with SCORE_PHI."""
        if CLAMP_DELTAS is not None:
            deltas = deltas.clamp(-CLAMP_DELTAS, CLAMP_DELTAS)
        if weights is None:
            return SCORE_PHI(deltas)
        return SCORE_PHI(deltas, weights=weights)

    @torch.no_grad()
    def select_deltas(self, deltas: torch.Tensor) -> torch.Tensor:
        """Return selected cell indices of shape (N,) from a delta surface."""
        default_cell = self.model.regressor.default_cell

        # Calculate phi.
        weights = None if PHI_WEIGHTS is None else deltas.new_tensor(PHI_WEIGHTS)
        phi = self.calc_phi(deltas, weights=weights)

        # Rank on a per-column reweighted phi, if requested.
        if DELTA_WEIGHTS is not None:
            rank = self.calc_phi(deltas, weights=deltas.new_tensor(DELTA_WEIGHTS))
        else:
            rank = phi

        # Restrict the argmax to cells clearing per-column floors, if requested.
        cell_mask = None
        if DELTA_FLOORS is not None:
            floors = deltas.new_tensor([float("-inf") if f is None else f for f in DELTA_FLOORS])
            eligible = (deltas >= floors).all(dim=-1)
            cell_mask = eligible | ~eligible.any(dim=-1, keepdim=True)

        # Restrict the argmax to cells with t_start > t_end, if requested.
        if self.cell_mask is not None:
            diag = self.cell_mask.to(device=rank.device)
            cell_mask = diag if cell_mask is None else cell_mask & diag
            if cell_mask.ndim == 2:
                none = ~cell_mask.any(dim=-1)
                if none.any():
                    cell_mask = cell_mask.clone()
                    cell_mask[none, default_cell] = True

        if cell_mask is not None:
            rank = rank.masked_fill(~cell_mask, -float("inf"))

        # Sort each cell by the phi mass over its neighborhood, if requested.
        if TEMPERATURE is not None:
            probs = torch.softmax(rank / TEMPERATURE, dim=-1)
            rank = probs[..., self.neighbor_map.to(probs.device)].sum(dim=-1)
            if cell_mask is not None:
                rank = rank.masked_fill(~cell_mask, -float("inf"))

        # Select the best cell for each sample.
        selected = rank.argmax(dim=-1)

        # Stay on the default cell unless the selected cell clears a phi floor, if requested.
        if PHI_FLOOR is not None:
            n = selected.shape[0]
            rows = torch.arange(n, device=selected.device)
            gain = phi[rows, selected] - phi[:, default_cell]
            selected = torch.where(gain > PHI_FLOOR, selected, selected.new_full((n,), default_cell))

        return selected

    @torch.no_grad()
    def select_batch(self, *inputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return (t_start, t_end) arrays of shape (N,) for one batch of model inputs."""

        # Predicted raw PSNR/CLIP (averaged over ensemble members), per-sample normalized, then phi.
        raw = torch.stack([model.pred_raw(*inputs) for model in self.models]).mean(dim=0)
        selected_surface = self.model.to_selector(raw)

        pairs = self.t_pairs[self.select_deltas(selected_surface).detach().cpu().numpy()]
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
    selector = SelectorModel.load(weights_path, device=device)
    sample_ids = sorted(sid for split in bundle.splits.values() for sid in split.sample_ids)
    embs = bundle.train.x
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
            embs.mask_features[sel],
            embs.mask_tokens[sel],
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

    # Parse the command line arguments (already read by the module-level guard).
    parse_args()

    # Set the random seeds.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Evaluate the timestep selector.
    eval(RUN_DIR)


if __name__ == "__main__":
    main()
