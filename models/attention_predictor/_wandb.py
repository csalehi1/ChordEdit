# _wandb.py

"""
Weights and Biases tracking for train.py.
"""

from __future__ import annotations

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from pathlib import Path

from dotenv import load_dotenv

from _helpers import current_commit_id
from settings import *


USE_WANDB = True
WANDB_ENTITY = "dfmirick-harvard-university"
WANDB_PROJECT = "attention-predictor"
WANDB_MODE = "offline"                              # "online", "offline", or "disabled"
WANDB_GROUP = f"{CHORD_EDIT_MODEL}_{DIR_NAME}"    # optional label grouping related runs

# Pinned to the package dir so the key is found whatever the working directory.
load_dotenv(Path(_DIR) / ".env")

# Short names for the long metric columns, so that panel titles stay readable.
_ALIASES = {PSNR_COL: "psnr", CLIP_COL: "clip"}

def _log_prep(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix one split's metrics for wandb, shortening long column names."""
    out = {}
    for key, value in metrics.items():
        for col, alias in _ALIASES.items():
            key = key.replace(col, alias)
        out[f"{prefix}/{key}"] = value
    return out


def init_run(run_dir: Path, config_extra: dict):
    """Start a wandb run, or return None when tracking is off or unavailable."""
    if not USE_WANDB:
        return None
    try:
        import wandb
    except ImportError:
        print("Missing wandb, continuing without tracking.")
        return None
    if WANDB_MODE == "online" and not os.environ.get("WANDB_API_KEY"):
        print(f"No WANDB_API_KEY in {Path(_DIR) / '.env'}, continuing without tracking.")
        return None

    # The CONFIG dict is the same as the one save_run_settings writes.
    tags = [str(t) for t in (CHORD_EDIT_MODEL, DIR_NAME, run_dir.name, *RUN_TAGS)]
    if PIE_BENCH:
        tags.append("pie-bench")
    try:
        run = wandb.init(
            entity=WANDB_ENTITY or None,
            project=WANDB_PROJECT,
            # Named from the run dir, never from WANDB_NAME/WANDB_RUN_ID: a
            # parallel launcher copying its environment into every child would
            # otherwise give all of them the same identity.
            name=run_dir.name,
            group=WANDB_GROUP or None,
            tags=tags,
            mode=WANDB_MODE,
            dir=str(run_dir),
            config=dict(CONFIG) | config_extra | {"commit": current_commit_id()},
        )
    except Exception as exc:
        print(f"wandb.init failed ({exc}); continuing without tracking.")
        return None
    print(f"Tracking to wandb: {run.url or WANDB_MODE}")
    return run


def log_epoch(
    run,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr: float,
    seconds: float,
) -> None:
    """Record one epoch's metrics under split-prefixed keys."""
    if run is None:
        return
    run.log({
        **_log_prep("train", train_metrics),
        **_log_prep("val", val_metrics),
        "lr": lr,
        "epoch_seconds": seconds,
    }, step=epoch)


def log_summary(
    run,
    test_metrics: dict[str, float],
    val_best_metrics: dict[str, float],
    best_epoch: int,
    epochs_ran: int,
) -> None:
    """Record final metrics in the run summary."""
    if run is None:
        return
    run.summary.update({
        **_log_prep("test", test_metrics),
        **_log_prep("val_best", val_best_metrics),
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
    })


def finish_run(run) -> None:
    """Finish the run."""
    if run is not None:
        run.finish()
