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
WANDB_MODE = "online"                             # "online", "offline", or "disabled"
WANDB_GROUP = f"{CHORD_EDIT_MODEL}_{DIR_NAME}"    # optional label grouping related runs

# Pinned to the package dir so the key is found whatever the working directory.
load_dotenv(Path(_DIR) / ".env")

# Short names for the long metric columns, so the panel titles stay readable.
_ALIASES = {PSNR_COL: "psnr", CLIP_COL: "clip"}


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix one split's metrics for wandb, shortening the long column names."""
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
        print("wandb is not installed; continuing without tracking. pip install wandb")
        return None
    if WANDB_MODE == "online" and not os.environ.get("WANDB_API_KEY"):
        print(f"No WANDB_API_KEY in {Path(_DIR) / '.env'}; continuing without tracking.")
        return None

    # CONFIG is the same dict save_run_settings writes, so the tracked config
    # and the run's settings.json snapshot agree by construction.
    config = dict(CONFIG) | config_extra | {"commit": current_commit_id()}
    tags = [str(t) for t in (CHORD_EDIT_MODEL, DIR_NAME)]
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
            config=config,
        )
    except Exception as exc:
        print(f"wandb.init failed ({exc}); continuing without tracking.")
        return None
    print(f"Tracking to wandb: {run.url or WANDB_MODE}")
    return run


def log_epoch(
    run,
    epoch: int,
    train_regression: dict[str, float],
    train_selection: dict[str, float],
    val_regression: dict[str, float],
    val_selection: dict[str, float],
    lr: float,
    seconds: float,
) -> None:
    """Log one epoch's train and val metrics under split-prefixed keys."""
    if run is None:
        return
    run.log({
        **_prefixed("train", train_regression),
        **_prefixed("train", train_selection),
        **_prefixed("val", val_regression),
        **_prefixed("val", val_selection),
        "lr": lr,
        "epoch_seconds": seconds,
    }, step=epoch)


def log_summary(
    run,
    test_regression: dict[str, float],
    test_selection: dict[str, float],
    val_best_selection: dict[str, float],
    best_epoch: int,
    epochs_ran: int,
) -> None:
    """Record final quality in the run summary.

    Summary rather than log, so the runs table ranks on the best checkpoint's
    quality instead of whatever the last epoch happened to produce.
    """
    if run is None:
        return
    run.summary.update({
        **_prefixed("test", test_regression),
        **_prefixed("test", test_selection),
        **_prefixed("val_best", val_best_selection),
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
    })


def finish_run(run) -> None:
    """Close the run. Safe to call on None and after an exception."""
    if run is not None:
        run.finish()
