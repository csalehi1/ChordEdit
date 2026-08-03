"""Run settings and small shared helpers for classification.

Call load_run_settings(RUN_DIR) before importing _data / embeddings / model so
those modules bind constants from the per-run settings snapshot.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

SETTINGS_FILENAME = "settings.json"
SETTINGS_ENV_VAR = "CE_SETTINGS_JSON"
_PACKAGE_DIR = Path(__file__).resolve().parent
_SETTINGS_MODULE = _PACKAGE_DIR / "settings.py"


def _s():
    """Current settings module (live or per-run snapshot in sys.modules)."""
    import settings
    return settings


def current_commit_id() -> str:
    """Short id of the most recent commit, or "nogit" outside a repository."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def save_run_settings(run_dir: Path) -> Path:
    """Save the config this run used, so the run can be replayed exactly."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / SETTINGS_FILENAME
    dst.write_text(json.dumps(_s().CONFIG, indent=2, sort_keys=True) + "\n")
    return dst


def load_run_settings(run_dir: Path):
    """Bind the settings module to a run's saved config.

    Points CE_SETTINGS_JSON at the run's settings.json and imports settings
    fresh under that name, so evaluation sees exactly the training config.
    """
    run_dir = Path(run_dir)
    already = [n for n in ("_data", "embeddings", "model") if n in sys.modules]
    if already:
        raise RuntimeError(
            f"load_run_settings() must be called before importing modules that bind settings "
            f"(already imported: {', '.join(already)})."
        )

    path = run_dir / SETTINGS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run train.py to save the config for this run."
        )
    os.environ[SETTINGS_ENV_VAR] = str(path)

    spec = importlib.util.spec_from_file_location("settings", _SETTINGS_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load settings from {_SETTINGS_MODULE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["settings"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_live_settings():
    """Load package settings.py under a private module name (for OUTPUTS_DIR bootstrap)."""
    spec = importlib.util.spec_from_file_location("_live_settings", _SETTINGS_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load live settings from {_SETTINGS_MODULE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_run_dir(outputs_dir: Path, run_dir: Path | None = None) -> Path:
    if run_dir is not None:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    candidates = sorted(p for p in Path(outputs_dir).iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories in {outputs_dir}")
    return candidates[-1]


def resolve_device(gpu: int | str | None = None) -> torch.device:
    """None -> cuda:0 if available else cpu; int -> cuda:N; \"cpu\" -> cpu."""
    if gpu is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(gpu, str) and gpu.lower() == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{int(gpu)}")


def load_inputs_df() -> pd.DataFrame:
    """Load INPUTS_CSV with the canonical zero-padded sample_id for merges."""
    s = _s()
    df = pd.read_csv(s.INPUTS_CSV, dtype={s.SAMPLE_ID_COL: str})
    df[s.SAMPLE_ID_COL] = df[s.SAMPLE_ID_COL].map(lambda x: f"{int(x):08d}")
    return df
