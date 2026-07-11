"""Save, load, and overlay per-run settings snapshots (config.json)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

RUN_CONFIG_NAME = "config.json"
_SETTINGS_MODULE = "settings"
_TUPLE_KEYS = frozenset({"M_TARGET_COLS", "GRID_T_START", "GRID_T_END"})
_PATH_KEYS = frozenset(
    {
        "METRICS_CSV",
        "STRINGS_CSV",
        "OUTPUTS_DIR",
        "SD_TURBO_ROOT",
        "_GENERATED_DIR",
        "_DATASET_DIR",
    },
)
_CALLABLE_SOURCES = {
    "compute_weighted_combined_score": ("scores", "compute_weighted_combined_score"),
}
_SETTINGS_DEPENDENTS = ("data_io", "model_m", "model_t", "_helpers", "active_label")


def _settings_module(module: ModuleType | None = None) -> ModuleType:
    return module or importlib.import_module(_SETTINGS_MODULE)


def _setting_repr(value: object) -> str:
    if callable(value):
        return getattr(value, "__name__", repr(value))
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _encode_setting(key: str, value: object) -> object:
    if callable(value):
        if key == "T_TARGET_FUNC":
            return {"$fn": "compute_weighted_combined_score"}
        name = getattr(value, "__name__", None)
        if name and name != "<lambda>":
            return {"$fn": name}
        raise ValueError(f"Cannot serialize callable setting {key}")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _encode_setting(k, v) for k, v in value.items()}
    return value


def _decode_setting(key: str, value: object) -> object:
    if isinstance(value, dict) and "$fn" in value:
        fn_name = value["$fn"]
        mod_name, attr = _CALLABLE_SOURCES.get(fn_name, (None, fn_name))
        if mod_name is None:
            raise ValueError(f"Unknown callable in run config: {fn_name}")
        return getattr(importlib.import_module(mod_name), attr)
    if key in _PATH_KEYS and isinstance(value, str):
        return Path(value)
    if key in _TUPLE_KEYS and isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return {k: _decode_setting(k, v) for k, v in value.items()}
    return value


def collect_run_config(module: ModuleType | None = None) -> dict[str, object]:
    module = _settings_module(module)
    return {
        key: _encode_setting(key, getattr(module, key))
        for key in sorted(vars(module))
        if key.isupper() and not key.startswith("_")
    }


def save_run_config(run_dir: Path, module: ModuleType | None = None) -> Path:
    run_dir = Path(run_dir)
    out_path = run_dir / RUN_CONFIG_NAME
    out_path.write_text(
        json.dumps(collect_run_config(module), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def load_run_config(run_dir: Path) -> dict[str, object]:
    path = Path(run_dir) / RUN_CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run train_m.py to snapshot config for this run."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: _decode_setting(key, value) for key, value in raw.items()}


def apply_run_config(run_dir: Path, module: ModuleType | None = None) -> Path:
    module = _settings_module(module)
    config_path = Path(run_dir) / RUN_CONFIG_NAME
    for key, value in load_run_config(run_dir).items():
        setattr(module, key, value)
    for name in _SETTINGS_DEPENDENTS:
        sys.modules.pop(name, None)
    return config_path


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


def _encoded_equal(key: str, left: object, right: object) -> bool:
    return json.dumps(_encode_setting(key, left), sort_keys=True) == json.dumps(
        _encode_setting(key, right), sort_keys=True
    )


def settings_value_diff(
    live_module: ModuleType | None = None,
    run_dir: Path | None = None,
    run_config: dict[str, object] | None = None,
) -> list[str]:
    live_module = _settings_module(live_module)
    if run_config is None:
        if run_dir is None:
            raise ValueError("settings_value_diff requires run_dir or run_config")
        run_config = load_run_config(run_dir)

    keys = sorted(
        k
        for k in set(vars(live_module)) | set(run_config)
        if k.isupper() and not k.startswith("_")
    )
    diffs: list[str] = []
    for key in keys:
        live_val = getattr(live_module, key, None)
        run_val = run_config.get(key)
        if key not in vars(live_module):
            diffs.append(f"+ {key} = {_setting_repr(run_val)}")
        elif key not in run_config:
            diffs.append(f"- {key} = {_setting_repr(live_val)}")
        elif not _encoded_equal(key, live_val, run_val):
            diffs.append(
                f"~ {key}: live={_setting_repr(live_val)}  run={_setting_repr(run_val)}"
            )
    return diffs
