"""Build and save packed M embedding tables without training.

Edit the globals below, then run from anywhere:

    python models/modified-classification/scripts/cache_embeddings.py

Scattered = many per-sample .pt files listed in EMBEDDINGS_CSV.
Packed = one stacked table at
models/modified-classification/.cache/packed_embeddings/<CHORD_EDIT_MODEL>-<DIR_NAME>.pt.

When FREEZE_ENCODERS is True (default), packs scattered files into the packed
cache — no VAE/UNet/regressor. When False (or EMBEDDINGS_CSV is unset), encodes
with ChordEdit and writes a packed cache at the same path.
"""

from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# --- Script globals (edit these; do not rely on settings.py prompts) ---------------

DIR_NAME = "UltraEdit_Region_1000"
CHORD_EDIT_MODEL = "sd_turbo"  # "sd_turbo" | "sdxl_turbo" | "flux"
GPU = "0"  # physical GPU index, or "" for CPU (only used when encoding on the fly)

DATASET_DIR = Path(f"/shared/ssd_30T/mirick/datasets/ultra_edit/{DIR_NAME}")
EMBEDDINGS_SAMPLES_DIRNAME = "annotation_embeddings"

_SLUG = DIR_NAME.replace("_", "").lower()
GENERATED_DIR = Path(f"/shared/ssd_30T/mirick/generated/ultra_edit/{DIR_NAME}")
EMBEDDINGS_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/{CHORD_EDIT_MODEL}/{DIR_NAME}")
INPUTS_CSV = GENERATED_DIR / f"id_to_inputs_{_SLUG}.csv"
METRICS_CSV = GENERATED_DIR / f"id_to_metrics_{_SLUG}.csv"
EMBEDDINGS_CSV = EMBEDDINGS_DIR / f"id_to_embeddings_{_SLUG}.csv"

PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PACKAGE_DIR / "outputs" / DIR_NAME
PACKED_EMBEDDINGS_PATH = PACKAGE_DIR / ".cache" / "packed_embeddings" / f"{CHORD_EDIT_MODEL}-{DIR_NAME}.pt"

# True + EMBEDDINGS_CSV set -> pack scattered .pt files into .cache/.
# False or EMBEDDINGS_CSV=None -> encode with ChordEdit and pack under .cache/.
FREEZE_ENCODERS = True
EMBED_BATCH_SIZE = 16
SAMPLE_ID_COL = "sample_id"
TARGET_T_DELTA = 0.0  # None = keep every t_delta row when loading metrics

# --- Bootstrap package imports (answer settings.py dataset prompt) ----------------

os.environ["CUDA_VISIBLE_DEVICES"] = GPU

_REPO_ROOT = PACKAGE_DIR.parents[1]
for _p in (str(_REPO_ROOT), str(PACKAGE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_builtins_input = builtins.input
builtins.input = lambda _prompt="": DIR_NAME  # settings.py asks for DIR_NAME once

import settings as _settings  # noqa: E402

builtins.input = _builtins_input

# Push script globals into settings so _data / model_m see them.
if CHORD_EDIT_MODEL not in _settings.CHORD_EDIT_MODEL_CONFIGS:
    raise ValueError(
        f"Unknown CHORD_EDIT_MODEL={CHORD_EDIT_MODEL!r}; "
        f"expected one of {sorted(_settings.CHORD_EDIT_MODEL_CONFIGS)}"
    )
_cfg = _settings.CHORD_EDIT_MODEL_CONFIGS[CHORD_EDIT_MODEL]

_settings.DIR_NAME = DIR_NAME
_settings.CHORD_EDIT_MODEL = CHORD_EDIT_MODEL
_settings.CHORD_EDIT_MODEL_ROOT = _cfg["root"]
_settings.IMAGE_SIZE = int(_cfg["image_size"])
_settings.CHORD_EDIT_PIPELINE_TYPE = str(_cfg["pipeline_type"])
_settings.GENERATED_DIR = GENERATED_DIR
_settings.DATASET_DIR = DATASET_DIR
_settings.EMBEDDINGS_DIR = EMBEDDINGS_DIR
_settings.EMBEDDINGS_SAMPLES_DIRNAME = EMBEDDINGS_SAMPLES_DIRNAME
_settings.INPUTS_CSV = INPUTS_CSV
_settings.METRICS_CSV = METRICS_CSV
_settings.EMBEDDINGS_CSV = EMBEDDINGS_CSV if FREEZE_ENCODERS else None
_settings.OUTPUTS_DIR = OUTPUTS_DIR
_settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
_settings.FREEZE_ENCODERS = FREEZE_ENCODERS
_settings.EMBED_BATCH_SIZE = EMBED_BATCH_SIZE
_settings.SAMPLE_ID_COL = SAMPLE_ID_COL
_settings.TARGET_T_DELTA = TARGET_T_DELTA

import torch  # noqa: E402
from _data import get_embeddings, load_df  # noqa: E402


def main() -> None:
    pack_from_csv = FREEZE_ENCODERS and _settings.EMBEDDINGS_CSV is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  DIR_NAME={DIR_NAME}  CHORD_EDIT_MODEL={CHORD_EDIT_MODEL}")
    print(f"FREEZE_ENCODERS={FREEZE_ENCODERS}  EMBEDDINGS_CSV={_settings.EMBEDDINGS_CSV}")
    print(f"PACKED_EMBEDDINGS_PATH={PACKED_EMBEDDINGS_PATH}")

    if pack_from_csv:
        print("Packing scattered embeddings (no ChordEdit / regressor load)...")
        model = SimpleNamespace()  # unused when packing precomputed embeddings
    else:
        from model_m import SurrogateModel

        print("Encoding on the fly with ChordEdit encoders...")
        model = SurrogateModel(freeze_encoders=FREEZE_ENCODERS, device=device)

    samples = load_df().drop_duplicates(SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    print(f"Caching embeddings for {len(samples)} samples...")
    sample_ids, *_ = get_embeddings(samples, model, batch_size=EMBED_BATCH_SIZE)
    print(f"Done ({len(sample_ids)} samples).")


if __name__ == "__main__":
    main()
