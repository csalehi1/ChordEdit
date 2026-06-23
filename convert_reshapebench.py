"""
Converts the ReShapeBench HF dataset into a PIE-Bench-compatible folder layout
that run_pie_bench.py can consume directly via --pie-root.

Usage:
    python convert_reshapebench.py                 # full dataset
    python convert_reshapebench.py --max-samples 3  # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger("convert_reshapebench")

REPO_ID = "3087richard/ReShapeBench"

# The dataset's `mask` field stores a path like "masks/000101.png" that is
# relative to *some* config subfolder inside the HF repo (e.g. "single_object/").
# We don't know which subfolder a given row belongs to from the row itself,
# so we try the known candidates in order until one downloads successfully.
MASK_PREFIX_CANDIDATES = ["single_object", "multi_object", ""]


def fetch_mask_file(mask_rel_path: str, dest_path: Path) -> bool:
    """Try each known prefix until the mask file is found in the HF repo."""
    if dest_path.exists():
        return True

    for prefix in MASK_PREFIX_CANDIDATES:
        repo_path = f"{prefix}/{mask_rel_path}" if prefix else mask_rel_path
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type="dataset",
            )
        except Exception:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(Path(local_path).read_bytes())
        return True

    LOGGER.warning("Could not locate mask file for %s under any known prefix", mask_rel_path)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=str, default="reshapebench_export")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    out_root = Path(args.out_root).expanduser().resolve()
    img_dir = out_root / "annotation_images"
    mask_dir = out_root / "annotation_masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading %s ...", REPO_ID)
    ds = load_dataset(REPO_ID)[args.split]

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    mapping: dict[str, dict] = {}
    saved, skipped = 0, 0

    for row in ds:
        sample_id = row["id"]
        img_filename = f"{sample_id}.png"
        img_path = img_dir / img_filename

        # image is already a decoded PIL.Image -- just save it.
        if not img_path.exists():
            row["image"].convert("RGB").save(img_path)

        # mask is a relative path string pointing into the HF repo -- fetch the real file.
        mask_filename = f"{sample_id}.png"
        mask_dest = mask_dir / mask_filename
        mask_ok = fetch_mask_file(row["mask"], mask_dest)

        if not mask_ok:
            skipped += 1
            continue

        mapping[sample_id] = {
            "image_path": img_filename,
            "mask_path": mask_filename,
            "source_prompt": row["source_prompt"],
            "target_prompt": row["target_prompt"],
            "editing_instruction": row["instruction"],
            "foreground": row["foreground"],
            "foreground_target": row["foreground_target"],
        }
        saved += 1

    with open(out_root / "mapping_file.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    LOGGER.info("Done. Saved %d samples, skipped %d. Output: %s", saved, skipped, out_root)


if __name__ == "__main__":
    main()