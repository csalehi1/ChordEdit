"""
daniel_create_grid_image
========================

Build a labeled t_start x t_end image grid for every source image produced by
scripts/daniel_run_grid_metrics_pie.py and save one plot per source image
(per t_delta condition) into an outputs folder.

Axes:
  x-axis = t_start (increasing left -> right)
  y-axis = t_end   (increasing bottom -> top)

It reads the already-generated cell PNGs from
    ablation_outputs/grid_metrics_sdturbo_top/<sample>/t_delta_<slug>/cells/
so it does no GPU inference -- it just assembles plots.

Usage:
    python scripts/daniel_create_grid_image.py
    python scripts/daniel_create_grid_image.py \
        --input-root ablation_outputs/grid_metrics_sdturbo_top \
        --output-dir ablation_outputs/grid_metrics_sdturbo_top/plots
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))         # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

import argparse
import logging
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from run_grid_ablation import GRID_VALUES
from run_local_ablation import ensure_dir, param_slug


LOGGER = logging.getLogger("daniel_create_grid_image")

DEFAULT_INPUT_ROOT = (
    Path(__file__).resolve().parent.parent
    / "ablation_outputs"
    / "grid_metrics_sdturbo_top"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble t_start (x) x t_end (y) grid plots for each source image."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=str(DEFAULT_INPUT_ROOT),
        help="Directory containing <sample>/t_delta_<slug>/cells/*.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write the assembled grid plots.",
    )
    parser.add_argument(
        "--cell-px",
        type=float,
        default=1.1,
        help="Size in inches of each cell in the figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--append-datetime",
        action="store_true",
        help="Append a _YYYYmmdd_HHMMSS suffix to the output directory name.",
    )
    return parser.parse_args()


def load_cell(cells_dir: Path, t_start: float, t_end: float) -> Image.Image | None:
    filename = f"{param_slug('t_start', t_start)}__{param_slug('t_end', t_end)}.png"
    path = cells_dir / filename
    if not path.exists():
        return None
    with Image.open(path) as img:
        return img.convert("RGB")


def make_grid_plot(
    *,
    title: str,
    cells_dir: Path,
    values: list[float],
    destination: Path,
    cell_px: float = 1.1,
    dpi: int = 120,
) -> bool:
    n = len(values)
    # Rows go top -> bottom; we want t_end increasing bottom -> top, so the top
    # row is the largest t_end and the bottom row is the smallest.
    t_end_top_to_bottom = list(reversed(values))

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(n * cell_px, n * cell_px),
        squeeze=False,
    )

    any_image = False
    for row_idx, t_end in enumerate(t_end_top_to_bottom):
        for col_idx, t_start in enumerate(values):
            ax = axes[row_idx][col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d0d0d0")
                spine.set_linewidth(0.5)

            image = load_cell(cells_dir, t_start, t_end)
            if image is not None:
                ax.imshow(image)
                any_image = True
            else:
                ax.set_facecolor("#f2f2f2")

            # t_start labels along the bottom row.
            if row_idx == n - 1:
                ax.set_xlabel(f"{t_start:.1f}", fontsize=8)
            # t_end labels along the left column.
            if col_idx == 0:
                ax.set_ylabel(f"{t_end:.1f}", fontsize=8, rotation=0, labelpad=12, va="center")

    if not any_image:
        plt.close(fig)
        return False

    fig.suptitle(title, fontsize=12)
    # Shared axis captions.
    fig.supxlabel("t_start", fontsize=12)
    fig.supylabel("t_end", fontsize=12)

    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
    ensure_dir(destination.parent)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)
    return True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.append_datetime:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir.with_name(f"{output_dir.name}_{timestamp}")
    values = list(GRID_VALUES)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    ensure_dir(output_dir)

    sample_dirs = sorted(
        p for p in input_root.iterdir() if p.is_dir() and p.name != output_dir.name
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found under {input_root}")

    written = 0
    for sample_dir in sample_dirs:
        condition_dirs = sorted(
            p for p in sample_dir.iterdir() if p.is_dir() and p.name.startswith("t_delta_")
        )
        if not condition_dirs:
            LOGGER.warning("No t_delta_* directories in %s; skipping.", sample_dir.name)
            continue

        for condition_dir in condition_dirs:
            cells_dir = condition_dir / "cells"
            if not cells_dir.exists():
                LOGGER.warning("Missing cells dir: %s; skipping.", cells_dir)
                continue

            t_delta_slug = condition_dir.name.replace("t_delta_", "")
            title = f"{sample_dir.name} | x=t_start, y=t_end | t_delta={t_delta_slug.replace('p', '.')}"
            destination = output_dir / f"{sample_dir.name}__{condition_dir.name}.png"

            ok = make_grid_plot(
                title=title,
                cells_dir=cells_dir,
                values=values,
                destination=destination,
                cell_px=args.cell_px,
                dpi=args.dpi,
            )
            if ok:
                written += 1
                LOGGER.info("Saved %s", destination)
            else:
                LOGGER.warning("No cells found for %s; nothing written.", condition_dir)

    LOGGER.info("Done. Wrote %d grid plot(s) to %s", written, output_dir)


if __name__ == "__main__":
    main()
