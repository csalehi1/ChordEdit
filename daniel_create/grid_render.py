"""Build a labeled t_start by t_end overview image (grid_clean.png)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import settings


def save_clean_grid(
    cells_dir: Path,
    destination: Path,
    grid_values: List[float],
    t_delta: float,
    title: str,
) -> bool:
    """Paste every cell into one composite with axis ticks and save grid_clean.png."""
    thumb = settings.THUMB_PX
    gap = settings.GAP
    stride = thumb + gap
    width = len(grid_values) * stride + gap
    height = len(grid_values) * stride + gap
    canvas = Image.new("RGB", (width, height), "#ffffff")
    found_any = False

    for row, t_end in enumerate(reversed(grid_values)):
        for col, t_start in enumerate(grid_values):
            x = gap + col * stride
            y = gap + row * stride
            start = f"t_start_{t_start:.1f}".replace(".", "p")
            end = f"t_end_{t_end:.1f}".replace(".", "p")
            cell_path = cells_dir / f"{start}__{end}{settings.CELL_EXTENSION}"
            if not cell_path.exists():
                continue
            with Image.open(cell_path) as image:
                canvas.paste(image.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y))
            found_any = True

    if not found_any:
        return False

    centers = gap + np.arange(len(grid_values)) * stride + thumb / 2
    figure, axis = plt.subplots(figsize=(width / settings.FIG_DPI, height / settings.FIG_DPI), dpi=settings.FIG_DPI)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.imshow(canvas, interpolation="none", extent=(0, width, height, 0), zorder=1)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_xticks(centers, [f"{v:.1f}" for v in grid_values])
    axis.set_yticks(centers, [f"{v:.1f}" for v in reversed(grid_values)])
    axis.set_xlabel(f"t_start, $\\delta={t_delta:.2f}$", fontsize=12)
    axis.set_ylabel(f"t_end, $\\delta={t_delta:.2f}$", fontsize=12)
    axis.set_title(title, fontsize=14)

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=settings.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return True
