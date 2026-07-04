"""Build labeled t_start by t_end grid images.

Copied/condensed from scripts/daniel_create_grid_image.py. Produces:

  * grid_clean  — labeled composite, no metric overlay
  * grid_psnr   — per-cell viridis tint by whole-image PSNR
  * grid_clip   — per-cell viridis tint by mask-restricted CLIP-edited score
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

import settings
from common import ensure_dir, param_slug

THUMB_PX = settings.THUMB_PX
GAP = settings.GAP
FIG_DPI = settings.FIG_DPI
OVERLAY_ALPHA = settings.OVERLAY_ALPHA
CMAP = settings.CMAP
ORIGIN_CELL = (0.0, 0.0)


@dataclass(frozen=True)
class CellSlot:
    x: int
    y: int
    key: Tuple[float, float]


def _axis_centers(count: int) -> np.ndarray:
    return GAP + np.arange(count) * (THUMB_PX + GAP) + THUMB_PX / 2


def _tint_rgba(value: float, *, vmin: float, vmax: float, alpha: float) -> Tuple[int, int, int, int]:
    """Continuous global-scale viridis tint for one cell."""
    denom = (vmax - vmin) if vmax > vmin else 1.0
    t = max(0.0, min(1.0, (value - vmin) / denom))
    red, green, blue, _ = plt.colormaps[CMAP](t)
    return int(red * 255), int(green * 255), int(blue * 255), int(255 * alpha)


def build_base_grid(
    cells_dir: Path,
    values_start: List[float],
    values_end: List[float],
    *,
    cell_extension: str = settings.CELL_EXTENSION,
) -> Optional[Tuple[Image.Image, List[CellSlot]]]:
    """Paste every cell image into one composite; return it plus per-cell slots."""
    stride = THUMB_PX + GAP
    width = len(values_start) * stride + GAP
    height = len(values_end) * stride + GAP
    canvas = Image.new("RGB", (width, height), "#ffffff")
    slots: List[CellSlot] = []
    found = False

    for row, t_end in enumerate(reversed(values_end)):
        for col, t_start in enumerate(values_start):
            x = GAP + col * stride
            y = GAP + row * stride
            key = (round(t_start, 1), round(t_end, 1))
            cell = cells_dir / f"{param_slug('t_start', t_start)}__{param_slug('t_end', t_end)}{cell_extension}"
            if not cell.exists():
                continue
            with Image.open(cell) as image:
                canvas.paste(image.convert("RGB").resize((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS), (x, y))
            found = True
            if key != ORIGIN_CELL:
                slots.append(CellSlot(x, y, key))

    return (canvas, slots) if found else None


def apply_overlay(
    base: Image.Image,
    slots: List[CellSlot],
    scores: Dict[Tuple[float, float], float],
    *,
    vmin: float,
    vmax: float,
) -> Image.Image:
    """Per-cell continuous viridis tint keyed by metric value."""
    tint_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for slot in slots:
        value = scores.get(slot.key)
        if value is None:
            continue
        rgba = _tint_rgba(value, vmin=vmin, vmax=vmax, alpha=OVERLAY_ALPHA)
        tint_layer.paste(Image.new("RGBA", (THUMB_PX, THUMB_PX), rgba), (slot.x, slot.y))
    return Image.alpha_composite(base.convert("RGBA"), tint_layer).convert("RGB")


def save_grid_figure(
    composite: Image.Image,
    destination: Path,
    *,
    title: str,
    values_start: List[float],
    values_end: List[float],
    t_delta: float,
    colorbar_label: Optional[str] = None,
    colorbar_vmin: Optional[float] = None,
    colorbar_vmax: Optional[float] = None,
) -> None:
    """Render the composite with axis ticks/labels and an optional colorbar."""
    width, height = composite.size
    centers_x = _axis_centers(len(values_start))
    centers_y = _axis_centers(len(values_end))

    figure, axis = plt.subplots(figsize=(width / FIG_DPI, height / FIG_DPI), dpi=FIG_DPI)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.imshow(composite, interpolation="none", extent=(0, width, height, 0), zorder=1)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)

    axis.set_xticks(centers_x, [f"{v:.1f}" for v in values_start])
    axis.set_yticks(centers_y, [f"{v:.1f}" for v in reversed(values_end)])
    axis.set_xlabel(f"t_start, $\\delta={t_delta:.2f}$", fontsize=12)
    axis.set_ylabel(f"t_end, $\\delta={t_delta:.2f}$", fontsize=12)
    axis.set_title(title, fontsize=14)

    if colorbar_label and colorbar_vmin is not None and colorbar_vmax is not None:
        cmap, norm = plt.colormaps[CMAP], Normalize(vmin=colorbar_vmin, vmax=colorbar_vmax)
        colorbar = figure.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label, fontsize=14)
        colorbar.ax.yaxis.set_label_position("left")

    ensure_dir(destination.parent)
    save_kwargs = {"dpi": FIG_DPI, "bbox_inches": "tight", "facecolor": "white"}
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["pil_kwargs"] = {"quality": 75}
    figure.savefig(destination, **save_kwargs)
    plt.close(figure)


def save_clean_grid(
    cells_dir: Path,
    destination: Path,
    values: List[float],
    t_delta: float,
    title: str,
) -> bool:
    """Write grid_clean for one sample. Returns False if no cells were found."""
    built = build_base_grid(cells_dir, values, values)
    if built is None:
        return False
    base_canvas, _ = built
    save_grid_figure(
        base_canvas, destination, title=title, values_start=values, values_end=values, t_delta=t_delta
    )
    return True


def save_metric_grids(
    cells_dir: Path,
    sample_dir: Path,
    values: List[float],
    t_delta: float,
    title: str,
    scores_by_metric: Dict[str, Tuple[str, str, Dict[Tuple[float, float], float]]],
) -> None:
    """Write grid_clean + one tinted overlay grid per metric.

    ``scores_by_metric`` maps output filename -> (colorbar_label, _, scores).
    """
    built = build_base_grid(cells_dir, values, values)
    if built is None:
        return
    base_canvas, slots = built

    save_grid_figure(
        base_canvas, sample_dir / "grid_clean.png", title=title,
        values_start=values, values_end=values, t_delta=t_delta,
    )

    for name, (label, _, scores) in scores_by_metric.items():
        if not scores:
            continue
        vmin, vmax = min(scores.values()), max(scores.values())
        overlay = apply_overlay(base_canvas, slots, scores, vmin=vmin, vmax=vmax)
        save_grid_figure(
            overlay, sample_dir / name, title=title,
            values_start=values, values_end=values, t_delta=t_delta,
            colorbar_label=label, colorbar_vmin=vmin, colorbar_vmax=vmax,
        )
