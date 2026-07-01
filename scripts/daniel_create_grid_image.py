"""
Build labeled t_start x t_end image grids for every source image produced by
scripts/daniel_run_grid_metrics_pie.py and save three plots per source image
(per t_delta condition):

  1. grid_clean — no metric overlay
  2. grid_psnr  — whole-image PSNR overlay
  3. grid_clip  — mask-restricted CLIP-edited overlay

With --topo-lines: contour-filled metric sections (global bands) + contour lines on overlay plots.

Output layout:
    outputs/plots/grid_metrics_sdturbo_top/<sourceimage>/<t_delta>/grid_{clean,psnr,clip}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize
from PIL import Image

from run_grid_ablation import GRID_VALUES
from run_local_ablation import ensure_dir, param_slug


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent.parent / "ablation_outputs" / "grid_metrics_sdturbo_top"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots" / "grid_metrics_sdturbo_top"
DEFAULT_MAPPING_PATH = Path("/data/home/mirick/datasets/PIE-Bench_v1/mapping_file.json")

APPEND_DATETIME = True
PSNR_COL = "psnr"
CLIP_COL = "clip_edited"
ORIGIN_CELL = (0.0, 0.0)

THUMB_PX = 128
GAP = 3
FIG_DPI = 100
OVERLAY_ALPHA = 0.5
DRAW_TOPO_LINES = False
TOPO_N_LEVELS = 16
TOPO_CMAP = "viridis"
TOPO_LINE_COLOR = "white"
TOPO_LINE_WIDTH = 2.0


@dataclass(frozen=True)
class CellSlot:
    x: int
    y: int
    key: tuple[float, float]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Input directory containing grid metrics.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for grid images.")
    parser.add_argument("--mapping-path", default=str(DEFAULT_MAPPING_PATH), help="Path to mapping file.")
    parser.add_argument("--metrics-csv", default=None, help="Path to metrics CSV file.")
    parser.add_argument("--t-delta", type=float, default=0.0, help="t_delta value to filter metrics.")
    parser.add_argument("--topo-lines", action="store_true", default=DRAW_TOPO_LINES, help="Contour-filled metric bands + contour lines on overlay plots.")
    return parser.parse_args()


def _axis_centers(count: int, thumb_px: int, gap: int) -> np.ndarray:
    """Calculate the x/y positions of cell centers along the grid axis."""
    return gap + np.arange(count) * (thumb_px + gap) + thumb_px / 2


def _metric_levels(vmin: float, vmax: float) -> np.ndarray:
    """Generate a set of metric levels for the colormap."""
    if vmax <= vmin:
        return np.array([vmin, vmax])
    return np.linspace(vmin, vmax, TOPO_N_LEVELS + 1)


def _topo_cmap_norm(levels: np.ndarray) -> tuple[plt.Colormap, BoundaryNorm]:
    """Create a colormap and normalization for topographic band shading."""
    band_count = len(levels) - 1
    return plt.colormaps[TOPO_CMAP].resampled(band_count), BoundaryNorm(levels, band_count)


def _tint_rgba(value: float, *, vmin: float, vmax: float, alpha: float) -> tuple[int, int, int, int]:
    """Continuous global-scale tint for one cell (non-topo mode)."""
    denom = (vmax - vmin) if vmax > vmin else 1.0
    t = max(0.0, min(1.0, (value - vmin) / denom))
    red, green, blue, _ = plt.colormaps[TOPO_CMAP](t)
    return int(red * 255), int(green * 255), int(blue * 255), int(255 * alpha)


def _origin_cell_xy(values_start: list[float], values_end: list[float]) -> tuple[int, int] | None:
    """Top-left pixel of the (0, 0) cell in the composite."""
    if ORIGIN_CELL[0] not in values_start or ORIGIN_CELL[1] not in values_end:
        return None
    column = values_start.index(ORIGIN_CELL[0])
    row = len(values_end) - 1 - values_end.index(ORIGIN_CELL[1])
    stride = THUMB_PX + GAP
    return GAP + column * stride, GAP + row * stride


def _score_grid(
    scores: dict[tuple[float, float], float],
    values_start: list[float],
    values_end: list[float],
) -> np.ndarray:
    """Cell-center metric scores (includes origin for contour interpolation)."""
    grid = np.full((len(values_end), len(values_start)), np.nan, dtype=float)
    for row, t_end in enumerate(reversed(values_end)):
        for col, t_start in enumerate(values_start):
            key = (round(t_start, 1), round(t_end, 1))
            if key in scores:
                grid[row, col] = scores[key]
    return grid


def _cut_origin_hole(overlay: np.ndarray, origin_xy: tuple[int, int]) -> None:
    """Punch a transparent hole in the topo overlay above the origin cell."""
    origin_x, origin_y = origin_xy
    overlay[origin_y : origin_y + THUMB_PX, origin_x : origin_x + THUMB_PX, 3] = 0


def _render_topo_overlay_surface(
    width: int,
    height: int,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    padded: np.ndarray,
    levels: np.ndarray,
    origin_xy: tuple[int, int] | None,
) -> np.ndarray:
    """Render contour fill + lines to an RGBA surface, then cut out the origin cell."""
    figure = plt.figure(figsize=(width / FIG_DPI, height / FIG_DPI), dpi=FIG_DPI)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")
    figure.patch.set_alpha(0)
    axis.patch.set_alpha(0)

    # Draw the contour fill.
    cmap, norm = _topo_cmap_norm(levels)
    axis.contourf(mesh_x, mesh_y, padded, levels=levels, cmap=cmap, norm=norm, alpha=OVERLAY_ALPHA, antialiased=True, corner_mask=True,)
    # Draw the contour lines
    line_levels = levels[1:-1] if len(levels) > 2 else levels
    axis.contour(mesh_x, mesh_y, padded, levels=line_levels, colors=TOPO_LINE_COLOR, linewidths=TOPO_LINE_WIDTH, corner_mask=True,)
    figure.canvas.draw()
    pixel_width, pixel_height = figure.canvas.get_width_height()
    overlay = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(pixel_height, pixel_width, 4).copy()
    plt.close(figure)
    if overlay.shape[0] != height or overlay.shape[1] != width:
        overlay = np.array(Image.fromarray(overlay).resize((width, height), Image.Resampling.NEAREST))

    # Cut out the origin cell.
    if origin_xy is not None:
        _cut_origin_hole(overlay, origin_xy)
    return overlay


def _score_dict(metrics: pd.DataFrame, sample_id: str, t_delta: float, column: str) -> dict[tuple[float, float], float]:
    """Create a dictionary of metric scores for the given sample and t_delta."""
    rows = metrics[(metrics["sample_id"] == int(sample_id)) & (metrics["t_delta"] == float(t_delta))]
    return {
        (round(float(row["t_start"]), 1), round(float(row["t_end"]), 1)): float(row[column])
        for row in rows.to_dict("records")
    }


def _padded_contour_field(
    grid: np.ndarray,
    *,
    width: int,
    height: int,
    centers_x: np.ndarray,
    centers_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge-pad cell-center scores so contours span the full composite."""
    padded = np.full((grid.shape[0] + 2, grid.shape[1] + 2), np.nan)
    padded[1:-1, 1:-1] = grid

    # Pad the top and bottom edges.
    for col in range(1, padded.shape[1] - 1):
        top = grid[0, col - 1]
        bottom = grid[-1, col - 1]
        if np.isfinite(top):
            padded[0, col] = top
        if np.isfinite(bottom):
            padded[-1, col] = bottom
    for row in range(1, padded.shape[0] - 1):
        left = grid[row - 1, 0]
        right = grid[row - 1, -1]
        if np.isfinite(left):
            padded[row, 0] = left
        if np.isfinite(right):
            padded[row, -1] = right

    # Corners stay NaN after edge padding; fill from the nearest grid corner.
    for pad_row, pad_col, grid_row, grid_col in (
        (0, 0, 0, 0),
        (0, -1, 0, -1),
        (-1, 0, -1, 0),
        (-1, -1, -1, -1),
    ):
        value = grid[grid_row, grid_col]
        if np.isfinite(value):
            padded[pad_row, pad_col] = value

    x_coords = np.concatenate(([0.0], centers_x, [float(width)]))
    y_coords = np.concatenate(([0.0], centers_y, [float(height)]))
    return *np.meshgrid(x_coords, y_coords), padded


def _build_base_grid(
    cells_dir: Path,
    values_start: list[float],
    values_end: list[float],
    *,
    cell_extension: str = ".png",
) -> tuple[Image.Image, list[CellSlot]] | None:
    """Build a base grid of cells from the given directory."""
    stride = THUMB_PX + GAP
    width = len(values_start) * stride + GAP
    height = len(values_end) * stride + GAP
    canvas = Image.new("RGB", (width, height), "#ffffff")
    slots: list[CellSlot] = []
    found = False

    # Iterate over the values_end.
    for row, t_end in enumerate(reversed(values_end)):
        # Iterate over the values_start.
        for col, t_start in enumerate(values_start):
            x = GAP + col * stride
            y = GAP + row * stride
            key = (round(t_start, 1), round(t_end, 1))
            cell = cells_dir / f"{param_slug('t_start', t_start)}__{param_slug('t_end', t_end)}{cell_extension}"
            if not cell.exists():
                continue
            # Load the cell image and paste it onto the canvas.
            with Image.open(cell) as image:
                canvas.paste(image.convert("RGB").resize((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS), (x, y))
            found = True
            # Add the cell slot to the list of slots.
            if key != ORIGIN_CELL:
                slots.append(CellSlot(x, y, key))

    # Return the canvas and slots if any cells were found, otherwise return None.
    return (canvas, slots) if found else None


def _apply_overlay(
    base: Image.Image,
    slots: list[CellSlot],
    scores: dict[tuple[float, float], float],
    *,
    vmin: float,
    vmax: float,
) -> Image.Image:
    """Per-cell continuous viridis tint (used when topo-lines is off)."""
    tint_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for slot in slots:
        value = scores.get(slot.key)
        if value is None:
            continue
        rgba = _tint_rgba(value, vmin=vmin, vmax=vmax, alpha=OVERLAY_ALPHA)
        tint_layer.paste(Image.new("RGBA", (THUMB_PX, THUMB_PX), rgba), (slot.x, slot.y))
    return Image.alpha_composite(base.convert("RGBA"), tint_layer).convert("RGB")


def _save_grid_figure(
    composite: Image.Image,
    destination: Path,
    *,
    title: str,
    values_start: list[float],
    values_end: list[float],
    t_delta: float,
    colorbar_label: str | None = None,
    colorbar_vmin: float | None = None,
    colorbar_vmax: float | None = None,
    colorbar_levels: np.ndarray | None = None,
    scores: dict[tuple[float, float], float] | None = None,
    draw_contours: bool = False,
) -> None:
    """Save a grid figure to the given destination."""
    width, height = composite.size
    centers_x = _axis_centers(len(values_start), THUMB_PX, GAP)
    centers_y = _axis_centers(len(values_end), THUMB_PX, GAP)

    figure, axis = plt.subplots(figsize=(width / FIG_DPI, height / FIG_DPI), dpi=FIG_DPI)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.imshow(composite, interpolation="none", extent=(0, width, height, 0), zorder=1)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)

    if draw_contours and scores is not None and colorbar_levels is not None:
        grid = _score_grid(scores, values_start, values_end)
        if np.any(np.isfinite(grid)):
            mesh_x, mesh_y, padded = _padded_contour_field(
                grid, width=width, height=height, centers_x=centers_x, centers_y=centers_y
            )
            origin_xy = _origin_cell_xy(values_start, values_end)
            topo_surface = _render_topo_overlay_surface(
                width, height, mesh_x, mesh_y, padded, colorbar_levels, origin_xy
            )
            axis.imshow(topo_surface, interpolation="none", extent=(0, width, height, 0), zorder=2)

    # Set the x and y ticks.
    axis.set_xticks(centers_x, [f"{v:.1f}" for v in values_start])
    axis.set_yticks(centers_y, [f"{v:.1f}" for v in reversed(values_end)])
    axis.set_xlabel(f"t_start, $\\delta={t_delta:.2f}$", fontsize=12)
    axis.set_ylabel(f"t_end, $\\delta={t_delta:.2f}$", fontsize=12)
    axis.set_title(title, fontsize=14)

    # Add the colorbar if requested.
    if colorbar_label and colorbar_vmin is not None and colorbar_vmax is not None:
        if colorbar_levels is not None and len(colorbar_levels) > 1:
            cmap, norm = _topo_cmap_norm(colorbar_levels)
        else:
            cmap, norm = plt.colormaps[TOPO_CMAP], Normalize(vmin=colorbar_vmin, vmax=colorbar_vmax)
        colorbar = figure.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label, fontsize=14)
        colorbar.ax.yaxis.set_label_position("left")
        if colorbar_levels is not None and len(colorbar_levels) > 1:
            colorbar.set_ticks(colorbar_levels)
            colorbar.ax.tick_params(labelsize=9)

    # Save the figure.
    ensure_dir(destination.parent)
    save_kwargs = {"dpi": FIG_DPI, "bbox_inches": "tight", "facecolor": "white"}
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["pil_kwargs"] = {"quality": 75}
    figure.savefig(destination, **save_kwargs)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if APPEND_DATETIME:
        output_dir = output_dir.with_name(f"{output_dir.name}_{datetime.now():%Y%m%d_%H%M%S}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    metrics_csv = (
        Path(args.metrics_csv).expanduser().resolve()
        if args.metrics_csv
        else next(iter(sorted(input_dir.glob("id_to_metrics*.csv"))), None)
    )
    if metrics_csv is None or not metrics_csv.exists():
        raise FileNotFoundError(f"No metrics CSV under {input_dir}")

    metrics = pd.read_csv(metrics_csv)
    if "whole_psnr" in metrics.columns and PSNR_COL not in metrics.columns:
        metrics = metrics.rename(columns={"whole_psnr": PSNR_COL})
    metrics = metrics[metrics["t_delta"] == args.t_delta]
    if metrics.empty:
        raise ValueError(f"No rows in {metrics_csv} for t_delta={args.t_delta}")

    psnr_vmin, psnr_vmax = float(metrics[PSNR_COL].min()), float(metrics[PSNR_COL].max())
    clip_vmin, clip_vmax = float(metrics[CLIP_COL].min()), float(metrics[CLIP_COL].max())
    psnr_levels = _metric_levels(psnr_vmin, psnr_vmax)
    clip_levels = _metric_levels(clip_vmin, clip_vmax)
    values_start = sorted({round(float(v), 1) for v in metrics["t_start"].unique()}) or list(GRID_VALUES)
    values_end = sorted({round(float(v), 1) for v in metrics["t_end"].unique()}) or list(GRID_VALUES)
    mapping = json.loads(Path(args.mapping_path).read_text(encoding="utf-8"))
    ensure_dir(output_dir)

    # Iterate over sample directories.
    for sample_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        # Iterate over t_delta directories for this sample.
        for condition_dir in sorted(p for p in sample_dir.iterdir() if p.is_dir() and p.name.startswith("t_delta_")):
            # Skip if t_delta doesn't match the requested value.
            t_delta = float(condition_dir.name.replace("t_delta_", "").replace("p", "."))
            if t_delta != args.t_delta:
                continue
            # Skip if cells directory doesn't exist.
            cells_dir = condition_dir / "cells"
            if not cells_dir.is_dir():
                continue

            # Build the base grid.
            built = _build_base_grid(cells_dir, values_start, values_end)
            if built is None:
                continue
            base_canvas, slots = built

            # Get the sample ID and category.
            parts = sample_dir.name.split("_")
            sample_id, category = parts[-1], "_".join(parts[:-2])
            meta = mapping.get(sample_id, {})
            title = (
                f"Images for {sample_id} from {category}\n"
                f'Source Prompt: "{meta.get("original_prompt", "")}"\n'
                f'Target Prompt: "{meta.get("editing_prompt", "")}"'
            )
            plot_dir = output_dir / sample_dir.name / condition_dir.name
            figure_kwargs = dict(title=title, values_start=values_start, values_end=values_end, t_delta=t_delta)

            # Save the clean grid.
            _save_grid_figure(base_canvas, plot_dir / "grid_clean.png", **figure_kwargs)

            # Save the PSNR and CLIP grids.
            for name, scores, vmin, vmax, levels, label in (
                ("grid_psnr.png", _score_dict(metrics, sample_id, t_delta, PSNR_COL), psnr_vmin, psnr_vmax, psnr_levels, "Whole PSNR"),
                ("grid_clip.png", _score_dict(metrics, sample_id, t_delta, CLIP_COL), clip_vmin, clip_vmax, clip_levels, CLIP_COL),
            ):
                overlay = (
                    base_canvas if args.topo_lines
                    else _apply_overlay(base_canvas, slots, scores, vmin=vmin, vmax=vmax)
                )
                _save_grid_figure(
                    overlay, plot_dir / name,
                    colorbar_label=label,
                    colorbar_vmin=vmin,
                    colorbar_vmax=vmax,
                    colorbar_levels=levels if args.topo_lines else None,
                    scores=scores if args.topo_lines else None,
                    draw_contours=args.topo_lines,
                    **figure_kwargs,
                )
                print(f"Saved {plot_dir / name}")


if __name__ == "__main__":
    main()
