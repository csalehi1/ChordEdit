"""
Score a t_start by t_end grid ablation with the *official* PIE-Bench CLIP
metric and plot the t_delta=0 vs t_delta=0.15 comparison.

Unlike plot_pie_t_delta_comparison.py (which used openai/clip-vit-base-patch32 on
the whole image), this script reproduces the DirectInversion / PnP-Inversion
PIE-Bench protocol:

    * CLIP backbone:  openai/clip-vit-large-patch14 via torchmetrics CLIPScore
                      (returns 100 * cosine similarity).
    * CLIP Whole:     the whole generated image is scored against the target
                      editing prompt (brackets stripped). Selected by default.
    * CLIP Edited:    the generated image is first masked to the per-sample
                      editing region (the RLE `mask` field in mapping_file.json)
                      before scoring against the same prompt.

Pick the variant with --clip-type {whole,edited,none}. 'none' skips CLIP and
plots PSNR only. Only the selected variant is computed on a given run, but the
two CLIP variants cache under separate fields (clip_whole / clip_edited) so
switching back is free.

Each grid cell shows four numbers in a 2x2 layout (the CLIP row is the selected
--clip-type variant):
    PSNR(d=0)   PSNR(d=0.15)
    CLIP(d=0)   CLIP(d=0.15)

Scores are cached to ./cache/metrics_cache_pie_<clip-slug>.json so reruns only
process new cells. The cache file is namespaced by CLIP model so switching
backbones never mixes incomparable scores. The comparison plot is written to
./results/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# model root + dataset defaults
DEFAULT_PIE_ROOT = Path(__file__).resolve().parent.parent / "pie_bench"
DEFAULT_MAPPING_FILE = "mapping_file.json"
DEFAULT_OUTPUT_ROOT = "ablation_outputs/pie_grid_t_start_t_end"
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"

# The two t_delta conditions the grid compares, in plot order (left, right).
T_DELTAS: Tuple[float, float] = (0.0, 0.15)

PSNR_SIZE = 256   # generated/source images are resized to this before PSNR
CLIP_SIZE = 512   # mask is applied / image fed to CLIP at this resolution

# Block (t_end row, t_start col) outlined in the grid as the reference operating
# point. None disables the outline.
HIGHLIGHT_CELL_RC: Optional[Tuple[int, int]] = (1, 3)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments controlling the dataset, CLIP variant, and
    output locations. See the module docstring for the overall behaviour."""
    parser = argparse.ArgumentParser(
        description="PIE-Bench PSNR/CLIP comparison grid over t_start x t_end (delta=0 vs 0.15)."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory containing per-sample subdirectories with meta.json.",
    )
    parser.add_argument(
        "--pie-root",
        type=str,
        default=str(DEFAULT_PIE_ROOT),
        help="Root of PIE-Bench dataset (provides per-sample editing masks/prompts).",
    )
    parser.add_argument("--mapping-file", type=str, default=DEFAULT_MAPPING_FILE)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for the metrics cache. Default: ./cache",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for the generated plot. Default: ./results",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output PNG path. Default: <results-dir>/metric_grid_pie_<clip-type>_<n>samples.png",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=DEFAULT_CLIP_MODEL,
        help="CLIP backbone for torchmetrics CLIPScore (PIE-Bench uses ViT-L/14).",
    )
    parser.add_argument(
        "--clip-type",
        choices=["whole", "edited", "none"],
        default="whole",
        help="Which CLIP score to compute and plot: 'whole' (full image), "
             "'edited' (masked to the edit region), or 'none' (PSNR only). "
             "'whole' and 'edited' cache separately.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Limit to the first N samples (for faster testing).",
    )
    return parser.parse_args()


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Return the PSNR (dB) between two equal-shape uint8 RGB arrays.

    Computes mean squared error over all pixels/channels and converts to
    decibels against a 255 peak; byte-identical inputs are clamped to 100 dB to
    avoid a division by zero.
    """
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def load_rgb_array(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load an image as an HxWx3 uint8 RGB array, optionally LANCZOS-resized to
    ``size`` (a (width, height) tuple)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if size is not None:
            img = img.resize(size, Image.Resampling.LANCZOS)
        return np.array(img)


def strip_brackets(prompt: str) -> str:
    """Remove the PIE-Bench [edit] markers from a prompt, keeping the words
    themselves (e.g. 'a [rusty] bike' -> 'a rusty bike')."""
    return prompt.replace("[", "").replace("]", "")


def mask_decode(encoded_mask: List[int], image_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Decode the PIE-Bench run-length mask into a binary {0,1} HxW array.

    The encoding is a flat list of (start, length) pairs over the row-major
    flattened image. Mirrors the DirectInversion / PnP-Inversion reference,
    including the one-pixel border guard that forces the image edges on to
    absorb boundary annotation errors.
    """
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        mask_array[encoded_mask[i] : encoded_mask[i] + splice_len] = 1.0
    mask_array = mask_array.reshape(image_shape)
    mask_array[0, :] = 1.0
    mask_array[-1, :] = 1.0
    mask_array[:, 0] = 1.0
    mask_array[:, -1] = 1.0
    return mask_array


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize a binary mask to ``size`` x ``size``, returning a
    float {0,1} array. A mask already at the target size is returned unchanged."""
    if mask.shape == (size, size):
        return mask
    pil = Image.fromarray((mask * 255).astype(np.uint8))
    pil = pil.resize((size, size), Image.Resampling.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)


# ── torchmetrics / transformers compatibility ─────────────────────────────────

def _patch_clip_features(model) -> None:
    """Make a CLIP model's feature methods return bare tensors.

    torchmetrics CLIPScore (<=1.8.x) expects ``get_image_features`` /
    ``get_text_features`` to return the projected embedding tensor, but
    transformers >=5 returns a ``BaseModelOutputWithPooling`` whose
    ``pooler_output`` holds those embeddings. This wraps both methods to coerce
    their output back to a tensor so the torchmetrics scoring path (100 * cosine)
    works unchanged.
    """
    def coerce(out):
        if isinstance(out, torch.Tensor):
            return out
        pooled = getattr(out, "pooler_output", None)
        if pooled is not None:
            return pooled
        if isinstance(out, (tuple, list)):
            return out[1] if len(out) > 1 else out[0]
        return out

    orig_image = model.get_image_features
    orig_text = model.get_text_features
    model.get_image_features = lambda *a, **k: coerce(orig_image(*a, **k))
    model.get_text_features = lambda *a, **k: coerce(orig_text(*a, **k))


def load_clip_model(clip_model: str, device: str):
    """Instantiate a torchmetrics CLIPScore metric for ``clip_model`` on
    ``device``.

    torchmetrics and torch are imported lazily here so that a PSNR-only run
    (--clip-type none) or a fully-cached run never pays the import/load cost.
    The underlying HF model is patched for transformers>=5 compatibility (see
    :func:`_patch_clip_features`).
    """
    from torchmetrics.multimodal.clip_score import CLIPScore
    metric = CLIPScore(model_name_or_path=clip_model).to(device)
    _patch_clip_features(metric.model)
    return metric


def compute_clip(clip_metric, image_path: Path, text: str, mask: Optional[np.ndarray], device: str) -> float:
    """Score one generated image against ``text`` with torchmetrics CLIPScore.

    The image is loaded at ``CLIP_SIZE``; when a ``mask`` is given (CLIP-Edited)
    everything outside the edit region is zeroed out first. Returns
    ``100 * cosine`` between the CLIP image and text embeddings. The metric's
    running state is reset after each call so the value is the score for this
    single image, not an accumulated average.
    """
    arr = load_rgb_array(image_path, size=(CLIP_SIZE, CLIP_SIZE))
    if mask is not None:
        arr = (arr.astype(np.float32) * mask[:, :, None]).astype(np.uint8)
    img_tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).to(device)
    with torch.no_grad():
        score = clip_metric(img_tensor, text)
    clip_metric.reset()
    return float(score.detach().cpu())


def make_lazy_clip_loader(clip_model: str, device: str, clip_type: str) -> Callable[[], object]:
    """Return a memoized zero-argument loader for the CLIPScore metric.

    The model is loaded on the first call and reused thereafter, so a run whose
    cells are all cached (or a --clip-type none run, which never calls it) never
    pays the model-load cost.
    """
    state: Dict[str, object] = {"metric": None}

    def get_clip():
        if state["metric"] is None:
            print(f"Loading {clip_model} on {device} for CLIP-{clip_type.capitalize()} scoring...")
            state["metric"] = load_clip_model(clip_model, device)
        return state["metric"]

    return get_clip


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cell_cache_key(sample_id: str, t_start: float, t_end: float, t_delta: float) -> str:
    """Build a stable, filesystem/JSON-safe cache key for one grid cell. Floats
    are formatted compactly with the decimal point replaced by 'p'."""
    def f(v: float) -> str:
        return f"{v:g}".replace(".", "p")
    return f"{sample_id}|ts{f(t_start)}|te{f(t_end)}|td{f(t_delta)}"


def _clip_slug(model_name: str) -> str:
    """Turn a CLIP model id into a filesystem-safe slug used to namespace the
    cache file (e.g. 'openai/clip-vit-large-patch14' -> 'openai_clip_vit_large_patch14')."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", model_name).strip("_")


def load_cache(path: Path) -> Dict[str, Dict[str, float]]:
    """Load the metrics cache (cell-key -> {metric: value}) from ``path``,
    returning an empty dict when the file does not exist yet."""
    if path.exists():
        with path.open() as fh:
            return json.load(fh)
    return {}


def save_cache(path: Path, cache: Dict[str, Dict[str, float]]) -> None:
    """Write the metrics cache to ``path`` as indented JSON."""
    with path.open("w") as fh:
        json.dump(cache, fh, indent=2)


# ── Dataset traversal ─────────────────────────────────────────────────────────

def discover_sample_dirs(output_root: Path, n_samples: Optional[int]) -> List[Path]:
    """Return the sorted per-sample directories under ``output_root`` that hold a
    meta.json, truncated to the first ``n_samples`` when that is given."""
    sample_dirs = sorted(
        d for d in output_root.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )
    if n_samples:
        sample_dirs = sample_dirs[:n_samples]
    return sample_dirs


def iter_sample_cells(sample_dir: Path) -> Iterator[Tuple[float, float, float, Path]]:
    """Yield ``(t_start, t_end, t_delta, cell_path)`` for each grid cell recorded
    in ``sample_dir/meta.json``.

    Cells whose generated image file is missing on disk are skipped, so callers
    only see cells they can actually score.
    """
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return
    with meta_path.open() as fh:
        meta = json.load(fh)
    for t_delta_result in meta.get("t_delta_results", []):
        t_delta = float(t_delta_result["t_delta"])
        for cell_info in t_delta_result.get("cells", []):
            cell_path = Path(cell_info["path"])
            if not cell_path.exists():
                continue
            yield float(cell_info["t_start"]), float(cell_info["t_end"]), t_delta, cell_path


def load_pie_annotations(mapping_path: Path, want_mask: bool) -> Dict[str, Dict[str, object]]:
    """Load per-sample target prompts (and optionally masks) from the PIE mapping.

    Returns ``sample_id -> {"prompt": stripped target prompt, "mask": HxW float
    array | None}``. The RLE mask is decoded/resized only when ``want_mask`` is
    True (i.e. --clip-type edited); CLIP-Whole never needs it, so that work is
    skipped. The prompt is always kept, even for samples lacking a mask.
    """
    with mapping_path.open("r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    annotations: Dict[str, Dict[str, object]] = {}
    for sample_id, meta in mapping.items():
        prompt = meta.get("editing_prompt") or meta.get("edited_prompt") or ""
        encoded = meta.get("mask")
        mask = (
            resize_mask(mask_decode(encoded), CLIP_SIZE)
            if want_mask and encoded is not None
            else None
        )
        annotations[sample_id] = {"prompt": strip_brackets(prompt), "mask": mask}
    return annotations


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_grid(
    sample_dirs: List[Path],
    annotations: Dict[str, Dict[str, object]],
    cache: Dict[str, Dict[str, float]],
    cache_path: Path,
    *,
    clip_field: Optional[str],
    use_mask: bool,
    get_clip: Callable[[], object],
    device: str,
) -> Tuple[List[float], List[float]]:
    """Fill ``cache`` with PSNR and (optionally) CLIP scores for every grid cell.

    For each sample it walks the cells in meta.json, computes only the metrics
    that are missing from the cache, and reuses scores across byte-identical
    generated images within the sample (the whole t_start=0 row collapses across
    t_delta, since the pipeline clamps t_delta to 0 when t_delta >= t_start).
    CLIP is the dominant cost, so the model is loaded lazily via ``get_clip`` and
    only when a non-deduped cell actually needs it. ``clip_field`` is the cache
    field to fill (``clip_whole``/``clip_edited``) or None to skip CLIP entirely.

    The cache is flushed to ``cache_path`` after each sample so a partial run is
    recoverable. Returns the sorted unique t_start and t_end axis values seen.
    """
    t_start_set: set = set()
    t_end_set: set = set()
    cache_dirty = False

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        print(f"[{idx}/{len(sample_dirs)}] {sample_dir.name}", flush=True)
        source_path = sample_dir / "source.png"
        if not source_path.exists():
            continue

        annotation = annotations.get(sample_dir.name)
        edit_prompt = annotation["prompt"] if annotation else None
        edit_mask = annotation["mask"] if annotation else None

        source_arr: Optional[np.ndarray] = None      # PSNR reference, loaded once per sample
        scored_by_hash: Dict[str, Dict[str, float]] = {}  # image-content -> scores, for dedup

        for t_start, t_end, t_delta, cell_path in iter_sample_cells(sample_dir):
            t_start_set.add(t_start)
            t_end_set.add(t_end)

            ck = _cell_cache_key(sample_dir.name, t_start, t_end, t_delta)
            cached = cache.get(ck, {})
            need_psnr = "psnr" not in cached
            need_clip = (
                clip_field is not None
                and edit_prompt is not None
                and (not use_mask or edit_mask is not None)
                and clip_field not in cached
            )
            if not need_psnr and not need_clip:
                continue

            # Reuse scores from an identical generated image in this sample.
            img_hash = hashlib.md5(cell_path.read_bytes()).hexdigest()
            prior = scored_by_hash.get(img_hash)
            if prior is not None:
                if need_psnr and "psnr" in prior:
                    cached["psnr"] = prior["psnr"]
                    need_psnr = False
                if need_clip and clip_field in prior:
                    cached[clip_field] = prior[clip_field]
                    need_clip = False

            if need_psnr:
                if source_arr is None:
                    source_arr = load_rgb_array(source_path, size=(PSNR_SIZE, PSNR_SIZE))
                gen_arr = load_rgb_array(cell_path, size=(PSNR_SIZE, PSNR_SIZE))
                cached["psnr"] = compute_psnr(source_arr, gen_arr)

            if need_clip:
                mask = edit_mask if use_mask else None
                cached[clip_field] = compute_clip(get_clip(), cell_path, edit_prompt, mask, device)

            rec = scored_by_hash.setdefault(img_hash, {})
            if "psnr" in cached:
                rec["psnr"] = cached["psnr"]
            if clip_field is not None and clip_field in cached:
                rec[clip_field] = cached[clip_field]

            cache[ck] = cached
            cache_dirty = True

        # Flush after each sample so a partial run is not lost.
        if cache_dirty:
            save_cache(cache_path, cache)
            cache_dirty = False

    return sorted(t_start_set), sorted(t_end_set)


def aggregate_scores(
    sample_dirs: List[Path],
    cache: Dict[str, Dict[str, float]],
    clip_field: Optional[str],
) -> Tuple[Dict[Tuple[float, float, float], float], Dict[Tuple[float, float, float], float]]:
    """Average PSNR and the selected CLIP field across samples, per grid cell.

    Returns ``(avg_psnr, avg_clip)`` dicts keyed by ``(t_start, t_end, t_delta)``.
    Cells absent from the cache or missing a metric are simply omitted from the
    corresponding mean; when ``clip_field`` is None the CLIP dict is empty.
    """
    psnr_acc: Dict[Tuple[float, float, float], List[float]] = defaultdict(list)
    clip_acc: Dict[Tuple[float, float, float], List[float]] = defaultdict(list)

    for sample_dir in sample_dirs:
        for t_start, t_end, t_delta, _ in iter_sample_cells(sample_dir):
            ck = _cell_cache_key(sample_dir.name, t_start, t_end, t_delta)
            entry = cache.get(ck, {})
            key = (t_start, t_end, t_delta)
            if "psnr" in entry:
                psnr_acc[key].append(entry["psnr"])
            if clip_field is not None and clip_field in entry:
                clip_acc[key].append(entry[clip_field])

    def avg(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    avg_psnr = {k: avg(v) for k, v in psnr_acc.items()}
    avg_clip = {k: avg(v) for k, v in clip_acc.items()}
    return avg_psnr, avg_clip


# ── Plotting ──────────────────────────────────────────────────────────────────

def _fmt_metric(value: float) -> str:
    """Format a metric value to three decimals, or an em dash when missing (NaN)."""
    return f"{value:.3f}" if not np.isnan(value) else "—"


def plot_grid(
    avg_psnr: Dict[Tuple[float, float, float], float],
    avg_clip: Dict[Tuple[float, float, float], float],
    t_start_values: List[float],
    t_end_values: List[float],
    output_file: Path,
    *,
    clip_label: str,
    clip_model: str,
    n_samples: int,
) -> None:
    """Render the t_start x t_end comparison grid and save it to ``output_file``.

    Each block is a 2x2 of subcells: PSNR on top, CLIP on bottom, with t_delta=0
    on the left and t_delta=0.15 on the right. Within each top/bottom pair the
    larger value is shaded green and the smaller grey (white when both are
    missing). The block at :data:`HIGHLIGHT_CELL_RC` is outlined as a reference
    operating point. x is t_start, y is t_end.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n_rows = len(t_end_values)
    n_cols = len(t_start_values)

    GOOD = "#b6e8b6"
    PLAIN = "#f0f0f0"
    EMPTY = "white"
    BORDER = "gray"

    internal_gap = 0.06
    sub = (1 - internal_gap) / 2  # subcell size in block-data coords
    td0, td15 = T_DELTAS

    def pick(v0: float, v15: float) -> Tuple[str, str]:
        """Return (color_for_delta0, color_for_delta0.15): green for the larger
        value, grey for the smaller, white when a value is missing."""
        if np.isnan(v0) and np.isnan(v15):
            return EMPTY, EMPTY
        if np.isnan(v0):
            return EMPTY, GOOD
        if np.isnan(v15):
            return GOOD, EMPTY
        if v0 > v15:
            return GOOD, PLAIN
        if v15 > v0:
            return PLAIN, GOOD
        return EMPTY, EMPTY

    fig, ax = plt.subplots(figsize=(n_cols + 2, n_rows + 2))

    for row_i, t_end in enumerate(t_end_values):
        for col_j, t_start in enumerate(t_start_values):
            psnr0 = avg_psnr.get((t_start, t_end, td0), float("nan"))
            psnr15 = avg_psnr.get((t_start, t_end, td15), float("nan"))
            clip0 = avg_clip.get((t_start, t_end, td0), float("nan"))
            clip15 = avg_clip.get((t_start, t_end, td15), float("nan"))

            psnr_c0, psnr_c15 = pick(psnr0, psnr15)
            clip_c0, clip_c15 = pick(clip0, clip15)

            subcells = [  # dx, dy, color, label
                (0, sub, psnr_c0, "P$_0$:" + _fmt_metric(psnr0)),
                (sub, sub, psnr_c15, "P$_{15}$:" + _fmt_metric(psnr15)),
                (0, 0, clip_c0, "C$_0$:" + _fmt_metric(clip0)),
                (sub, 0, clip_c15, "C$_{15}$:" + _fmt_metric(clip15)),
            ]

            highlight = HIGHLIGHT_CELL_RC == (row_i, col_j)
            for dx, dy, color, label in subcells:
                x = col_j + dx
                y = row_i + dy
                ax.add_patch(Rectangle(
                    (x, y), sub, sub,
                    facecolor=color,
                    edgecolor="black" if highlight else BORDER,
                    lw=0.8 if highlight else 0.5,
                ))
                ax.text(
                    x + sub / 2, y + sub / 2, label,
                    ha="center", va="center",
                    family="monospace", fontsize=5,
                )

    margin = internal_gap
    ax.set(xlim=(-margin, n_cols), ylim=(-margin, n_rows), aspect="equal")

    ax.set_xticks([col + 0.5 for col in range(n_cols)])
    ax.set_xticklabels([f"{v:.2f}" for v in t_start_values], fontsize=8)
    ax.set_yticks([row + 0.5 for row in range(n_rows)])
    ax.set_yticklabels([f"{v:.2f}" for v in t_end_values], fontsize=8)
    ax.tick_params(axis="both", which="both", length=0, pad=4)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("t_start", fontsize=9, labelpad=6)
    ax.set_ylabel("t_end", fontsize=9, labelpad=6)
    ax.set_title(
        f"PIE-Bench mean PSNR and {clip_label} comparison over t-values, "
        f"d=0 and d=0.15\nwhere n_samples={n_samples}, model={clip_model.split('/')[-1]}",
        fontsize=8, pad=6,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

CLIP_LABELS = {
    "whole": "CLIP-Whole",
    "edited": "CLIP-Edited (masked)",
    "none": "(no CLIP)",
}


def main() -> None:
    """Orchestrate a run: discover samples, load annotations/cache, score the
    grid for the selected --clip-type, aggregate, and write the plot."""
    args = parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"metrics_cache_pie_{_clip_slug(args.clip_model)}.json"
    results_dir = Path(args.results_dir).expanduser().resolve()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Which CLIP variant this run computes and plots.
    compute_clip_enabled = args.clip_type != "none"
    use_mask = args.clip_type == "edited"
    clip_field = f"clip_{args.clip_type}" if compute_clip_enabled else None

    sample_dirs = discover_sample_dirs(output_root, args.n_samples)
    print(f"Found {len(sample_dirs)} sample directories under {output_root}.")

    # Prompts/masks are only needed when computing a CLIP score.
    annotations: Dict[str, Dict[str, object]] = {}
    if compute_clip_enabled:
        mapping_path = Path(args.pie_root).expanduser().resolve() / args.mapping_file
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"PIE mapping file not found: {mapping_path}. "
                "Pass --pie-root pointing at the PIE-Bench dataset."
            )
        annotations = load_pie_annotations(mapping_path, want_mask=use_mask)
        print(
            f"Loaded {len(annotations)} PIE annotations "
            f"({'prompts+masks' if use_mask else 'prompts only'}) from {mapping_path}"
        )

    cache = load_cache(cache_path)
    print(f"Cache loaded: {len(cache)} entries from {cache_path}")

    get_clip = make_lazy_clip_loader(args.clip_model, device, args.clip_type)

    t_start_values, t_end_values = score_grid(
        sample_dirs,
        annotations,
        cache,
        cache_path,
        clip_field=clip_field,
        use_mask=use_mask,
        get_clip=get_clip,
        device=device,
    )

    avg_psnr, avg_clip = aggregate_scores(sample_dirs, cache, clip_field)

    output_file = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else results_dir / f"metric_grid_pie_{args.clip_type}_{len(sample_dirs)}samples.png"
    )
    plot_grid(
        avg_psnr,
        avg_clip,
        t_start_values,
        t_end_values,
        output_file,
        clip_label=CLIP_LABELS[args.clip_type],
        clip_model=args.clip_model,
        n_samples=len(sample_dirs),
    )


if __name__ == "__main__":
    main()
