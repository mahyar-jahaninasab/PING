
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

ArrayLike = Union[np.ndarray, 'torch.Tensor']

ICAPS_WIDTH_IN = 6.69
ICAPS_HEIGHT_IN = 5.20
ICAPS_DPI = 300

DEFAULT_TRAJ_CMAP = "tab10"
REGION_FILL_ALPHA = 0.28
REGION_EDGE_WIDTH = 1.2
TRAJ_LINE_ALPHA = 0.3

mpl.rcParams.update({
    "figure.dpi": ICAPS_DPI,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    if _HAS_TORCH and isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1) 
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Array must be 2-D with shape (N,2), got {arr.shape}")
    return arr.astype(np.float64)


def _ensure_results_dir(results_dir: Union[str, Path]) -> Path:
    p = Path(results_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_icaps_figure(
    trajectories: List[ArrayLike],
    inside_rectangles: List[Dict],
    outside_rectangles: List[Dict],
    mask_rectangles: Optional[List[Dict]] = None,
    highlight_points: Optional[List[Tuple[float, float]]] = None,
    labels: Optional[List[str]] = None,
    traj_colors: Optional[List[str]] = None,
    domain: Optional[Tuple[float, float, float, float]] = None,
    figure_name: str = "icaps_figure",
    results_dir: Union[str, Path] = "results",
    show_grid: bool = True,
    title: Optional[str] = None,
    legend_loc: str = "upper right",
    figsize: Tuple[float, float] = (ICAPS_WIDTH_IN, ICAPS_HEIGHT_IN),
    traj_alpha: float = TRAJ_LINE_ALPHA,
):
    """
    Create and save an ICAPS-compliant figure with:
    - inside_rectangles: highlight INSIDE (solid fill)
    - outside_rectangles: highlight OUTSIDE (global shading with holes)
    - mask_rectangles: second OUTSIDE mask (independent shading)
    """
    if mask_rectangles is None:
        mask_rectangles = []

    trajs = [_to_numpy(t) for t in trajectories]
    out_dir = _ensure_results_dir(results_dir)

    rects_all = inside_rectangles + outside_rectangles + mask_rectangles
    if domain is None and rects_all:
        xs = np.concatenate([t[:, 0] for t in trajs] +
                            [np.array([r['xmin'], r['xmin'] + r['width']]) for r in rects_all])
        ys = np.concatenate([t[:, 1] for t in trajs] +
                            [np.array([r['ymin'], r['ymin'] + r['height']]) for r in rects_all])
        margin_x = 0.1 * (xs.max() - xs.min() or 1.0)
        margin_y = 0.1 * (ys.max() - ys.min() or 1.0)
        xmin, xmax = xs.min() - margin_x, xs.max() + margin_x
        ymin, ymax = ys.min() - margin_y, ys.max() + margin_y
    else:
        xmin, xmax, ymin, ymax = domain or (0, 1, 0, 1)

    fig, ax = plt.subplots(figsize=figsize, dpi=ICAPS_DPI)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    if title:
        ax.set_title(title)
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    if outside_rectangles:
        overlay = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor="black", alpha=0.12, zorder=1
        )
        ax.add_patch(overlay)

        for r in outside_rectangles:
            rx, ry, rw, rh = r["xmin"], r["ymin"], r["width"], r["height"]
            color = r.get("color", "#FFCDD2")
            hole = patches.Rectangle((rx, ry), rw, rh, facecolor="white", edgecolor="none", zorder=2)
            ax.add_patch(hole)
            border = patches.Rectangle((rx, ry), rw, rh, facecolor="none",
                                       edgecolor=color, linewidth=REGION_EDGE_WIDTH, zorder=3)
            ax.add_patch(border)
    if mask_rectangles:
        overlay2 = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     facecolor="blue", alpha=0.10, zorder=1.5)
        ax.add_patch(overlay2)
        for r in mask_rectangles:
            rx, ry, rw, rh = r["xmin"], r["ymin"], r["width"], r["height"]
            color = r.get("color", "#BBDEFB")
            hole2 = patches.Rectangle((rx, ry), rw, rh, facecolor="white", edgecolor="none", zorder=2.5)
            ax.add_patch(hole2)
            border2 = patches.Rectangle((rx, ry), rw, rh, facecolor="none",
                                        edgecolor=color, linewidth=REGION_EDGE_WIDTH, zorder=3.5)
            ax.add_patch(border2)

    for r in inside_rectangles:
        rx, ry, rw, rh = r["xmin"], r["ymin"], r["width"], r["height"]
        color = r.get("color", "#E8F5E9")
        alpha = r.get("alpha", REGION_FILL_ALPHA)
        rect = patches.Rectangle((rx, ry), rw, rh, facecolor=color, edgecolor=color,
                                 alpha=alpha, linewidth=REGION_EDGE_WIDTH, zorder=4)
        ax.add_patch(rect)

    n_traj = len(trajs)
    if traj_colors is None:
        cmap = plt.get_cmap(DEFAULT_TRAJ_CMAP)
        traj_colors = [cmap(i % cmap.N) for i in range(n_traj)]

    for i, t in enumerate(trajs):
        ax.plot(t[:, 0], t[:, 1], color=traj_colors[i], linewidth=1.6,
                label=(labels[i] if labels and i < len(labels) else None), 
                zorder=5, alpha=traj_alpha)
        
        ax.scatter(t[0, 0], t[0, 1], s=20, marker="o", edgecolors="k",
                   facecolors=traj_colors[i], zorder=6, alpha=0.9)
        ax.scatter(t[-1, 0], t[-1, 1], s=20, marker="s", edgecolors="k",
                   facecolors=traj_colors[i], zorder=6, alpha=0.9)

    if highlight_points:
        for j, p in enumerate(highlight_points[:2]):
            ax.scatter(p[0], p[1], s=70, marker=("D" if j == 0 else "*"),
                       edgecolors="black", facecolors=("cyan" if j == 0 else "yellow"),
                       linewidths=0.8, zorder=7)

    # if labels:
    #     ax.legend(loc=legend_loc)

    ax.set_xlabel("State 0")
    ax.set_ylabel("State 1")

    pdf_path = out_dir / f"{figure_name}.pdf"
    png_path = out_dir / f"{figure_name}.png"

    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", dpi=ICAPS_DPI, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=ICAPS_DPI, bbox_inches="tight")
    plt.close(fig)

    return {"pdf": pdf_path, "png": png_path}
