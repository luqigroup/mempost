"""Shared style and helpers for the talk figures: a semantic color palette, a
projection-grade rcParams block, reflectivity / posterior-std panel renderers, a gold-star
marker, a cached-DPS loader, and a dual PDF+PNG saver.

Palette convention: warm = memorized / collapsed (N=16), cool = generalizing (N=2048),
gold = the truth / special point.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPS_DIR = os.path.join(REPO, "plots/seismic_dps")

PALETTE = {
    "memorized":    "#A81414",   # N=16 / overfit
    "generalizing": "#008EC4",   # N=2048 / generalizing
    "truth":        "#FFCA06",   # the truth / gold star
    "baseline":     "#969696",   # reference lines
    "text":         "#434343",   # axis + body text
    "faint":        "#C0C0C0",   # gridlines
}

FS_ANNOT = 15

# frozen display constants so every panel is comparable
REFL_VMAX = 800.0            # signed reflectivity, symmetric gray window
STD_VMAX = 175.0            # shared magma range for all posterior-std panels
DX, DZ = 0.020, 0.0125     # km/cell -> physical extent, aspect="equal"


def dps_npz(N, idx):
    return os.path.join(DPS_DIR, f"dps_N{N}_trainidx{idx}.npz")


def load_dps(N, idx):
    """Load a cached DPS run, with an actionable error if it is missing."""
    p = dps_npz(N, idx)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no cached DPS run at {p}\n  build it with: python scripts/seismic_dps_posterior.py "
            f"--phase run --N {N} --truth-mode train-img --truth-idx {idx} --skip-mem")
    return np.load(p)


def refl_panel(ax, img, vmax=REFL_VMAX):
    """Reflectivity panel: de-mean -> water-mute -> fixed gray +/-vmax -> physical extent,
    bicubic, no ticks, full box. Returns the image handle."""
    from mempost.utils.seismic import water_mute
    img = np.asarray(img, np.float32) - float(np.asarray(img).mean())
    img = water_mute(img); n = img.shape[-1]
    im = ax.imshow(img.T, cmap="gray", vmin=-vmax, vmax=vmax, aspect="equal",
                   extent=[0, n * DX, n * DZ, 0], interpolation="bicubic")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(PALETTE["text"]); s.set_linewidth(0.8)
    return im


def std_panel(ax, std, vmax=STD_VMAX):
    """Pointwise posterior-std panel: water-mute -> magma [0, vmax] (shared) -> physical
    extent, no ticks, full box. Returns the handle (for a shared colorbar)."""
    from mempost.utils.seismic import water_mute
    ps = water_mute(np.asarray(std, np.float32)); n = ps.shape[-1]
    im = ax.imshow(ps.T, cmap="magma", vmin=0, vmax=vmax, aspect="equal",
                   extent=[0, n * DX, n * DZ, 0])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(PALETTE["text"]); s.set_linewidth(0.8)
    return im


def apply_talk_style():
    """Projection-grade rcParams: large serif fonts, despined, vector-safe."""
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 20,
        "axes.titlesize": 22, "axes.labelsize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 15, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": PALETTE["text"], "axes.labelcolor": PALETTE["text"],
        "text.color": PALETTE["text"], "xtick.color": PALETTE["text"], "ytick.color": PALETTE["text"],
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "lines.linewidth": 2.4, "lines.markersize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
    })


def gold_star(ax, x, y, s=360, zorder=9):
    """Gold star with a black edge -- the truth / special-point marker."""
    return ax.scatter([x], [y], marker="*", s=s, color=PALETTE["truth"],
                      edgecolors="black", linewidths=1.2, zorder=zorder)


def save_fig(fig, stem, tight=True):
    """Save a figure as both vector PDF and dpi-300 PNG, then close.
    tight=False keeps the fixed figure bbox so same-figsize panels stay byte-for-byte the
    same size -- flipbook images then never shift on flip."""
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    bbox = "tight" if tight else None
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches=bbox)
    plt.close(fig)
    print(f"Saved {stem}.pdf, {stem}.png")
