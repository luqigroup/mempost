"""Setup panel: a Parihaka reflectivity model + the sparse linearized-Born acquisition
(12 sources, 24 receivers on a shallow line) + the two-prior framing. Grounds the
experiment before the results. The forward operator A is LINEAR (Born), which is what
makes the memorized-prior posterior theory EXACT here.

    python scripts/plot_talk_setup.py

Data: plots/seismic_dps/dps_N16_trainidx0.npz (truth) for the background reflectivity.
Geometry mirrors SparseBornImager: nsrc/nrec on a shallow line across the full width.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from talk_style import PALETTE, apply_talk_style, refl_panel, save_fig, load_dps, FS_ANNOT, DX  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSRC, NREC, N = 12, 24, 256
C_ACQ = "#277A2B"          # data/forward-direction green (design spec)


def main():
    apply_talk_style()
    img = load_dps(16, 0)["truth"]                       # a representative Parihaka reflectivity
    W = N * DX                                           # domain width in km (5.12)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    refl_panel(ax, img)
    # sparse acquisition on a shallow line (depths in the water column, near the surface)
    sx = np.linspace(0, W, NSRC)
    rx = np.linspace(0, W, NREC)
    ax.scatter(rx, np.full(NREC, 0.11), marker="v", s=55, color=PALETTE["baseline"],
               edgecolors="black", linewidths=0.5, zorder=6)
    ax.scatter(sx, np.full(NSRC, 0.045), marker="*", s=240, color=C_ACQ,
               edgecolors="black", linewidths=0.8, zorder=7)

    # no baked title / prior-caption: the beamer frametitle + \caption* carry those on the slide
    handles = [Line2D([0], [0], marker="*", ls="", ms=15, color=C_ACQ, markeredgecolor="black",
                      label=f"{NSRC} sources"),
               Line2D([0], [0], marker="v", ls="", ms=10, color=PALETTE["baseline"],
                      markeredgecolor="black", label=f"{NREC} receivers")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, fontsize=FS_ANNOT - 1, handletextpad=0.3, columnspacing=1.4)
    fig.subplots_adjust(top=0.90, bottom=0.03)
    save_fig(fig, os.path.join(REPO, "figs/talk/setup"))
    print("[done] setup panel")


if __name__ == "__main__":
    main()
