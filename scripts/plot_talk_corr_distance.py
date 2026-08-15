"""Talk version of the payoff plot: posterior recovery corr(mean, truth) vs how far the
true model sits from the N=16 training cluster, for the memorized N=16 prior (red) and the
generalizing N=2048 prior (blue). Same data, SNR, xi throughout. As the truth leaves the
overfit region the N=16 posterior collapses while N=2048 holds.

    python scripts/plot_talk_corr_distance.py

Data: plots/seismic_dps/dps_N{16,2048}_trainidx{idx}.npz (corr).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from talk_style import PALETTE, apply_talk_style, save_fig, gold_star, load_dps, FS_ANNOT  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (image index, 2-PC distance to the nearest N=16 training image)
TRUTHS = [(0, 0), (3304, 12037), (2612, 28487), (3770, 47805), (3159, 64971)]


def _corr(N, idx):
    return float(load_dps(N, idx)["corr"])


def main():
    apply_talk_style()
    rows = sorted((dist, idx, _corr(16, idx), _corr(2048, idx)) for idx, dist in TRUTHS)
    dist = [r[0] / 1000.0 for r in rows]                      # -> 10^3 2-PC units, tidier axis

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.plot(dist, [r[2] for r in rows], "o-", color=PALETTE["memorized"], lw=2.6, ms=10,
            label=r"$N=16$ (memorized)", zorder=4)
    ax.plot(dist, [r[3] for r in rows], "s-", color=PALETTE["generalizing"], lw=2.6, ms=9,
            label=r"$N=2048$ (generalizing)", zorder=4)
    ax.axvspan(-2.5, 2.5, color=PALETTE["faint"], alpha=0.30, zorder=0)   # the N=16 sample-dense region
    ax.text(0.0, 0.05, "in-dist.", ha="center", va="bottom", fontsize=FS_ANNOT - 4,
            color=PALETTE["baseline"], rotation=90)
    ax.text(52, 0.36, "collapses", fontsize=FS_ANNOT, color=PALETTE["memorized"], ha="center", va="bottom")
    ax.text(52, 0.90, "holds", fontsize=FS_ANNOT, color=PALETTE["generalizing"], ha="center", va="bottom")
    # the in-distribution red point is "perfect" -- but that is lookup, not skill (the talk's irony)
    gold_star(ax, dist[0], rows[0][2])
    ax.annotate("perfect —\nbut lookup", xy=(dist[0], rows[0][2]), xytext=(11, 0.70),
                fontsize=FS_ANNOT - 1, color=PALETTE["memorized"], ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE["memorized"], lw=1.0))

    ax.set_xlabel(r"distance of true model from $N{=}16$ training")
    ax.set_ylabel(r"corr(mean, truth)")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(-5, 70)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))
    save_fig(fig, os.path.join(REPO, "figs/talk/corr_vs_distance"))
    for r in rows:
        print(f"  img {r[1]:>4}  dist={r[0]:>6}  N16 {r[2]:.3f}  N2048 {r[3]:.3f}")


if __name__ == "__main__":
    main()
