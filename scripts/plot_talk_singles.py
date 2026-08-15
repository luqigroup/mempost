"""Single-panel reflectivity and posterior-std figures for the talk flipbooks: each is a
bare, identically-sized panel so consecutive slides line up and flip in place.

    python scripts/plot_talk_singles.py

Reads the cached DPS runs (plots/seismic_dps/dps_N{16,2048}_trainidx{0,3159}.npz, written by
seismic_dps_posterior.py). Writes figs/talk/single/: reflectivity singles (gray, +/-800) for
truth / posterior mean / a posterior sample at idx 0 and 3159, and posterior-std singles
(magma, shared 0-175) at idx 0. CPU, seconds.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from talk_style import (apply_talk_style, refl_panel, std_panel, save_fig,  # noqa: E402
                        load_dps, FS_ANNOT, STD_VMAX)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figs/talk/single")
REFL_FIGSIZE = (6.8, 4.3)     # every reflectivity single uses this -> identical placement
STD_FIGSIZE = (7.4, 4.3)      # std singles (room for the shared colorbar)


def _refl(img, stem):
    fig, ax = plt.subplots(figsize=REFL_FIGSIZE)
    refl_panel(ax, img)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save_fig(fig, os.path.join(OUT, stem), tight=False)   # fixed bbox -> flip never shifts


def _std(std, stem):
    fig, ax = plt.subplots(figsize=STD_FIGSIZE)
    im = std_panel(ax, std)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_ticks([0, STD_VMAX]); cb.set_label("posterior std (a.u.)", fontsize=FS_ANNOT - 2)
    fig.subplots_adjust(left=0.02, right=0.9, top=0.98, bottom=0.02)
    save_fig(fig, os.path.join(OUT, stem), tight=False)


def main():
    apply_talk_style()
    os.makedirs(OUT, exist_ok=True)

    for idx in (0, 3159):
        d16, d20 = load_dps(16, idx), load_dps(2048, idx)
        _refl(d16["truth"], f"refl_idx{idx}_truth")
        _refl(d16["post_mean"], f"refl_idx{idx}_meanN16")
        _refl(d20["post_mean"], f"refl_idx{idx}_meanN2048")
        _refl(d16["samples"][0], f"refl_idx{idx}_sampN16")
        _refl(d20["samples"][0], f"refl_idx{idx}_sampN2048")

    d16, d20 = load_dps(16, 0), load_dps(2048, 0)
    _std(d16["post_std"], "std_idx0_N16")
    _std(d20["post_std"], "std_idx0_N2048")

    print("[done] single-image figures in figs/talk/single/")


if __name__ == "__main__":
    main()
