"""DPS loss decomposition -- answers "is the prior loss going down / is the likelihood
over-weighted?". Left: the data-misfit term reaches the noise floor for BOTH priors, so
the data fit does not distinguish them. Right: the prior term (distance of the running
estimate to the nearest training image) is the discriminator -- for N=16 it DROPS (the
posterior snaps onto a stored training image, gold star), for N=2048 it RISES (the
estimate moves away from any single training model). The likelihood is not over-weighted;
the memorized prior wins on its own terms.

The x-axis is the GUIDED-step ordinal: DPS only applies (and logs) data guidance for the
last (1 - guide_from) fraction of reverse-diffusion steps, so index 0 here is the first
guided step, not t=T.

    python scripts/plot_talk_loss_decomp.py --truth-idx 0

Data: plots/seismic_dps/dps_N{16,2048}_trainidx{idx}.npz (diag_data_fid, diag_prior_d1).
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from talk_style import (PALETTE, apply_talk_style, save_fig, gold_star,  # noqa: E402
                        load_dps, FS_ANNOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-idx", type=int, default=0)
    args = ap.parse_args()
    apply_talk_style()

    D = {N: load_dps(N, args.truth_idx) for N in (16, 2048)}
    nstep = len(D[16]["diag_data_fid"])
    assert len(D[2048]["diag_data_fid"]) == nstep, "N=16 / N=2048 diag lengths differ"
    step = np.arange(nstep)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    for N, key, label in [(16, "memorized", r"$N=16$ (memorized)"),
                          (2048, "generalizing", r"$N=2048$ (generalizing)")]:
        axL.plot(step, D[N]["diag_data_fid"], "-", lw=2.6, color=PALETTE[key], label=label)
        axR.plot(step, D[N]["diag_prior_d1"], "-", lw=2.6, color=PALETTE[key], label=label)

    # left: both data-misfit curves land in the noise-floor band -> the data cannot distinguish them
    axL.axhspan(1.0, 1.4, color=PALETTE["faint"], alpha=0.30, zorder=0)
    axL.axhline(1.0, ls=":", lw=1.4, color=PALETTE["baseline"], zorder=1)
    axL.text(nstep - 1, 1.45, "both reach the\nnoise floor", ha="right", va="bottom",
             fontsize=FS_ANNOT - 3, color=PALETTE["baseline"])
    axL.set_yscale("log")
    axL.set_title("data misfit", fontsize=FS_ANNOT + 2)
    axL.set_xlabel("guided step")
    axL.set_ylabel(r"$\|Ax-y\|^2\,/$ floor")
    axL.legend(loc="upper right", fontsize=FS_ANNOT - 3)

    # right: the prior term is the discriminator
    axR.set_title("distance to nearest training model", fontsize=FS_ANNOT + 2)
    axR.set_xlabel("guided step")
    axR.set_ylabel("prior-space distance")
    d16, d20 = D[16]["diag_prior_d1"], D[2048]["diag_prior_d1"]
    gold_star(axR, nstep - 1, float(d16[-1]))                    # the snap-on point (special)
    axR.annotate("snaps ONTO\na training image", xy=(nstep - 1, float(d16[-1])),
                 xytext=(nstep * 0.60, float(d16[-1]) + 52), fontsize=FS_ANNOT - 1,
                 color=PALETTE["memorized"], ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=PALETTE["memorized"], lw=1.2))
    axR.annotate("moves AWAY", xy=(nstep - 1, float(d20[-1])),
                 xytext=(nstep * 0.42, float(d20[-1]) + 22), fontsize=FS_ANNOT - 1,
                 color=PALETTE["generalizing"], ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=PALETTE["generalizing"], lw=1.2))

    fig.subplots_adjust(wspace=0.32)
    save_fig(fig, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               f"figs/talk/loss_decomp_idx{args.truth_idx}"))
    print(f"[nums] data_fid N16 {D[16]['diag_data_fid'][-1]:.2f} N2048 {D[2048]['diag_data_fid'][-1]:.2f}; "
          f"prior_d1 N16 {d16[0]:.0f}->{d16[-1]:.0f}  N2048 {d20[0]:.0f}->{d20[-1]:.0f}")


if __name__ == "__main__":
    main()
