"""Memorization illustration: a kernel-density prior fit to N samples of a two-mode
Gaussian mixture, swept over N, plus the empirical delta-function prior.

As N shrinks the learned density breaks from the smooth truth into spikes at the
training points -- the stylized picture of a memorizing prior. Bandwidths are chosen
to exaggerate the effect for the talk.

Writes figs/kde_N{200,50,10,5}.{png,pdf} and figs/delta_vs_truth.{png,pdf}. CPU, seconds.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figs")

TRUTH = "#434343"   # true density + rug
LEARNED = "#1B9E77"  # learned (KDE / delta) density

# two-mode Gaussian mixture, peak height fixed regardless of sigma so panels compare
MU, SIG, W = (-3.0, 3.0), (0.85, 0.85), (0.5, 0.5)
_SCALE = 0.45 / sum(w * norm.pdf(MU[0], m, s) for w, m, s in zip(W, MU, SIG))


def true_pdf(x):
    return _SCALE * sum(w * norm.pdf(x, m, s) for w, m, s in zip(W, MU, SIG))


def sample_gmm(n, rng):
    k = rng.binomial(n, W[0])
    s = np.concatenate([rng.normal(MU[0], SIG[0], k), rng.normal(MU[1], SIG[1], n - k)])
    rng.shuffle(s)
    return s


def _style(ax):
    ax.set_xlim(-7, 7); ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#aaaaaa")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(42)
    x = np.linspace(-7, 7, 800)
    samples = sample_gmm(200, rng)
    Ns = [200, 50, 10, 5]
    bw = {200: 0.3, 50: 0.1, 10: 0.04, 5: 0.02}   # sharper for small N -> spikier

    # one shared y-limit across the flipbook so only the curves move between frames
    ymax = true_pdf(x).max()
    for n in (200, 50):
        ymax = max(ymax, (gaussian_kde(samples[:n], bw_method=bw[n])(x) * _SCALE).max())
    ymax *= 1.15

    for n in Ns:
        kde = gaussian_kde(samples[:n], bw_method=bw[n])(x) * _SCALE
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.fill_between(x, true_pdf(x), alpha=0.12, color=TRUTH)
        ax.plot(x, true_pdf(x), color=TRUTH, lw=2.0, label=r"true $p(\mathbf{x})$")
        ax.fill_between(x, kde, alpha=0.15, color=LEARNED)
        ax.plot(x, kde, color=LEARNED, lw=2.5, label=r"learned $p_\theta(\mathbf{x})$")
        ax.scatter(samples[:n], -0.012 * np.ones(n), color=TRUTH, s=35, zorder=5,
                   clip_on=False, marker="|", linewidths=1.2)
        ax.set_title(f"$N = {n}$", fontsize=22, color=TRUTH, pad=10)
        ax.set_ylim(-0.025, ymax)
        _style(ax)
        ax.legend(loc="upper right", fontsize=14, frameon=False)
        fig.tight_layout()
        # fixed canvas (no tight bbox) so the flipbook does not shift between frames
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"kde_N{n}.{ext}"), dpi=250, transparent=True)
        plt.close(fig)
        print(f"Saved kde_N{n}")

    # empirical delta-function prior: spikes at the five training points
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.fill_between(x, true_pdf(x), alpha=0.12, color=TRUTH)
    ax.plot(x, true_pdf(x), color=TRUTH, lw=2.0, label=r"true $p(\mathbf{x})$")
    h = true_pdf(x).max() * 1.05
    for i, xi in enumerate(samples[:5]):
        ax.plot([xi, xi], [0, h], color=LEARNED, lw=3.5,
                label=r"empirical $\hat{p}_N(\mathbf{x})$" if i == 0 else None)
        ax.plot(xi, h, "^", color=LEARNED, markersize=12)
    ax.set_ylim(-0.015, h * 1.35)
    _style(ax)
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"delta_vs_truth.{ext}"), bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("Saved delta_vs_truth")


if __name__ == "__main__":
    main()
