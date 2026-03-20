"""Generate individual paper-quality panels for Helmholtz figures from saved results.

Usage:
    conda activate mempost
    cd ~/Codes/mempost
    python scripts/generate_paper_panels.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mempost.utils.kl_prior import VelocityKLPrior
from mempost.utils.memorization_metrics import nearest_neighbor_distances, ratio_mem_values

# --------------- config ---------------
RESULTS_PATHS = {
    0.07: os.path.expanduser(
        "~/Codes/mempost/data/backup_avg3_256/checkpoints_step-0.07/results.pth"
    ),
    0.1: os.path.expanduser(
        "~/Codes/mempost/data/backup_avg3_256/checkpoints_step-0.1/results.pth"
    ),
}
RESULTS_PATH = RESULTS_PATHS[0.07]  # default for non-calibration panels
OUT_DIR = os.path.expanduser(
    "~/Documents/paper-IMAGE2026otlo/figs/helmholtz_panels"
)
os.makedirs(OUT_DIR, exist_ok=True)

N_VALUES = [50, 200, 1000]
MEM_THRESHOLD = 0.7
DPI = 300
PANEL_SIZE = (2.5, 2.5)  # inches per panel
CBAR_WIDTH = "4%"
CBAR_PAD = 0.05
CMAP_VEL = "terrain"
CMAP_ERR = "magma"
CMAP_STD = "OrRd"
SCATTER_FIGSIZE = (3.5, 3.5)

# KL prior for reconstruction
KL_KWARGS = dict(K=10, grid_size=200, v_background=2.0, sigma_m=0.02)

# colors
COL_TRAIN = "#1f77b4"
COL_POST = "#d62728"
COL_TRUE = "black"
COLORS_BAR = {50: "#d62728", 200: "#ff7f0e", 1000: "#2ca02c"}


def load_results():
    r = torch.load(RESULTS_PATH, map_location="cpu", weights_only=False)
    return r


def reconstruct_velocity(kl, z):
    """z: (n, K) -> v: (n, grid, grid) numpy."""
    z_np = np.asarray(z, dtype=np.float32)
    return kl.to_velocity(kl.reconstruct(z_np))  # (n, grid, grid) numpy


def save_panel(data, fname, cmap, vmin, vmax, title=None):
    """Save a single image panel with a tight colorbar."""
    fig, ax = plt.subplots(1, 1, figsize=PANEL_SIZE)
    im = ax.imshow(data.T, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", origin="upper", interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    # tight colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=CBAR_WIDTH, pad=CBAR_PAD)
    plt.colorbar(im, cax=cax)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_scatter(z_train, z_samples, z_true, ratios, N, fname):
    """Scatter plot of first two KL dims: training, posterior, true."""
    fig, ax = plt.subplots(1, 1, figsize=SCATTER_FIGSIZE)

    # training points
    ax.scatter(z_train[:, 0], z_train[:, 1], c=COL_TRAIN, s=25,
               alpha=0.4, zorder=1, label="Training")
    # posterior samples
    ax.scatter(z_samples[:, 0], z_samples[:, 1], facecolors="none",
               edgecolors=COL_POST, s=30, linewidth=0.8, alpha=0.6,
               zorder=2, label="Posterior")
    # true model
    ax.scatter(z_true[0], z_true[1], c=COL_TRUE, marker="*", s=200,
               zorder=3, label="True")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=12)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_bar_chart(mem_counts, fname):
    """Bar chart of memorization fraction across N values."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    xs = np.arange(len(N_VALUES))
    n_total = 256
    fracs = [mem_counts[N] / n_total for N in N_VALUES]
    bars = ax.bar(xs, fracs, width=0.6, edgecolor="k", linewidth=0.5,
                  color=[COLORS_BAR[N] for N in N_VALUES])
    # labels above bars
    for x, f, N in zip(xs, fracs, N_VALUES):
        ax.text(x, f + 0.02, f"{mem_counts[N]}/{n_total}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(N) for N in N_VALUES])
    ax.set_xlabel("$N$", fontsize=14)
    ax.set_ylabel("Frac. memorized", fontsize=14)
    ax.set_ylim(0, 1.12)
    ax.tick_params(labelsize=12)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_histogram(ratios, N, fname, xlim_global):
    """Histogram of memorization ratios."""
    fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.2))
    ax.hist(ratios, bins=30, color=COLORS_BAR[N], alpha=0.7,
            edgecolor="black", linewidth=0.5)
    ax.axvline(MEM_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"$r={MEM_THRESHOLD}$")
    ax.set_xlim(xlim_global)
    ax.set_xlabel("Memorization ratio $r$", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    n_mem = int((ratios < MEM_THRESHOLD).sum())
    ax.set_title(f"$N = {N}$: {n_mem}/{len(ratios)} memorized",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


LOSS_FIGSIZE = (5.5, 2.0)


def save_loss(results, fname, ylim):
    """Likelihood loss over DPS reverse diffusion steps, all N overlaid."""
    fig, ax = plt.subplots(1, 1, figsize=LOSS_FIGSIZE)
    for N in N_VALUES:
        lik = np.asarray(results[f"dps_hist_{N}"]["likelihood"])
        # Recorded at t % 10 == 0 during reversed(range(200)):
        # t = 190, 180, 170, ..., 10, 0  (20 entries)
        steps = np.arange(190, -1, -10)
        ax.plot(steps, lik, marker=MARKERS[N], markersize=4, linewidth=1.2,
                color=COLORS_BAR[N], label=f"$N = {N}$")
    ax.set_xlabel("Diffusion timestep $t$", fontsize=9)
    ax.set_ylabel(r"$\|\mathbf{d}_\mathrm{obs} - F(\mathbf{m})\|^2$", fontsize=9)
    ax.set_ylim(ylim)
    ax.set_xlim(200, 0)  # decreasing: T -> 0
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _get(r, key):
    """Get value from results dict as numpy array."""
    v = r[key]
    if hasattr(v, 'numpy'):
        return v.numpy()
    return np.asarray(v)


def _binned_calibration(std_flat, err_flat, n_bins=30):
    """Equal-count binning of (std, |error|) pairs. Returns bin_centers, bin_means."""
    order = np.argsort(std_flat)
    std_sorted = std_flat[order]
    err_sorted = err_flat[order]
    edges = np.array_split(np.arange(len(std_sorted)), n_bins)
    centers, means = [], []
    for idx in edges:
        if len(idx) == 0:
            continue
        centers.append(std_sorted[idx].mean())
        means.append(err_sorted[idx].mean())
    return np.array(centers), np.array(means)


CALIB_FIGSIZE = (3.5, 3.5)
MARKERS = {50: "o", 200: "s", 1000: "^"}


def save_calibration_vspace(results_dict, step, kl, v_true, xlim, ylim, fname):
    """Binned calibration in velocity space: pointwise std vs |mean - true|."""
    fig, ax = plt.subplots(1, 1, figsize=CALIB_FIGSIZE)
    for N in N_VALUES:
        r = results_dict[step]
        z_samp = _get(r, f"z_samples_{N}")
        v_fields = []
        for i in range(z_samp.shape[0]):
            v_i = kl.to_velocity(kl.reconstruct(z_samp[i]))
            v_fields.append(v_i)
        v_fields = np.stack(v_fields, axis=0)
        v_std = v_fields.std(axis=0).ravel()
        v_err = np.abs(v_fields.mean(axis=0) - v_true).ravel()
        cx, cy = _binned_calibration(v_std, v_err)
        ax.plot(cx, cy, marker=MARKERS[N], markersize=4, linewidth=1.2,
                color=COLORS_BAR[N], label=f"$N = {N}$")
    lim = max(xlim[1], ylim[1])
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1, label="$y = x$")
    ax.set_xlabel(r"Pointwise $\mathrm{std}(\mathbf{v})$ (km/s)", fontsize=9)
    ax.set_ylabel(r"$|\overline{\mathbf{v}} - \mathbf{v}^\ast|$ (km/s)", fontsize=9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_calibration_zspace(results_dict, step, xlim, ylim, fname):
    """Binned calibration in z-space: per-KL-dim std vs |mean - true|."""
    fig, ax = plt.subplots(1, 1, figsize=CALIB_FIGSIZE)
    z_true = _get(results_dict[step], "z_true")
    for N in N_VALUES:
        r = results_dict[step]
        z_samp = _get(r, f"z_samples_{N}")  # (n_samples, K)
        z_std = z_samp.std(axis=0)   # (K,)
        z_err = np.abs(z_samp.mean(axis=0) - z_true)  # (K,)
        cx, cy = _binned_calibration(z_std, z_err, n_bins=10)
        ax.plot(cx, cy, marker=MARKERS[N], markersize=7, linewidth=1.5,
                color=COLORS_BAR[N], label=f"$N = {N}$")
    lim = max(xlim[1], ylim[1])
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1, label="$y = x$")
    ax.set_xlabel(r"Pointwise std (KL-space)", fontsize=13)
    ax.set_ylabel(r"Pointwise error (KL-space)", fontsize=13)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=11, loc="upper left")
    ax.tick_params(labelsize=12)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=DPI,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    print("Loading results...")
    r = load_results()
    kl = VelocityKLPrior(**KL_KWARGS)

    z_true = _get(r, "z_true")
    if z_true.ndim == 1:
        z_true_2d = z_true[np.newaxis, :]
    else:
        z_true_2d = z_true
    v_true = reconstruct_velocity(kl, z_true_2d)[0]

    # ---- Global ranges for shared colorbars ----
    v_range = v_true.max() - v_true.min()
    vmin_vel = v_true.min() + 0.1 * v_range
    vmax_vel = v_true.max() - 0.1 * v_range

    # Compute all posterior stats first to get global ranges
    means, stds, errors = {}, {}, {}
    most_mem_samples, nn_samples = {}, {}

    for N in N_VALUES:
        print(f"Reconstructing velocities for N={N}...")
        z_samp = _get(r, f"z_samples_{N}")
        z_train = _get(r, f"z_train_{N}")
        ratios = _get(r, f"ratios_{N}")

        v_samples = reconstruct_velocity(kl, z_samp)
        means[N] = v_samples.mean(axis=0)
        stds[N] = v_samples.std(axis=0)
        errors[N] = np.abs(means[N] - v_true)

        # Most memorized sample (lowest ratio)
        idx_most_mem = int(np.argmin(ratios))
        most_mem_samples[N] = v_samples[idx_most_mem]

        # Nearest training neighbor to the most memorized sample
        nn_idx = int(_get(r, f"nn_indices_{N}")[idx_most_mem])
        nn_samples[N] = reconstruct_velocity(kl, z_train[nn_idx:nn_idx+1])[0]

    # Global ranges (shared across all N)
    global_err_max = max(errors[N].max() for N in N_VALUES)
    # Use 95th percentile for std vmax to make structure more visible
    all_std_vals = np.concatenate([stds[N].ravel() for N in N_VALUES])
    global_std_max = np.percentile(all_std_vals, 95)
    all_ratios = np.concatenate([_get(r, f"ratios_{N}") for N in N_VALUES])
    ratio_xlim = (all_ratios.min() * 0.9, all_ratios.max() * 1.05)

    # ---- Save panels ----
    print("Saving panels...")

    # 1. Ground truth
    save_panel(v_true, "true.png", CMAP_VEL, vmin_vel, vmax_vel)

    # 2. Posterior means (shared velocity colorbar)
    for N in N_VALUES:
        save_panel(means[N], f"mean_N{N}.png", CMAP_VEL, vmin_vel, vmax_vel)

    # 3. Absolute errors (shared magma colorbar)
    for N in N_VALUES:
        save_panel(errors[N], f"error_N{N}.png", CMAP_ERR, 0, global_err_max)

    # 4. Posterior stds (shared magma colorbar)
    for N in N_VALUES:
        save_panel(stds[N], f"std_N{N}.png", CMAP_STD, 0, global_std_max)

    # 5. Most memorized DPS sample (shared velocity colorbar)
    for N in N_VALUES:
        save_panel(most_mem_samples[N], f"dps_N{N}.png",
                   CMAP_VEL, vmin_vel, vmax_vel)

    # 6. Nearest training neighbor (shared velocity colorbar)
    for N in N_VALUES:
        save_panel(nn_samples[N], f"nn_N{N}.png",
                   CMAP_VEL, vmin_vel, vmax_vel)

    # 7. Scatter plots (first two KL dims)
    for N in N_VALUES:
        z_samp = _get(r, f"z_samples_{N}")
        z_train = _get(r, f"z_train_{N}")
        ratios = _get(r, f"ratios_{N}")
        save_scatter(z_train, z_samp, z_true, ratios, N,
                     f"scatter_01_N{N}.png")

    # 8. Bar chart
    mem_counts = {}
    for N in N_VALUES:
        ratios = _get(r, f"ratios_{N}")
        mem_counts[N] = int((ratios < MEM_THRESHOLD).sum())
    save_bar_chart(mem_counts, "bar_memorized.png")

    # 9. Histograms
    for N in N_VALUES:
        ratios = _get(r, f"ratios_{N}")
        save_histogram(ratios, N, f"histogram_N{N}.png", ratio_xlim)

    # 10. Loss over DPS reverse diffusion steps
    all_lik = np.concatenate([
        np.asarray(r[f"dps_hist_{N}"]["likelihood"]) for N in N_VALUES
    ])
    loss_ylim = (0, all_lik.max() * 1.05)
    save_loss(r, "loss_dps.png", loss_ylim)

    # 11. Calibration plots (v-space and z-space, step=0.07 only)
    print("Computing calibration plots...")
    results_dict = {0.07: r}  # already loaded

    # Compute limits from binned calibration data (tight to last data point)
    v_binned_max_x, v_binned_max_y = 0, 0
    z_binned_max_x, z_binned_max_y = 0, 0
    for N in N_VALUES:
        # v-space: use already-computed stds/errors, bin them
        cx, cy = _binned_calibration(stds[N].ravel(), errors[N].ravel())
        v_binned_max_x = max(v_binned_max_x, cx.max())
        v_binned_max_y = max(v_binned_max_y, cy.max())
        # z-space
        z_samp = _get(r, f"z_samples_{N}")
        z_std = z_samp.std(axis=0)
        z_err = np.abs(z_samp.mean(axis=0) - z_true)
        cx, cy = _binned_calibration(z_std, z_err, n_bins=10)
        z_binned_max_x = max(z_binned_max_x, cx.max())
        z_binned_max_y = max(z_binned_max_y, cy.max())

    PAD = 1.05  # 5% padding beyond last data point
    v_cal_max = max(v_binned_max_x, v_binned_max_y) * PAD
    z_cal_max = max(z_binned_max_x, z_binned_max_y) * PAD
    v_xlim = (0, v_cal_max)
    v_ylim = (0, v_cal_max)
    z_xlim = (0, z_cal_max)
    z_ylim = (0, z_cal_max)

    save_calibration_vspace(results_dict, 0.07, kl, v_true,
                            v_xlim, v_ylim,
                            "calibration_vspace.png")
    save_calibration_zspace(results_dict, 0.07,
                            z_xlim, z_ylim,
                            "calibration_zspace.png")

    # 12. Memorization comparison: 3 most memorized N=50 samples
    #     with distinct nearest neighbors, 2x3 grid
    print("Generating memorization comparison panels (N=50)...")
    z_samp_50 = _get(r, "z_samples_50")
    z_train_50 = _get(r, "z_train_50")
    ratios_50 = _get(r, "ratios_50")
    nn_indices_50 = _get(r, "nn_indices_50").astype(int)

    # Pick 3 most memorized with distinct nearest neighbors
    order = np.argsort(ratios_50)
    seen_nn = set()
    picks = []
    for idx in order:
        nn = nn_indices_50[idx]
        if nn not in seen_nn:
            picks.append(int(idx))
            seen_nn.add(nn)
        if len(picks) == 3:
            break

    for j, si in enumerate(picks):
        nn = nn_indices_50[si]
        v_samp = reconstruct_velocity(kl, z_samp_50[si:si+1])[0]
        v_nn = reconstruct_velocity(kl, z_train_50[nn:nn+1])[0]
        save_panel(v_samp, f"mem_sample_{j}.png",
                   CMAP_VEL, vmin_vel, vmax_vel)
        save_panel(v_nn, f"mem_nn_{j}.png",
                   CMAP_VEL, vmin_vel, vmax_vel)
        print(f"  Pair {j}: sample {si} (r={ratios_50[si]:.3f}) "
              f"-> train {nn}")

    # Print summary
    print("\n=== Memorization summary ===")
    for N in N_VALUES:
        ratios = _get(r, f"ratios_{N}")
        n_mem = int((ratios < MEM_THRESHOLD).sum())
        print(f"  N={N}: {n_mem}/256 memorized ({100*n_mem/256:.0f}%)")
    print(f"\n=== Colorbar ranges ===")
    print(f"  Velocity: [{vmin_vel:.4f}, {vmax_vel:.4f}]")
    print(f"  Error:    [0, {global_err_max:.4f}]")
    print(f"  Std:      [0, {global_std_max:.4f}]")
    print(f"\nAll panels saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
