"""Stylized GMM posterior collapse experiment.

Demonstrates posterior collapse under a memorized GMM prior using
analytical computations (no neural network training needed).

Produces:
  - Figure 1 (1D): Prior, likelihood, and posterior at three sigma values,
    showing collapse from continuous distribution to delta-function mixture.
  - Figure 2 (2D): Posterior density with linearized posterior confidence
    ellipses showing collapse from overlapping to isolated components.
  - Figure 3: Effective posterior modes vs sigma for different N.

Authors: Ali Siahkoohi, Davide Sabeddu
"""

import argparse
import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from projorg import plotsdir, setup_environment

from mempost import (
    gmm_log_density,
    gmm_score,
    gmm_posterior,
    gmm_posterior_weights,
    linearized_posterior_components,
)

CONFIG_FILE = "stylized_gmm.json"

# Paper figure output directory
FIGS_DIR = os.path.expanduser("~/Documents/IMAGE26-brainstorming/figs")


# ---------- Forward operators ----------


def forward_1d(x: torch.Tensor) -> torch.Tensor:
    """Nonlinear forward operator F: R -> R.

    F(x) = x + 0.3 * x^3

    Args:
        x: Model parameters [batch_size, 1] or [1].

    Returns:
        Predicted data, same shape as x.
    """
    return x + 0.3 * x ** 3


def forward_2d(x: torch.Tensor) -> torch.Tensor:
    """Nonlinear forward operator F: R^2 -> R^2.

    F(x) = [x_1 + 0.3 * x_1 * x_2,  x_2 + 0.2 * x_1^2]

    Args:
        x: Model parameters [batch_size, 2] or [2].

    Returns:
        Predicted data, same shape as x.
    """
    squeeze = x.ndim == 1
    if squeeze:
        x = x.unsqueeze(0)
    y = torch.stack([
        x[:, 0] + 0.3 * x[:, 0] * x[:, 1],
        x[:, 1] + 0.2 * x[:, 0] ** 2,
    ], dim=1)
    if squeeze:
        y = y.squeeze(0)
    return y


# ---------- Figure 1: 1D posterior collapse ----------


def run_figure1(args: argparse.Namespace) -> None:
    """Generate Figure 1: 1D posterior collapse at three sigma values.

    Shows prior, likelihood, and posterior in three panels. At large sigma,
    posterior is a smooth continuous distribution. At small sigma, it collapses
    to discrete spikes at the training examples weighted by likelihood.

    Args:
        args: Configuration arguments.
    """
    print("\n=== Figure 1: 1D posterior collapse ===")
    torch.manual_seed(int(args.seed))

    N = int(args.n_training)
    gamma = float(args.gamma)
    sigma_values = [float(s) for s in args.sigma_values.split(",")]

    # Training examples: placed so that two have comparable data fit.
    # F(x) = x + 0.3*x^3.
    # F(-0.4) = -0.4192, F(0.4) = 0.4192
    # With gamma=0.3, the likelihood exp(-|F(x)-y|^2/(2*0.09)) is
    # broad enough that x=-0.4 and x=0.4 both have nontrivial weight
    # when y_obs ≈ 0.0, showing discrete bi-modality in the collapsed
    # posterior.  Spacing them to ±0.4 makes the two peaks clearly
    # visible in panel (c).
    x_train = torch.tensor([[-1.5], [-0.4], [0.4], [1.2], [2.5]])[:N]

    # Observation at 0 — equidistant from F(-0.4) and F(0.4) so both
    # get near-equal weight.
    y_obs = torch.tensor([0.0])
    print(f"y_obs = {y_obs.item():.3f}")

    # Limiting weights
    weights = gmm_posterior_weights(x_train, forward_1d, y_obs, gamma)
    for n in range(N):
        print(
            f"  x_{n} = {x_train[n, 0]:.2f}, "
            f"F(x_{n}) = {forward_1d(x_train[n:n+1]).item():.3f}, "
            f"w_{n} = {weights[n].item():.4f}"
        )

    # Evaluation grid
    x_grid = torch.linspace(-3.5, 3.5, 500).unsqueeze(1)  # [500, 1]

    # Likelihood (fixed, independent of sigma)
    pred = forward_1d(x_grid)  # [500, 1]
    log_likelihood = -0.5 * ((pred - y_obs) ** 2).sum(dim=-1) / gamma ** 2

    fig, axes = plt.subplots(1, len(sigma_values), figsize=(5.5 * len(sigma_values), 4.5))
    if len(sigma_values) == 1:
        axes = [axes]

    dx = (x_grid[1, 0] - x_grid[0, 0]).item()

    for j, sigma in enumerate(sigma_values):
        ax = axes[j]
        x_np = x_grid[:, 0].numpy()

        # GMM prior
        log_prior = gmm_log_density(x_grid, x_train, sigma)
        prior = torch.exp(log_prior - log_prior.max())
        prior = prior / (prior.sum() * dx)

        # Posterior
        log_post = gmm_posterior(
            x_grid, x_train, sigma, forward_1d, y_obs, gamma,
        )
        post = torch.exp(log_post - log_post.max())
        post = post / (post.sum() * dx)

        # Linearized posterior (Eq. 7)
        mu_n, Sigma_n, w_n = linearized_posterior_components(
            x_train, sigma, forward_1d, y_obs, gamma,
        )
        lin_post = torch.zeros(x_grid.shape[0])
        for n in range(N):
            s_n = Sigma_n[n, 0, 0].item()  # 1D: scalar variance
            lin_post += w_n[n].item() * torch.exp(
                -0.5 * (x_grid[:, 0] - mu_n[n, 0].item())**2 / s_n
            ) / torch.tensor(2.0 * torch.pi * s_n).sqrt()
        lin_post = lin_post / (lin_post.sum() * dx)

        # Likelihood (scaled for display)
        lik = torch.exp(log_likelihood - log_likelihood.max())
        lik = lik / lik.max() * post.max() * 0.6

        # Plot posterior
        ax.fill_between(
            x_np, 0, post.numpy(),
            color="#d62728", alpha=0.3, label="Posterior",
        )
        ax.plot(x_np, post.numpy(), color="#d62728", linewidth=2.0)

        # Plot linearized posterior
        ax.plot(
            x_np, lin_post.numpy(),
            color="black", linewidth=1.8, linestyle="--",
            label="Linearized (Eq.\u20097)", alpha=0.85,
        )

        # Plot prior
        ax.plot(
            x_np, prior.numpy(),
            color="#1f77b4", linewidth=1.5, linestyle="-",
            label="Prior", alpha=0.7,
        )

        ax.plot(
            x_np, lik.numpy(),
            color="#ff7f0e", linewidth=1.5, linestyle="--",
            label="Likelihood", alpha=0.7,
        )

        # Vertical dashed lines at training point locations
        for n in range(N):
            ax.axvline(
                x_train[n, 0].item(), color="#2ca02c",
                linestyle=":", linewidth=0.8, alpha=0.5, zorder=1,
            )

        # Training examples
        ax.scatter(
            x_train[:, 0].numpy(),
            torch.zeros(N).numpy(),
            c="#2ca02c", marker="^", s=80, edgecolors="black",
            linewidths=0.5, zorder=5, label="Training",
        )

        ax.set_title(rf"$\sigma = {sigma}$", fontsize=15)
        ax.set_xlabel("$x$", fontsize=14)
        if j == 0:
            ax.set_ylabel("Density", fontsize=14)
            ax.legend(fontsize=11, loc="upper right")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=12)

        # Panel label outside plot (top-left)
        ax.text(
            -0.02, 1.02, f"({chr(ord('a') + j)})",
            transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right",
        )

    plt.tight_layout()

    # Save to experiment logs
    save_path = os.path.join(plotsdir(args.experiment), "figure1.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    # Save to paper figs
    os.makedirs(FIGS_DIR, exist_ok=True)
    paper_path = os.path.join(FIGS_DIR, "figure1.png")
    fig.savefig(paper_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {paper_path}")
    plt.close(fig)


# ---------- Figure 2: 2D posterior with linearized components ----------


def _confidence_ellipse(mu, Sigma, ax, n_std=2.0, **kwargs):
    """Draw a confidence ellipse from mean and 2x2 covariance."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    angle = torch.atan2(eigvecs[1, 1], eigvecs[0, 1]).item()
    width = 2.0 * n_std * eigvals[1].sqrt().item()
    height = 2.0 * n_std * eigvals[0].sqrt().item()

    ellipse = Ellipse(
        (mu[0].item(), mu[1].item()),
        width=width, height=height,
        angle=np.degrees(angle),
        **kwargs,
    )
    ax.add_patch(ellipse)
    return ellipse


def run_figure2(args: argparse.Namespace) -> None:
    """Generate Figure 2: 2D linearized posterior components.

    Shows the posterior density as filled contours with confidence ellipses
    from the linearized posterior (Eq. 7) at three sigma values.

    Args:
        args: Configuration arguments.
    """
    print("\n=== Figure 2: 2D linearized posterior ===")
    torch.manual_seed(int(args.seed))

    sigma_values = [float(s) for s in args.sigma_values.split(",")]
    N = int(args.n_training)
    gamma = float(args.gamma)
    grid_res = int(args.grid_resolution)

    # Training examples in 2D
    x_train = 1.8 * torch.randn(N, 2)

    # Observation
    y_obs = forward_2d(torch.tensor([0.0, 0.0]))

    # Tight grid around training data to reduce white space
    margin = 0.8
    x_lo = x_train[:, 0].min().item() - margin
    x_hi = x_train[:, 0].max().item() + margin
    y_lo = x_train[:, 1].min().item() - margin
    y_hi = x_train[:, 1].max().item() + margin

    x_grid = torch.linspace(x_lo, x_hi, grid_res)
    y_grid = torch.linspace(y_lo, y_hi, grid_res)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing="ij")
    points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    fig, axes = plt.subplots(
        1, len(sigma_values),
        figsize=(5.5 * len(sigma_values), 5.0),
    )
    if len(sigma_values) == 1:
        axes = [axes]

    # Vivid, distinct colors for each component
    vivid_colors = [
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#ff7f00",  # orange
        "#984ea3",  # purple
    ]

    for j, sigma in enumerate(sigma_values):
        print(f"  Computing sigma = {sigma:.3f} ...")
        ax = axes[j]

        # Posterior density contours (light gray background)
        log_post = gmm_posterior(
            points, x_train, sigma, forward_2d, y_obs, gamma,
        ).reshape(grid_res, grid_res)
        post = torch.exp(log_post - log_post.max())
        post_np = post.numpy()

        ax.contourf(
            X.numpy(), Y.numpy(), post_np,
            levels=15, cmap="Greys", alpha=0.35,
        )
        ax.contour(
            X.numpy(), Y.numpy(), post_np,
            levels=8, colors="gray", linewidths=0.5, alpha=0.5,
        )

        # Linearized posterior components (Eq. 7)
        mu_n, Sigma_n, w_n = linearized_posterior_components(
            x_train, sigma, forward_2d, y_obs, gamma,
        )

        # Draw confidence ellipses — vivid color per component, opacity ∝ weight
        w_max = w_n.max().item()
        for n in range(N):
            wt = w_n[n].item()
            rel_w = wt / (w_max + 1e-10)
            fill_alpha = max(0.15, 0.55 * rel_w)
            edge_alpha = max(0.4, rel_w)
            lw = max(1.2, 3.0 * rel_w)
            color = vivid_colors[n % len(vivid_colors)]

            # 2-sigma filled ellipse
            _confidence_ellipse(
                mu_n[n], Sigma_n[n], ax,
                n_std=2.0,
                facecolor=(*matplotlib.colors.to_rgb(color), fill_alpha),
                edgecolor=(*matplotlib.colors.to_rgb(color), edge_alpha),
                linewidth=lw,
                zorder=3,
            )
            # 1-sigma inner contour
            _confidence_ellipse(
                mu_n[n], Sigma_n[n], ax,
                n_std=1.0,
                facecolor="none",
                edgecolor=(*matplotlib.colors.to_rgb(color), edge_alpha),
                linewidth=lw * 0.7,
                linestyle="--",
                zorder=3,
            )

        # Training examples (x_n) as stars
        for n in range(N):
            ax.scatter(
                x_train[n, 0].item(), x_train[n, 1].item(),
                color=vivid_colors[n % len(vivid_colors)],
                marker="*", s=220, edgecolors="black",
                linewidths=0.7, zorder=5,
            )
        # Component means (mu_n) as crosses
        for n in range(N):
            ax.scatter(
                mu_n[n, 0].item(), mu_n[n, 1].item(),
                color=vivid_colors[n % len(vivid_colors)],
                marker="+", s=120, linewidths=2.5, zorder=6,
            )

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(rf"$\sigma = {sigma}$", fontsize=15)
        ax.set_xlabel("$x_1$", fontsize=14)
        if j == 0:
            ax.set_ylabel("$x_2$", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_aspect("equal")

        # Panel label outside plot (top-left)
        ax.text(
            -0.02, 1.02, f"({chr(ord('a') + j)})",
            transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right",
        )

    plt.tight_layout()

    save_path = os.path.join(plotsdir(args.experiment), "figure2.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    os.makedirs(FIGS_DIR, exist_ok=True)
    paper_path = os.path.join(FIGS_DIR, "figure2.png")
    fig.savefig(paper_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {paper_path}")
    plt.close(fig)


# ---------- Figure 3: Effective modes vs sigma ----------


def run_figure3(args: argparse.Namespace) -> None:
    """Generate Figure 3: effective posterior modes vs sigma for different N.

    Shows that sparse training sets (small N) collapse at larger sigma,
    consistent with the dominance condition.

    Args:
        args: Configuration arguments.
    """
    print("\n=== Figure 3: Effective modes vs sigma ===")
    torch.manual_seed(int(args.seed))

    gamma = float(args.gamma)
    n_values = [int(n) for n in args.n_values.split(",")]
    # Extend upper range so all curves can plateau near N
    sigma_sweep = torch.logspace(
        np.log10(float(args.sigma_sweep_min)),
        np.log10(5.0),  # go higher than config to let curves plateau
        int(args.sigma_sweep_steps) + 20,
    )

    # Generate a large pool of training examples
    max_N = max(n_values)
    x_pool_1d = torch.linspace(-3.0, 3.0, max_N).unsqueeze(1)

    # Observation at zero — symmetric, so no single training point
    # is strongly preferred by the likelihood, letting curves reach N
    y_obs = torch.tensor([0.0])

    effective_modes = {}
    for N in n_values:
        # Use evenly spaced subset
        indices = torch.linspace(0, max_N - 1, N).long()
        x_train = x_pool_1d[indices]

        modes = []
        for sigma in sigma_sweep:
            sigma_val = sigma.item()
            # Compute posterior weights at each training point
            # At finite sigma, the posterior is a weighted GMM; the effective
            # number of modes is exp(entropy of the component weights)
            weights = _compute_posterior_component_weights(
                x_train, sigma_val, forward_1d, y_obs, gamma,
            )
            log_w = torch.log(weights + 1e-30)
            entropy = -(weights * log_w).sum()
            modes.append(torch.exp(entropy).item())

        effective_modes[N] = torch.tensor(modes)
        print(
            f"  N = {N}: modes range "
            f"[{min(modes):.2f}, {max(modes):.2f}]"
        )

    # Plot — x-axis: large sigma (well-regularised) on left,
    # small sigma (memorised) on right.
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(n_values)))

    for i, N in enumerate(n_values):
        ax.plot(
            sigma_sweep.numpy(),
            effective_modes[N].numpy(),
            color=colors[i], linewidth=2.0,
            label=f"$N = {N}$",
        )
        # Horizontal dashed line at N
        ax.axhline(
            N, color=colors[i], linestyle="--",
            linewidth=0.8, alpha=0.4,
        )

        # Dominance condition (Eq. 8): σ_crit = sqrt(δ · γ / ||F'||)
        # δ = inter-point spacing for evenly spaced points in [-3, 3]
        indices = torch.linspace(0, max_N - 1, N).long()
        x_train_N = x_pool_1d[indices]
        delta = 6.0 / (N - 1) if N > 1 else 6.0
        # Average Jacobian norm: F'(x) = 1 + 0.9*x^2
        grad_F_norm = (1.0 + 0.9 * x_train_N**2).mean().item()
        sigma_crit = np.sqrt(delta * gamma / grad_F_norm)

        # Interpolate effective modes at sigma_crit
        sigma_np = sigma_sweep.numpy()
        modes_np = effective_modes[N].numpy()
        if sigma_crit >= sigma_np.min() and sigma_crit <= sigma_np.max():
            modes_at_crit = np.interp(sigma_crit, sigma_np, modes_np)
            ax.scatter(
                sigma_crit, modes_at_crit,
                color=colors[i], marker="D", s=70,
                edgecolors="black", linewidths=0.8, zorder=6,
            )

    # Legend entry for dominance condition markers
    ax.scatter(
        [], [], color="gray", marker="D", s=70,
        edgecolors="black", linewidths=0.8,
        label=r"$\sigma_{\mathrm{crit}}$ (Eq. 7)",
    )

    ax.set_xlabel(r"$\sigma$", fontsize=12)
    ax.set_ylabel("Effective posterior modes", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Flip so large sigma is on the left (well-regularised → memorised)
    ax.invert_xaxis()
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=10)
    ax.set_ylim(bottom=0.8)

    # Vertical line at typical sigma_min
    sigma_min = 0.01
    ax.axvline(
        sigma_min, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
    )
    ax.annotate(
        r"typical $\sigma_{\min}$",
        xy=(sigma_min, 0.95), xycoords=("data", "axes fraction"),
        fontsize=9, fontstyle="italic", color="gray",
        ha="right", va="top",
        xytext=(-6, 0), textcoords="offset points",
    )

    # Annotations
    ax.annotate(
        "well-regularised", xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=9, fontstyle="italic", color="gray", ha="left",
    )
    ax.annotate(
        "memorised", xy=(0.98, 0.02), xycoords="axes fraction",
        fontsize=9, fontstyle="italic", color="gray", ha="right",
    )

    plt.tight_layout()

    save_path = os.path.join(plotsdir(args.experiment), "figure3.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    os.makedirs(FIGS_DIR, exist_ok=True)
    paper_path = os.path.join(FIGS_DIR, "figure3.png")
    fig.savefig(paper_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {paper_path}")
    plt.close(fig)


def _compute_posterior_component_weights(
    x_train: torch.Tensor,
    sigma: float,
    forward_op,
    y_obs: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute posterior weights for each GMM component.

    Integrates p_n(x) * p(y|x) over x for each component n, which for
    Gaussian components and smooth likelihood can be approximated by
    evaluating the likelihood at the component center (valid when sigma
    is small relative to the likelihood scale).

    For larger sigma, we use numerical integration on a grid.

    Args:
        x_train: Training examples [N, d].
        sigma: Component standard deviation.
        forward_op: Callable F.
        y_obs: Observed data.
        gamma: Observation noise std.

    Returns:
        Normalized component weights [N].
    """
    N, d = x_train.shape

    if d == 1:
        # 1D: numerical integration on a fine grid
        x_grid = torch.linspace(-5.0, 5.0, 2000).unsqueeze(1)
        log_lik = -0.5 * ((forward_op(x_grid) - y_obs) ** 2).sum(
            dim=-1
        ) / gamma ** 2  # [G]

        weights = torch.zeros(N)
        for n in range(N):
            # Component n log-density
            log_comp = (
                -0.5 * ((x_grid - x_train[n]) ** 2).sum(dim=-1) / sigma ** 2
            )
            # Integrate component * likelihood
            log_integrand = log_comp + log_lik
            # Log-sum-exp for numerical stability
            weights[n] = torch.logsumexp(log_integrand, dim=0)

        return torch.softmax(weights, dim=0)
    else:
        # Higher-D: use Laplace approximation at component centers
        return gmm_posterior_weights(x_train, forward_op, y_obs, gamma)


def run_individual_panels(args: argparse.Namespace) -> None:
    """Generate individual panels for combined stylized figure.

    Produces 5 panels:
      - 1d_sigma0.3.png, 1d_sigma0.05.png  (wide/flat 1D posteriors)
      - 2d_sigma0.3.png, 2d_sigma0.05.png  (square 2D posteriors)
      - effective_modes.png                  (tall figure3)
    """
    print("\n=== Individual panels for combined figure ===")
    torch.manual_seed(int(args.seed))

    N = int(args.n_training)
    gamma = float(args.gamma)
    grid_res = int(args.grid_resolution)
    panel_dir = os.path.join(FIGS_DIR, "stylized_panels")
    os.makedirs(panel_dir, exist_ok=True)

    # ---- 1D panels (flat aspect ratio) ----
    x_train = torch.tensor([[-1.5], [-0.4], [0.4], [1.2], [2.5]])[:N]
    y_obs = torch.tensor([0.0])
    x_grid = torch.linspace(-3.5, 3.5, 500).unsqueeze(1)
    pred = forward_1d(x_grid)
    log_likelihood = -0.5 * ((pred - y_obs) ** 2).sum(dim=-1) / gamma ** 2
    dx = (x_grid[1, 0] - x_grid[0, 0]).item()

    # Pre-compute all posteriors to find global y-max for consistent ylim.
    all_1d_data = {}
    for sigma in [0.5, 0.3, 0.05]:
        log_prior = gmm_log_density(x_grid, x_train, sigma)
        prior = torch.exp(log_prior - log_prior.max())
        prior = prior / (prior.sum() * dx)
        log_post = gmm_posterior(x_grid, x_train, sigma, forward_1d, y_obs, gamma)
        post = torch.exp(log_post - log_post.max())
        post = post / (post.sum() * dx)
        mu_n, Sigma_n, w_n = linearized_posterior_components(
            x_train, sigma, forward_1d, y_obs, gamma)
        lin_post = torch.zeros(x_grid.shape[0])
        for n in range(N):
            s_n = Sigma_n[n, 0, 0].item()
            lin_post += w_n[n].item() * torch.exp(
                -0.5 * (x_grid[:, 0] - mu_n[n, 0].item())**2 / s_n
            ) / torch.tensor(2.0 * torch.pi * s_n).sqrt()
        lin_post = lin_post / (lin_post.sum() * dx)
        all_1d_data[sigma] = (prior, post, lin_post, mu_n, Sigma_n, w_n)
    # Compute normalized likelihood density (same for all sigma)
    lik_unnorm = torch.exp(log_likelihood - log_likelihood.max())
    lik_density = lik_unnorm / (lik_unnorm.sum() * dx)
    # Global y-max across posterior, prior, likelihood, linearized
    y_max_1d = max(
        max(all_1d_data[s][1].max().item() for s in all_1d_data),  # posterior
        max(all_1d_data[s][0].max().item() for s in all_1d_data),  # prior
        max(all_1d_data[s][2].max().item() for s in all_1d_data),  # linearized
        lik_density.max().item(),
    ) * 1.1

    PANEL_FIGSIZE = (4.5, 2.4)
    LABEL_FS = 9
    LEGEND_FS = 7.5
    TICK_FS = 8

    for sigma in [0.5, 0.3, 0.05]:
        fig, ax = plt.subplots(1, 1, figsize=PANEL_FIGSIZE)
        x_np = x_grid[:, 0].numpy()

        prior, post, lin_post, mu_n, Sigma_n, w_n = all_1d_data[sigma]

        lik = torch.exp(log_likelihood - log_likelihood.max())
        lik = lik / (lik.sum() * dx)

        ax.fill_between(x_np, 0, post.numpy(), color="#d62728", alpha=0.3,
                         label="Posterior")
        ax.plot(x_np, post.numpy(), color="#d62728", linewidth=2.0)
        ax.plot(x_np, lin_post.numpy(), color="black", linewidth=1.8,
                linestyle="--", label="Linearized", alpha=0.85)
        ax.plot(x_np, prior.numpy(), color="#1f77b4", linewidth=1.5,
                linestyle="-", label="Prior", alpha=0.7)
        ax.plot(x_np, lik.numpy(), color="#ff7f0e", linewidth=1.5,
                linestyle="--", label="Likelihood", alpha=0.7)

        for n in range(N):
            ax.axvline(x_train[n, 0].item(), color="#2ca02c",
                       linestyle=":", linewidth=0.8, alpha=0.5, zorder=1)
        ax.scatter(x_train[:, 0].numpy(), torch.zeros(N).numpy(),
                   c="#2ca02c", marker="^", s=80, edgecolors="black",
                   linewidths=0.5, zorder=5, label="Training")

        ax.set_xlabel("$x$", fontsize=LABEL_FS)
        ax.set_ylabel("Density", fontsize=LABEL_FS)
        ax.set_xlim(-2, 3)
        if sigma >= 0.3:
            ax.set_ylim(0, 2)
        else:
            ax.set_ylim(0, y_max_1d)
        ax.legend(fontsize=LEGEND_FS, loc="upper right", ncol=2)
        ax.tick_params(labelsize=TICK_FS)

        fname = f"1d_sigma{sigma}.png"
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(panel_dir, fname), dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  Saved: {fname}")

    # ---- 2D panels ----
    # Pentagon training layout, true model near training example 0.
    angles = torch.linspace(0, 2 * np.pi, N + 1)[:N]
    radius = 2.0
    x_train_2d = torch.stack([radius * torch.cos(angles),
                               radius * torch.sin(angles)], dim=1)
    # True model near training example 0 but distinct
    x_true_2d = 0.7 * x_train_2d[0]
    y_obs_2d = forward_2d(x_true_2d)

    margin = 0.8
    x_lo = x_train_2d[:, 0].min().item() - margin
    x_hi = x_train_2d[:, 0].max().item() + margin
    y_lo = x_train_2d[:, 1].min().item() - margin
    y_hi = x_train_2d[:, 1].max().item() + margin

    x_g = torch.linspace(x_lo, x_hi, grid_res)
    y_g = torch.linspace(y_lo, y_hi, grid_res)
    X, Y = torch.meshgrid(x_g, y_g, indexing="ij")
    points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # Colors matching 1D palette
    COL_POST = "#d62728"    # red — true posterior (shaded)
    COL_LIN = "black"       # black/gray — linearized posterior (contours)
    COL_PRIOR = "#1f77b4"   # blue — prior
    COL_LIK = "#ff7f0e"     # orange — likelihood (contours)
    COL_TRAIN = "#2ca02c"   # green — training markers

    # Pre-compute 2D likelihood (fixed across sigma)
    pred_2d = forward_2d(points)
    log_lik_2d = -0.5 * ((pred_2d - y_obs_2d) ** 2).sum(dim=-1) / gamma ** 2
    log_lik_2d = log_lik_2d.reshape(grid_res, grid_res)
    lik_2d = torch.exp(log_lik_2d - log_lik_2d.max()).numpy()

    from matplotlib.patches import Patch

    # Pre-compute prior density on grid (for blue contours)
    def gmm_prior_density(pts, x_tr, sig):
        """GMM prior density on grid."""
        log_p = gmm_log_density(pts, x_tr, sig)
        return torch.exp(log_p - log_p.max())

    for sigma in [0.5, 0.3, 0.05]:
        fig, ax = plt.subplots(1, 1, figsize=PANEL_FIGSIZE)

        # Exact posterior
        log_post = gmm_posterior(
            points, x_train_2d, sigma, forward_2d, y_obs_2d, gamma
        ).reshape(grid_res, grid_res)
        post = torch.exp(log_post - log_post.max())
        post_np = post.numpy()

        # Prior density
        prior_2d = gmm_prior_density(points, x_train_2d, sigma
                                      ).reshape(grid_res, grid_res).numpy()

        # Linearized posterior density (sum of weighted Gaussians)
        mu_n, Sigma_n, w_n = linearized_posterior_components(
            x_train_2d, sigma, forward_2d, y_obs_2d, gamma)
        lin_post_2d = torch.zeros(grid_res, grid_res)
        for n in range(N):
            diff = points - mu_n[n].unsqueeze(0)  # (M, 2)
            Sinv = torch.inverse(Sigma_n[n])       # (2, 2)
            mahal = (diff @ Sinv * diff).sum(dim=-1)  # (M,)
            det_S = torch.det(Sigma_n[n])
            component = w_n[n] * torch.exp(-0.5 * mahal) / \
                        (2 * np.pi * det_S.sqrt())
            lin_post_2d += component.reshape(grid_res, grid_res)
        lin_post_np = lin_post_2d.numpy()

        ax.set_facecolor("white")

        # 1. Prior contours (blue, solid)
        prior_levels = np.linspace(prior_2d.max() * 0.05,
                                    prior_2d.max(), 10)
        ax.contour(X.numpy(), Y.numpy(), prior_2d,
                   levels=prior_levels, colors=COL_PRIOR, linewidths=0.6,
                   alpha=0.5, zorder=1)

        # 2. Likelihood contours (orange, dashed)
        lik_levels = np.linspace(lik_2d.max() * 0.05, lik_2d.max(), 8)
        ax.contour(X.numpy(), Y.numpy(), lik_2d,
                   levels=lik_levels, colors=COL_LIK, linewidths=0.8,
                   linestyles="--", alpha=0.6, zorder=2)

        # 3. Linearized posterior contours (gray, solid, more visible)
        lin_levels = np.linspace(lin_post_np.max() * 0.03,
                                  lin_post_np.max(), 8)
        ax.contour(X.numpy(), Y.numpy(), lin_post_np,
                   levels=lin_levels, colors="gray", linewidths=1.0,
                   alpha=0.8, zorder=3)

        # 4. True posterior (red shaded only, no contour lines)
        post_levels = np.linspace(post_np.max() * 0.05,
                                   post_np.max(), 8)
        ax.contourf(X.numpy(), Y.numpy(), post_np,
                    levels=post_levels, cmap="Reds", alpha=0.6, zorder=4)

        # Training markers (green)
        ax.scatter(x_train_2d[:, 0].numpy(), x_train_2d[:, 1].numpy(),
                   c=COL_TRAIN, marker="^", s=60, edgecolors="black",
                   linewidths=0.5, zorder=8)

        ax.legend(
            handles=[
                Patch(facecolor=COL_POST, edgecolor=COL_POST,
                      alpha=0.4, linewidth=1.0, label="Posterior"),
                plt.Line2D([0], [0], color=COL_LIK, linewidth=0.8,
                           linestyle="--", alpha=0.6, label="Likelihood"),
                plt.Line2D([0], [0], color=COL_PRIOR, linewidth=0.8,
                           alpha=0.5, label="Prior"),
                plt.Line2D([0], [0], color="gray", linewidth=0.8,
                           alpha=0.6, label="Linearized"),
                plt.Line2D([0], [0], marker="^", color=COL_TRAIN,
                           markersize=7, markeredgecolor="black",
                           linestyle="None", label="Training"),
            ],
            fontsize=LEGEND_FS, loc="upper right", framealpha=0.92, ncol=2,
        )

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("$x_1$", fontsize=LABEL_FS)
        ax.set_ylabel("$x_2$", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)

        # Zoomed inset for sigma=0.05 to show linearized/posterior overlap
        if sigma == 0.05:
            from mpl_toolkits.axes_grid1.inset_locator import (
                inset_axes, mark_inset)
            # Zoom around the dominant training example (x_train_2d[0])
            cx = x_train_2d[0, 0].item()
            cy = x_train_2d[0, 1].item()
            zoom_w = 0.12
            axins = ax.inset_axes([0.30, 0.30, 0.40, 0.40])  # centered
            # Redraw only linearized and posterior in inset
            axins.contour(X.numpy(), Y.numpy(), lin_post_np,
                          levels=lin_levels, colors="gray", linewidths=1.4,
                          alpha=0.9)
            axins.contourf(X.numpy(), Y.numpy(), post_np,
                           levels=5, cmap="Reds", alpha=0.6)
            axins.set_xlim(cx - zoom_w, cx + zoom_w)
            axins.set_ylim(cy - zoom_w, cy + zoom_w)
            axins.set_xticks([])
            axins.set_yticks([])
            axins.set_facecolor("white")
            # Draw rectangle and connector lines
            ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.0)

        fname = f"2d_sigma{sigma}.png"
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(panel_dir, fname), dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"  Saved: {fname}")

    # ---- Effective modes (tall aspect ratio) ----
    n_values = [int(n) for n in args.n_values.split(",")]
    sigma_sweep = torch.logspace(
        np.log10(float(args.sigma_sweep_min)), np.log10(5.0),
        int(args.sigma_sweep_steps) + 20)

    max_N = max(n_values)
    x_pool_1d = torch.linspace(-3.0, 3.0, max_N).unsqueeze(1)
    y_obs_1d = torch.tensor([0.0])

    effective_modes = {}
    for Nv in n_values:
        indices = torch.linspace(0, max_N - 1, Nv).long()
        x_tr = x_pool_1d[indices]
        modes = []
        for sigma in sigma_sweep:
            weights = _compute_posterior_component_weights(
                x_tr, sigma.item(), forward_1d, y_obs_1d, gamma)
            log_w = torch.log(weights + 1e-30)
            entropy = -(weights * log_w).sum()
            modes.append(torch.exp(entropy).item())
        effective_modes[Nv] = torch.tensor(modes)

    fig, ax = plt.subplots(figsize=(3.0, 4.5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(n_values)))

    for i, Nv in enumerate(n_values):
        ax.plot(sigma_sweep.numpy(), effective_modes[Nv].numpy(),
                color=colors[i], linewidth=2.0, label=f"$N = {Nv}$")
        ax.axhline(Nv, color=colors[i], linestyle="--",
                    linewidth=0.8, alpha=0.4)

        indices = torch.linspace(0, max_N - 1, Nv).long()
        x_tr = x_pool_1d[indices]
        delta = 6.0 / (Nv - 1) if Nv > 1 else 6.0
        grad_F_norm = (1.0 + 0.9 * x_tr**2).mean().item()
        sigma_crit = np.sqrt(delta * gamma / grad_F_norm)

        sigma_np = sigma_sweep.numpy()
        modes_np = effective_modes[Nv].numpy()
        if sigma_crit >= sigma_np.min() and sigma_crit <= sigma_np.max():
            modes_at_crit = np.interp(sigma_crit, sigma_np, modes_np)
            ax.scatter(sigma_crit, modes_at_crit, color=colors[i],
                       marker="D", s=70, edgecolors="black",
                       linewidths=0.8, zorder=6)

    ax.scatter([], [], color="gray", marker="D", s=70,
               edgecolors="black", linewidths=0.8,
               label=r"$\sigma_{\mathrm{crit}}$")

    ax.set_xlabel(r"$\sigma$", fontsize=11)
    ax.set_ylabel(r"$\exp(H)$, effective components", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=9)
    ax.set_ylim(bottom=0.8)

    sigma_min = 0.01
    ax.axvline(sigma_min, color="gray", linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.annotate(r"typical $\sigma_{\min}$",
                xy=(sigma_min, 0.95), xycoords=("data", "axes fraction"),
                fontsize=8, fontstyle="italic", color="gray",
                ha="right", va="top", xytext=(-6, 0),
                textcoords="offset points")

    fig.savefig(os.path.join(panel_dir, "effective_modes.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  Saved: effective_modes.png")

    print(f"\nAll panels saved to: {panel_dir}")


def main():
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "seed"],
        sequence_args_and_types=[],
    )

    run_figure1(args)
    run_figure2(args)
    run_figure3(args)
    run_individual_panels(args)


if __name__ == "__main__":
    main()
