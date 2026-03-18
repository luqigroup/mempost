"""Plotting utilities for posterior collapse experiments.

All functions are pure (no side effects except file I/O) and accept
explicit data arguments rather than relying on class state.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Tuple


PLOT_CONFIG = {
    # Colors
    "color_prior": "#1f77b4",
    "color_posterior": "#d62728",
    "color_training": "#2ca02c",
    "color_score": "#4a4a4a",
    "color_likelihood": "#ff7f0e",
    "cmap_density": "RdBu_r",
    "cmap_score": "viridis",

    # Figure sizes
    "figsize_score_posterior": (14, 4.5),
    "figsize_modes": (6, 4),

    # Font sizes
    "fontsize_title": 13,
    "fontsize_label": 12,
    "fontsize_tick": 10,
    "fontsize_legend": 10,
    "fontsize_panel": 14,

    # Aesthetics
    "dpi": 300,
    "bbox_inches": "tight",
    "linewidth_default": 2.0,
    "alpha_fill": 0.3,
    "grid_alpha": 0.3,
    "markersize_training": 80,
}


def plot_score_field_and_posterior(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    score_fields: List[torch.Tensor],
    log_posteriors: List[torch.Tensor],
    x_train: torch.Tensor,
    sigma_values: List[float],
    y_obs: torch.Tensor,
    save_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot score field and posterior for multiple sigma values.

    Creates a 2-row, len(sigma_values)-column figure. Top row: score field
    magnitude with quiver overlay. Bottom row: posterior log-density.

    Args:
        x_grid: Grid x-coordinates [nx].
        y_grid: Grid y-coordinates [ny].
        score_fields: List of score tensors, each [nx, ny, 2].
        log_posteriors: List of log-posterior tensors, each [nx, ny].
        x_train: Training examples [N, 2].
        sigma_values: List of sigma values for panel titles.
        y_obs: Observed data point for annotation.
        save_path: Path to save the figure.
        config: Optional config overrides.
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    n_panels = len(sigma_values)
    fig, axes = plt.subplots(
        2, n_panels,
        figsize=(4.5 * n_panels, 8),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    X, Y = torch.meshgrid(x_grid, y_grid, indexing="ij")
    x_np = x_grid.numpy()
    y_np = y_grid.numpy()
    X_np, Y_np = X.numpy(), Y.numpy()
    x_train_np = x_train.numpy()

    for j in range(n_panels):
        # Top row: score field
        ax = axes[0, j]
        score = score_fields[j].numpy()
        magnitude = np.sqrt(score[:, :, 0] ** 2 + score[:, :, 1] ** 2)
        im = ax.pcolormesh(
            X_np, Y_np,
            np.log10(magnitude + 1e-10),
            cmap=cfg["cmap_score"],
            shading="auto",
        )
        # Quiver with subsampling
        step = max(1, len(x_np) // 15)
        ax.quiver(
            X_np[::step, ::step], Y_np[::step, ::step],
            score[::step, ::step, 0], score[::step, ::step, 1],
            color="white", alpha=0.7, scale=None,
        )
        ax.scatter(
            x_train_np[:, 0], x_train_np[:, 1],
            c=cfg["color_training"], marker="^",
            s=cfg["markersize_training"], edgecolors="white",
            linewidths=0.8, zorder=5, label="Training",
        )
        ax.set_title(
            rf"$\sigma = {sigma_values[j]:.2f}$",
            fontsize=cfg["fontsize_title"],
        )
        if j == 0:
            ax.set_ylabel("Score field\n$x_2$", fontsize=cfg["fontsize_label"])
        ax.set_xlabel("$x_1$", fontsize=cfg["fontsize_label"])
        ax.tick_params(labelsize=cfg["fontsize_tick"])

        # Bottom row: posterior
        ax = axes[1, j]
        log_post = log_posteriors[j].numpy()
        # Normalize for visualization
        log_post_shifted = log_post - log_post.max()
        posterior = np.exp(log_post_shifted)
        posterior = posterior / posterior.sum()
        im2 = ax.pcolormesh(
            X_np, Y_np, posterior,
            cmap=cfg["cmap_density"],
            shading="auto",
        )
        ax.scatter(
            x_train_np[:, 0], x_train_np[:, 1],
            c=cfg["color_training"], marker="^",
            s=cfg["markersize_training"], edgecolors="white",
            linewidths=0.8, zorder=5,
        )
        if j == 0:
            ax.set_ylabel(
                "Posterior $p(\\mathbf{x} \\mid \\mathbf{y})$\n$x_2$",
                fontsize=cfg["fontsize_label"],
            )
        ax.set_xlabel("$x_1$", fontsize=cfg["fontsize_label"])
        ax.tick_params(labelsize=cfg["fontsize_tick"])

    # Panel labels
    for j in range(n_panels):
        label_top = chr(ord("a") + j)
        label_bot = chr(ord("a") + n_panels + j)
        axes[0, j].text(
            0.03, 0.95, f"({label_top})",
            transform=axes[0, j].transAxes,
            fontsize=cfg["fontsize_panel"],
            fontweight="bold", va="top", color="white",
        )
        axes[1, j].text(
            0.03, 0.95, f"({label_bot})",
            transform=axes[1, j].transAxes,
            fontsize=cfg["fontsize_panel"],
            fontweight="bold", va="top",
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_effective_modes(
    sigma_values: torch.Tensor,
    effective_modes: Dict[int, torch.Tensor],
    save_path: str,
    gamma: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot effective number of posterior modes vs sigma for different N.

    Args:
        sigma_values: Sigma values [n_sigma].
        effective_modes: Dict mapping N -> effective modes [n_sigma].
        save_path: Path to save the figure.
        gamma: Observation noise std (for annotation).
        config: Optional config overrides.
    """
    cfg = {**PLOT_CONFIG, **(config or {})}
    fig, ax = plt.subplots(figsize=cfg["figsize_modes"])

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(effective_modes)))
    for i, (N, modes) in enumerate(sorted(effective_modes.items())):
        ax.plot(
            sigma_values.numpy(),
            modes.numpy(),
            color=colors[i],
            linewidth=cfg["linewidth_default"],
            label=f"$N = {N}$",
        )

    ax.set_xlabel(r"$\sigma$", fontsize=cfg["fontsize_label"])
    ax.set_ylabel("Effective posterior modes", fontsize=cfg["fontsize_label"])
    ax.set_xscale("log")
    ax.legend(fontsize=cfg["fontsize_legend"])
    ax.grid(True, alpha=cfg["grid_alpha"])
    ax.tick_params(labelsize=cfg["fontsize_tick"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=cfg["dpi"], bbox_inches=cfg["bbox_inches"])
    plt.close(fig)
    print(f"Saved: {save_path}")
