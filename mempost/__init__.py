"""Posterior under memorized diffusion priors.

This package implements analytical and numerical experiments demonstrating
how memorized score-based diffusion priors affect posterior inference
in Bayesian inverse problems.
"""

from .utils.gmm import (
    gmm_log_density,
    gmm_score,
    gmm_posterior,
    gmm_posterior_weights,
    linearized_posterior_components,
)
from .plotting import (
    plot_score_field_and_posterior,
    plot_effective_modes,
    PLOT_CONFIG,
)

__version__ = "0.1.0"

__all__ = [
    # GMM utilities
    "gmm_log_density",
    "gmm_score",
    "gmm_posterior",
    "gmm_posterior_weights",
    "linearized_posterior_components",
    # Plotting
    "plot_score_field_and_posterior",
    "plot_effective_modes",
    "PLOT_CONFIG",
]
