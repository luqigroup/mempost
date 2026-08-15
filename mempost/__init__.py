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

__version__ = "1.0.0"

__all__ = [
    "gmm_log_density",
    "gmm_score",
    "gmm_posterior",
    "gmm_posterior_weights",
    "linearized_posterior_components",
]
