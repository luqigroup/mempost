"""Gaussian mixture model utilities for memorized score analysis.

Implements the GMM prior, score function, and posterior arising from
a diffusion model that has memorized its training data (Baptista et al., 2025).
"""

import torch
from typing import Tuple, Callable


def gmm_log_density(
    x: torch.Tensor,
    x_train: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Compute log-density of the memorized GMM prior.

    p^N(x) = (1/N) sum_n N(x; x_n, sigma^2 I)

    Args:
        x: Evaluation points [batch_size, d] or [d].
        x_train: Training examples [N, d].
        sigma: Component standard deviation.

    Returns:
        Log-density values [batch_size] or scalar.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    # x: [B, d], x_train: [N, d]
    diff = x.unsqueeze(1) - x_train.unsqueeze(0)  # [B, N, d]
    d = x_train.shape[1]
    log_components = (
        -0.5 * (diff ** 2).sum(dim=-1) / sigma ** 2
        - 0.5 * d * torch.log(torch.tensor(2.0 * torch.pi * sigma ** 2))
    )  # [B, N]
    log_density = torch.logsumexp(log_components, dim=1) - torch.log(
        torch.tensor(float(x_train.shape[0]))
    )  # [B]
    return log_density.squeeze()


def gmm_score(
    x: torch.Tensor,
    x_train: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Compute score of the memorized GMM prior.

    nabla_x log p^N(x) = sum_n w_n(x) * (x_n - x) / sigma^2

    where w_n are softmax weights (eq 3 in paper).

    Args:
        x: Evaluation points [batch_size, d] or [d].
        x_train: Training examples [N, d].
        sigma: Component standard deviation.

    Returns:
        Score vectors [batch_size, d] or [d].
    """
    squeeze = x.ndim == 1
    if squeeze:
        x = x.unsqueeze(0)
    # x: [B, d], x_train: [N, d]
    diff = x_train.unsqueeze(0) - x.unsqueeze(1)  # [B, N, d] = x_n - x
    sq_dist = (diff ** 2).sum(dim=-1)  # [B, N]
    log_weights = -0.5 * sq_dist / sigma ** 2  # [B, N]
    weights = torch.softmax(log_weights, dim=1)  # [B, N]
    score = (weights.unsqueeze(-1) * diff).sum(dim=1) / sigma ** 2  # [B, d]
    if squeeze:
        score = score.squeeze(0)
    return score


def _gmm_component_log_likelihood(
    x: torch.Tensor,
    x_train: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Log of each GMM component density.

    Args:
        x: Evaluation points [batch_size, d].
        x_train: Training examples [N, d].
        sigma: Component standard deviation.

    Returns:
        Log-densities [batch_size, N].
    """
    diff = x.unsqueeze(1) - x_train.unsqueeze(0)  # [B, N, d]
    d = x_train.shape[1]
    return (
        -0.5 * (diff ** 2).sum(dim=-1) / sigma ** 2
        - 0.5 * d * torch.log(torch.tensor(2.0 * torch.pi * sigma ** 2))
    )


def gmm_posterior_weights(
    x_train: torch.Tensor,
    forward_op,
    y_obs: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute limiting posterior weights (eq 9 in paper).

    w_n proportional to exp(-||F(x_n) - y||^2 / (2 gamma^2))

    Args:
        x_train: Training examples [N, d].
        forward_op: Callable F: [N, d] -> [N, m].
        y_obs: Observed data [m].
        gamma: Observation noise standard deviation.

    Returns:
        Normalized weights [N].
    """
    predictions = forward_op(x_train)  # [N, m]
    residuals = predictions - y_obs.unsqueeze(0)  # [N, m]
    log_weights = -0.5 * (residuals ** 2).sum(dim=-1) / gamma ** 2  # [N]
    return torch.softmax(log_weights, dim=0)


def gmm_posterior(
    x: torch.Tensor,
    x_train: torch.Tensor,
    sigma: float,
    forward_op,
    y_obs: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute unnormalized log-posterior under the GMM prior.

    log p(x | y) propto log sum_n N(x; x_n, sigma^2 I) * p(y | x)

    For the full GMM posterior (eq 8), each component is weighted by
    both proximity to x and data fit.

    Args:
        x: Evaluation points [batch_size, d] or [d].
        x_train: Training examples [N, d].
        sigma: Component standard deviation.
        forward_op: Callable F: [batch_size, d] -> [batch_size, m].
        y_obs: Observed data [m].
        gamma: Observation noise standard deviation.

    Returns:
        Unnormalized log-posterior values [batch_size] or scalar.
    """
    squeeze = x.ndim == 1
    if squeeze:
        x = x.unsqueeze(0)
    B = x.shape[0]

    # GMM prior: log N(x; x_n, sigma^2 I) for each component
    log_prior_components = _gmm_component_log_likelihood(
        x, x_train, sigma
    )  # [B, N]

    # Likelihood: log N(y; F(x), gamma^2 I)
    predictions = forward_op(x)  # [B, m]
    residuals = predictions - y_obs.unsqueeze(0)  # [B, m]
    log_likelihood = -0.5 * (residuals ** 2).sum(dim=-1) / gamma ** 2  # [B]

    # log p(x|y) propto log[ sum_n p_n(x) * p(y|x) ]
    # = log[ p(y|x) * sum_n p_n(x) ]
    # = log p(y|x) + log sum_n p_n(x)
    log_prior = torch.logsumexp(log_prior_components, dim=1)  # [B]
    log_posterior = log_prior + log_likelihood  # [B]

    if squeeze:
        log_posterior = log_posterior.squeeze(0)
    return log_posterior


def linearized_posterior_components(
    x_train: torch.Tensor,
    sigma: float,
    forward_op: Callable,
    y_obs: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute linearized Gaussian mixture posterior components (Equation 7).

    Linearizes the forward operator around each training example to obtain
    a closed-form Gaussian mixture posterior.

    Args:
        x_train: Training examples [N, d].
        sigma: Prior component standard deviation.
        forward_op: Forward operator F: R^d -> R^m.
        y_obs: Observed data [m].
        gamma: Observation noise standard deviation.

    Returns:
        mu_n: Component means [N, d].
        Sigma_n: Component covariances [N, d, d].
        w_n: Normalized component weights [N].
    """
    N, d = x_train.shape
    m = y_obs.shape[0]

    mu_n = torch.zeros(N, d)
    Sigma_n = torch.zeros(N, d, d)
    log_w = torch.zeros(N)

    for n in range(N):
        xn = x_train[n].clone().requires_grad_(True)

        # Compute Jacobian J_n = dF/dx at x_n
        Fxn = forward_op(xn.unsqueeze(0)).squeeze(0)  # [m]
        J_n = torch.zeros(m, d)
        for i in range(m):
            if xn.grad is not None:
                xn.grad.zero_()
            Fxn[i].backward(retain_graph=True)
            J_n[i] = xn.grad.clone()

        # Covariance: Sigma_n = (sigma^-2 I + gamma^-2 J^T J)^-1
        JtJ = J_n.T @ J_n  # [d, d]
        precision = (1.0 / sigma**2) * torch.eye(d) + (1.0 / gamma**2) * JtJ
        Sigma = torch.linalg.inv(precision)
        Sigma_n[n] = Sigma

        # Mean: mu_n = x_n + Sigma_n gamma^-2 J^T (y - F(x_n))
        residual = y_obs - Fxn.detach()  # [m]
        mu_n[n] = x_train[n] + Sigma @ (J_n.T @ residual) / gamma**2

        # Weight: w_n propto exp(-0.5 ||y - F(x_n)||^2_{(J sigma^2 J^T + gamma^2 I)^-1})
        cov_y = J_n @ (sigma**2 * torch.eye(d)) @ J_n.T + gamma**2 * torch.eye(m)
        cov_y_inv = torch.linalg.inv(cov_y)
        log_w[n] = -0.5 * residual @ cov_y_inv @ residual

    w_n = torch.softmax(log_w, dim=0)
    return mu_n, Sigma_n, w_n
