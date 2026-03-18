"""KL expansion prior for squared-slowness velocity fields.

Parameterizes 2D squared-slowness fields m(x) = 1/v(x)^2 via a truncated
Karhunen-Loeve expansion with cosine eigenfunctions.  The latent space is
R^d with d = K^2, and the standard Gaussian z ~ N(0, I) maps to spatially
smooth perturbations around a background slowness m_bar = 1/v_bg^2.
"""

import numpy as np
import torch


class VelocityKLPrior:
    """KL expansion prior for squared-slowness fields.

    Eigenfunctions: phi_i(x) = 2 cos(pi (i1+0.5) x1) cos(pi (i2+0.5) x2)
    Eigenvalues:    lambda_i = sigma (alpha + pi^2 ((i1+0.5)^2 + (i2+0.5)^2))^{-s/2}
    Reconstruction: m(x) = m_bar + sigma_m sum_i z_i lambda_i phi_i(x)

    Args:
        K: Cutoff index (i1, i2 in 1..K), giving d = K^2 modes.
        grid_size: Spatial grid resolution (nx = nz = grid_size).
        v_background: Background velocity in km/s.
        sigma_m: Scale for squared-slowness perturbations.
        alpha: Regularity parameter.
        s: Decay parameter.
        sigma: Overall eigenvalue scale.
    """

    def __init__(self, K=10, grid_size=64, v_background=2.0, sigma_m=0.02,
                 alpha=0.0, s=1.1, sigma=1.0):
        self.K = K
        self.d = K * K
        self.grid_size = grid_size
        self.v_background = v_background
        self.m_background = 1.0 / v_background**2
        self.sigma_m = sigma_m
        self.alpha = alpha
        self.s = s
        self.sigma = sigma

        # Multi-index pairs (i1, i2) for i1, i2 in 1..K.
        i1 = np.arange(1, K + 1)
        i2 = np.arange(1, K + 1)
        i1_grid, i2_grid = np.meshgrid(i1, i2, indexing="ij")
        self.i1 = i1_grid.ravel()
        self.i2 = i2_grid.ravel()

        # Eigenvalues.
        arg = alpha + np.pi**2 * ((self.i1 + 0.5)**2 + (self.i2 + 0.5)**2)
        self.eigenvalues = sigma * arg**(-s / 2)

        # Precompute eigenfunctions on grid.
        x = np.linspace(0, 1, grid_size, endpoint=False) + 0.5 / grid_size
        x1_grid, x2_grid = np.meshgrid(x, x, indexing="ij")

        self.phi = np.zeros((self.d, grid_size, grid_size))
        for k in range(self.d):
            self.phi[k] = 2.0 * (
                np.cos(np.pi * (self.i1[k] + 0.5) * x1_grid)
                * np.cos(np.pi * (self.i2[k] + 0.5) * x2_grid)
            )

        # Precompute KL basis matrix G: (nx*nz, d).
        # G[:, i] = sigma_m * lambda_i * phi_i (flattened).
        self._G = np.zeros((grid_size * grid_size, self.d), dtype=np.float32)
        for k in range(self.d):
            self._G[:, k] = sigma_m * self.eigenvalues[k] * self.phi[k].ravel()

    def sample(self, N, seed=None):
        """Sample N KL coefficient vectors z ~ N(0, I)."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal((N, self.d)).astype(np.float32)

    def reconstruct(self, z):
        """Reconstruct squared-slowness: m(x) = m_bar + sigma_m sum z_i lambda_i phi_i.

        Args:
            z: KL coefficients, shape (..., d).

        Returns:
            m: Squared-slowness field, shape (..., grid_size, grid_size).
        """
        z = np.asarray(z, dtype=np.float32)
        original_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.d)
        weighted = z_flat * (self.sigma_m * self.eigenvalues[None, :])
        m_field = np.einsum("ni,ijk->njk", weighted, self.phi)
        m_field = m_field + self.m_background
        # Clip to avoid unphysical negative slowness.
        m_field = np.clip(m_field, 1e-6, None)
        return m_field.reshape(*original_shape, self.grid_size, self.grid_size)

    def reconstruct_torch(self, z):
        """Reconstruct squared-slowness from torch tensor (differentiable).

        Args:
            z: KL coefficients, shape (..., d), torch.Tensor.

        Returns:
            m: Squared-slowness field, shape (..., grid_size, grid_size).
        """
        G = torch.tensor(self._G, dtype=z.dtype, device=z.device)
        original_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.d)
        m_flat = z_flat @ G.T + self.m_background  # (batch, nx*nz)
        m_field = m_flat.reshape(*original_shape, self.grid_size, self.grid_size)
        return m_field.clamp(min=1e-6)

    def to_velocity(self, m):
        """Convert squared-slowness to velocity: v = 1/sqrt(m)."""
        m = np.asarray(m)
        return 1.0 / np.sqrt(np.clip(m, 1e-6, None))

    def kl_basis_matrix(self):
        """Return the KL basis matrix G: (nx*nz, d).

        G[:, i] = sigma_m * lambda_i * phi_i (flattened).
        Used for gradient projection: grad_z = G^T @ grad_m.ravel().
        """
        return self._G.copy()

    def project(self, m_fields):
        """Project squared-slowness fields onto KL basis via least-squares.

        Args:
            m_fields: Squared-slowness fields, shape (..., grid_size, grid_size).

        Returns:
            z: KL coefficients, shape (..., d).
        """
        m_fields = np.asarray(m_fields, dtype=np.float32)
        original_shape = m_fields.shape[:-2]
        m_flat = m_fields.reshape(-1, self.grid_size * self.grid_size)
        m_perturb = m_flat - self.m_background
        z = np.linalg.lstsq(self._G, m_perturb.T, rcond=None)[0].T
        return z.reshape(*original_shape, self.d).astype(np.float32)

    def score_torch(self, z):
        """Prior score: grad_z log p(z) = -z."""
        return -z
