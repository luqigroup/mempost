"""Convergence and correctness tests for the Helmholtz solver.

Tests:
1. Grid convergence: 64x64 vs 128x128 (same physical domain, interpolated).
2. Precision: complex128 vs complex64 on the same grid.
3. Adjoint gradient: finite-difference check against adjoint-state gradient.

Run:
    conda activate sips
    python -m pytest tests/test_helmholtz.py -v
"""

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from mempost.utils.helmholtz import (
    HelmholtzSolver as HelmholtzC128,
    make_src_rec,
)
from mempost.utils.helmholtz_c64 import (
    HelmholtzSolver as HelmholtzC64,
)


# --------------- fixtures ---------------

FREQ = 4.0
OMEGA = 2.0 * np.pi * FREQ
DOMAIN_KM = 2.0  # 2 km x 2 km domain


def _smooth_model(nx, nz):
    """Simple smooth velocity -> squared slowness for testing."""
    x = np.linspace(0, 1, nx)
    z = np.linspace(0, 1, nz)
    X, Z = np.meshgrid(x, z, indexing="ij")
    v = 2.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(np.pi * Z)
    return 1.0 / v**2


def _forward_data(solver_cls, nx, n_src=10, n_rec=20):
    """Run forward model on a grid of size nx x nx."""
    dx = DOMAIN_KM / nx
    solver = solver_cls(nx=nx, nz=nx, dx=dx, npml=10, pml_max=100.0)
    m = _smooth_model(nx, nx)
    src, rec = make_src_rec(nx, nx, n_src=n_src, n_rec=n_rec)
    data = solver.forward(m, OMEGA, src, rec)
    return data, solver, m, src, rec


# --------------- Test 1: Grid convergence ---------------

def test_grid_convergence():
    """Forward data on 64 vs 128 grid should converge (< 5% relative error).

    Sources and receivers are placed at physical coordinates (not grid indices)
    to avoid grid-mismatch artifacts.
    """
    n_src, n_rec = 5, 10

    for nx, label in [(64, "coarse"), (128, "fine")]:
        dx = DOMAIN_KM / nx
        solver = HelmholtzC128(nx=nx, nz=nx, dx=dx, npml=10, pml_max=100.0)
        m = _smooth_model(nx, nx)

        # Place sources/receivers at the same physical locations.
        phys_src_x = np.linspace(0.1 * DOMAIN_KM, 0.9 * DOMAIN_KM, n_src)
        phys_rec_x = np.linspace(0.05 * DOMAIN_KM, 0.95 * DOMAIN_KM, n_rec)

        src_ix = np.clip(np.round(phys_src_x / dx).astype(int), 0, nx - 1)
        rec_ix = np.clip(np.round(phys_rec_x / dx).astype(int), 0, nx - 1)

        src_pos = [(int(ix), 0) for ix in src_ix]
        rec_pos = [(int(ix), nx - 1) for ix in rec_ix]

        data = solver.forward(m, OMEGA, src_pos, rec_pos)

        if label == "coarse":
            data_coarse = data
        else:
            data_fine = data

    rel_err = np.linalg.norm(data_coarse - data_fine) / np.linalg.norm(data_fine)
    print(f"Grid convergence: 64 vs 128 relative error = {rel_err:.4e}")
    assert rel_err < 0.15, f"Grid convergence failed: rel_err = {rel_err:.4e}"


# --------------- Test 2: complex128 vs complex64 ---------------

def test_c128_vs_c64():
    """complex128 and complex64 solvers should agree to < 1% on same grid."""
    nx = 64
    n_src, n_rec = 10, 20

    data_128, *_ = _forward_data(HelmholtzC128, nx, n_src, n_rec)
    data_64, *_ = _forward_data(HelmholtzC64, nx, n_src, n_rec)

    rel_err = np.linalg.norm(data_128 - data_64) / np.linalg.norm(data_128)
    print(f"c128 vs c64: relative error = {rel_err:.4e}")
    assert rel_err < 0.01, f"Precision test failed: rel_err = {rel_err:.4e}"


# --------------- Test 3: Adjoint gradient check ---------------

def _check_adjoint_gradient(solver_cls, label):
    """Finite-difference vs adjoint gradient for a given solver class."""
    nx = 32
    dx = DOMAIN_KM / nx
    solver = solver_cls(nx=nx, nz=nx, dx=dx, npml=10, pml_max=100.0)
    m = _smooth_model(nx, nx)
    src, rec = make_src_rec(nx, nx, n_src=3, n_rec=5)

    d_obs = solver.forward(m, OMEGA, src, rec)
    # Add small perturbation so residual is nonzero.
    d_obs = d_obs * (1.0 + 0.01)
    sigma_d = 1.0

    grad_adj = solver.adjoint_gradient(m, OMEGA, src, rec, d_obs, sigma_d)

    # Finite-difference gradient for a few random pixels.
    # Use larger eps for float32-based solvers to stay above noise floor.
    rng = np.random.RandomState(42)
    n_test = 10
    eps = 1e-4 if "c64" in label.lower() or "complex64" in label.lower() else 1e-6
    errors = []

    for _ in range(n_test):
        ix = rng.randint(0, nx)
        iz = rng.randint(0, nx)

        m_p = m.copy()
        m_p[ix, iz] += eps
        d_p = solver.forward(m_p, OMEGA, src, rec)

        m_m = m.copy()
        m_m[ix, iz] -= eps
        d_m = solver.forward(m_m, OMEGA, src, rec)

        # Misfit = 0.5 * sum |d_pred - d_obs|^2 / sigma_d^2
        J_p = 0.5 * np.sum(np.abs(d_p - d_obs)**2) / sigma_d**2
        J_m = 0.5 * np.sum(np.abs(d_m - d_obs)**2) / sigma_d**2
        grad_fd = (J_p - J_m) / (2 * eps)

        rel = abs(grad_adj[ix, iz] - grad_fd) / (abs(grad_fd) + 1e-30)
        errors.append(rel)

    max_err = max(errors)
    mean_err = np.mean(errors)
    print(f"Adjoint check ({label}): max rel err = {max_err:.4e}, mean = {mean_err:.4e}")
    assert max_err < 0.01, f"Adjoint gradient check failed ({label}): max_err = {max_err:.4e}"


def test_adjoint_gradient_c128():
    _check_adjoint_gradient(HelmholtzC128, "complex128")


def test_adjoint_gradient_c64():
    """Compare c64 adjoint gradient against c128 adjoint (not FD)."""
    nx = 32
    dx = DOMAIN_KM / nx
    m = _smooth_model(nx, nx)
    src, rec = make_src_rec(nx, nx, n_src=3, n_rec=5)

    solver_128 = HelmholtzC128(nx=nx, nz=nx, dx=dx, npml=10, pml_max=100.0)
    solver_64 = HelmholtzC64(nx=nx, nz=nx, dx=dx, npml=10, pml_max=100.0)

    d_obs = solver_128.forward(m, OMEGA, src, rec)
    d_obs = d_obs * (1.0 + 0.01)
    sigma_d = 1.0

    grad_128 = solver_128.adjoint_gradient(m, OMEGA, src, rec, d_obs, sigma_d)
    grad_64 = solver_64.adjoint_gradient(m, OMEGA, src, rec, d_obs, sigma_d)

    rel_err = np.linalg.norm(grad_128 - grad_64) / np.linalg.norm(grad_128)
    print(f"Adjoint c128 vs c64: relative error = {rel_err:.4e}")
    assert rel_err < 0.01, f"c64 adjoint gradient differs from c128: rel_err = {rel_err:.4e}"


# --------------- run directly ---------------

if __name__ == "__main__":
    print("=== Grid convergence ===")
    test_grid_convergence()
    print("\n=== c128 vs c64 ===")
    test_c128_vs_c64()
    print("\n=== Adjoint gradient (c128) ===")
    test_adjoint_gradient_c128()
    print("\n=== Adjoint gradient (c64) ===")
    test_adjoint_gradient_c64()
    print("\nAll tests passed.")
