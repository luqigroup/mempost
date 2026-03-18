"""2D Helmholtz equation solver with PML — complex64 (single precision) variant.

Identical to helmholtz.py but uses complex64/float32 throughout for speed
comparison.  For a 64x64+PML grid this should be fine numerically.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class HelmholtzSolver:
    """2D Helmholtz solver with PML on a regular grid.

    The computational grid has size (nx + 2*npml) x (nz + 2*npml).
    Interior grid is (nx, nz) and represents the physical domain.

    On first call to forward() or adjoint_gradient(), the PML Laplacian
    is assembled and cached.  Subsequent calls only update the mass matrix
    (diagonal), which is O(N) instead of rebuilding the full sparse matrix.

    Args:
        nx: Number of interior grid points in x.
        nz: Number of interior grid points in z.
        dx: Grid spacing in km.
        npml: Number of PML grid points on each side.
        pml_max: Maximum PML damping coefficient (1/s).
    """

    def __init__(self, nx=64, nz=64, dx=0.03125, npml=10, pml_max=100.0):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.npml = npml
        self.pml_max = pml_max

        # Total grid including PML.
        self.nx_total = nx + 2 * npml
        self.nz_total = nz + 2 * npml
        self.N = self.nx_total * self.nz_total

        # Cached Laplacian and LU factorization.
        self._laplacian = None
        self._cached_omega = None

    def _pml_profile(self, n_total, npml, omega):
        """Compute complex PML stretch factors."""
        sigma = np.zeros(n_total)
        for i in range(npml):
            d = (npml - i) / npml
            sigma[i] = self.pml_max * d**2
            sigma[n_total - 1 - i] = self.pml_max * d**2
        sx = (1.0 + 1j * sigma / omega).astype(np.complex64)
        sx_half = (0.5 * (sx[:-1] + sx[1:])).astype(np.complex64)
        return sx, sx_half

    def _build_laplacian(self, omega):
        """Build the PML-modified Laplacian (model-independent part).

        Returns sparse matrix L such that A(m) = omega^2 diag(m) + L.
        """
        nx_t = self.nx_total
        nz_t = self.nz_total
        dx = self.dx

        sx, sx_half = self._pml_profile(nx_t, self.npml, omega)
        sz, sz_half = self._pml_profile(nz_t, self.npml, omega)

        # Precompute PML coefficients for x-direction.
        # c_xp[i] = 1/(sx[i] * sx_half[i]) for i < nx_t-1
        # c_xm[i] = 1/(sx[i] * sx_half[i-1]) for i > 0
        c_xp = np.zeros(nx_t, dtype=np.complex64)
        c_xm = np.zeros(nx_t, dtype=np.complex64)
        for i in range(nx_t - 1):
            c_xp[i] = 1.0 / (sx[i] * sx_half[i])
        for i in range(1, nx_t):
            c_xm[i] = 1.0 / (sx[i] * sx_half[i - 1])

        c_zp = np.zeros(nz_t, dtype=np.complex64)
        c_zm = np.zeros(nz_t, dtype=np.complex64)
        for j in range(nz_t - 1):
            c_zp[j] = 1.0 / (sz[j] * sz_half[j])
        for j in range(1, nz_t):
            c_zm[j] = 1.0 / (sz[j] * sz_half[j - 1])

        # Build sparse Laplacian using vectorized COO construction.
        # For each grid point (i, j) with linear index k = i*nz_t + j:
        #   diagonal: -(c_xp[i] + c_xm[i] + c_zp[j] + c_zm[j]) / dx^2
        #   right (i+1): c_xp[i] / dx^2
        #   left  (i-1): c_xm[i] / dx^2
        #   down  (j+1): c_zp[j] / dx^2
        #   up    (j-1): c_zm[j] / dx^2

        idx = np.arange(self.N)
        i_arr = idx // nz_t
        j_arr = idx % nz_t

        dx2 = dx**2
        diag_vals = -(c_xp[i_arr] + c_xm[i_arr] + c_zp[j_arr] + c_zm[j_arr]) / dx2

        rows_list = [idx]
        cols_list = [idx]
        vals_list = [diag_vals]

        # Right neighbor (i+1): exists when i < nx_t-1.
        mask_r = i_arr < nx_t - 1
        idx_r = idx[mask_r]
        rows_list.append(idx_r)
        cols_list.append(idx_r + nz_t)
        vals_list.append(c_xp[i_arr[mask_r]] / dx2)

        # Left neighbor (i-1): exists when i > 0.
        mask_l = i_arr > 0
        idx_l = idx[mask_l]
        rows_list.append(idx_l)
        cols_list.append(idx_l - nz_t)
        vals_list.append(c_xm[i_arr[mask_l]] / dx2)

        # Down neighbor (j+1): exists when j < nz_t-1.
        mask_d = j_arr < nz_t - 1
        idx_d = idx[mask_d]
        rows_list.append(idx_d)
        cols_list.append(idx_d + 1)
        vals_list.append(c_zp[j_arr[mask_d]] / dx2)

        # Up neighbor (j-1): exists when j > 0.
        mask_u = j_arr > 0
        idx_u = idx[mask_u]
        rows_list.append(idx_u)
        cols_list.append(idx_u - 1)
        vals_list.append(c_zm[j_arr[mask_u]] / dx2)

        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        all_vals = np.concatenate(vals_list)

        L = sp.coo_matrix(
            (all_vals, (all_rows, all_cols)), shape=(self.N, self.N)
        ).tocsc()
        return L

    def _get_laplacian(self, omega):
        """Get (possibly cached) PML Laplacian for the given omega."""
        if self._laplacian is None or self._cached_omega != omega:
            self._laplacian = self._build_laplacian(omega)
            self._cached_omega = omega
        return self._laplacian

    def _build_system_matrix(self, m_padded, omega):
        """Build A = omega^2 diag(m) + L_PML."""
        L = self._get_laplacian(omega)
        mass_vals = (np.complex64(omega**2) * m_padded.ravel()).astype(np.complex64)
        mass = sp.diags(mass_vals, format="csc")
        return L + mass

    def _pad_model(self, m):
        """Pad interior squared-slowness field with edge values for PML."""
        npml = self.npml
        m = np.asarray(m, dtype=np.float32)
        m_padded = np.zeros((self.nx_total, self.nz_total), dtype=np.complex64)
        m_padded[npml:npml + self.nx, npml:npml + self.nz] = m
        m_padded[:npml, npml:npml + self.nz] = m[0:1, :]
        m_padded[npml + self.nx:, npml:npml + self.nz] = m[-1:, :]
        m_padded[:, :npml] = m_padded[:, npml:npml + 1]
        m_padded[:, npml + self.nz:] = m_padded[:, npml + self.nz - 1:npml + self.nz]
        return m_padded

    def _source_vector(self, src_pos):
        """Create source vector for a point source at (ix, iz) in interior coords."""
        b = np.zeros(self.N, dtype=np.complex64)
        ix, iz = src_pos
        ix_pad = ix + self.npml
        iz_pad = iz + self.npml
        k = ix_pad * self.nz_total + iz_pad
        b[k] = 1.0 / self.dx**2
        return b

    def _receiver_projection(self, rec_positions):
        """Build sparse receiver projection matrix P: (n_rec, N)."""
        n_rec = len(rec_positions)
        rows = np.arange(n_rec)
        cols = np.array([
            (ix + self.npml) * self.nz_total + (iz + self.npml)
            for ix, iz in rec_positions
        ])
        vals = np.ones(n_rec, dtype=np.float32)
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_rec, self.N))

    def forward(self, m, omega, src_positions, rec_positions):
        """Forward model: squared-slowness -> data at receivers.

        Args:
            m: Squared-slowness on interior grid (nx, nz).
            omega: Angular frequency (rad/s).
            src_positions: List of (ix, iz) tuples for sources.
            rec_positions: List of (ix, iz) tuples for receivers.

        Returns:
            data: Complex data array (n_src, n_rec).
        """
        m_padded = self._pad_model(m)
        A = self._build_system_matrix(m_padded, omega)
        P = self._receiver_projection(rec_positions)

        # LU factorize once, solve for all sources.
        lu = spla.splu(A)
        n_src = len(src_positions)
        n_rec = len(rec_positions)
        data = np.zeros((n_src, n_rec), dtype=np.complex64)

        for s, src in enumerate(src_positions):
            b = self._source_vector(src)
            u = lu.solve(b)
            data[s] = P @ u

        return data

    def adjoint_gradient(self, m, omega, src_positions, rec_positions,
                         d_obs, sigma_d):
        """Compute gradient of data misfit w.r.t. squared-slowness via adjoint.

        Misfit: J(m) = 0.5 sum_s ||P u_s - d_obs[s]||^2 / sigma_d^2

        Uses LU factorization: one for forward solves, one for adjoint.
        A is complex symmetric so A^H = conj(A).

        Args:
            m: Squared-slowness on interior grid (nx, nz).
            omega: Angular frequency.
            src_positions: Source positions.
            rec_positions: Receiver positions.
            d_obs: Observed data (n_src, n_rec), complex.
            sigma_d: Data noise standard deviation.

        Returns:
            grad: Gradient on interior grid (nx, nz).
        """
        m_padded = self._pad_model(m)
        A = self._build_system_matrix(m_padded, omega)
        # Adjoint gradient needs full precision — upcast system matrix.
        A_128 = A.astype(np.complex128)
        P = self._receiver_projection(rec_positions)

        lu = spla.splu(A_128)
        lu_adj = spla.splu(A_128.conj())

        npml = self.npml
        grad_full = np.zeros(self.N, dtype=np.float64)

        for s, src in enumerate(src_positions):
            b = self._source_vector(src).astype(np.complex128)
            u = lu.solve(b)

            d_pred = P @ u
            r = d_pred - np.asarray(d_obs[s], dtype=np.complex128)

            rhs = P.T @ r / sigma_d**2
            lam = lu_adj.solve(rhs)

            grad_full += -omega**2 * np.real(u * np.conj(lam))

        grad_2d = grad_full.reshape(self.nx_total, self.nz_total)
        grad_interior = grad_2d[npml:npml + self.nx, npml:npml + self.nz]
        return np.real(grad_interior)

    def gradient_kl(self, z, kl_prior, omega, src_positions, rec_positions,
                    d_obs, sigma_d):
        """Compute gradient of data misfit w.r.t. KL coefficients.

        grad_z = G^T @ grad_m.ravel()
        """
        m = kl_prior.reconstruct(z)
        grad_m = self.adjoint_gradient(
            m, omega, src_positions, rec_positions, d_obs, sigma_d
        )
        G = kl_prior.kl_basis_matrix()
        grad_z = G.T @ grad_m.ravel()
        return grad_z.astype(np.float32)


def make_src_rec(nx, nz, n_src=5, n_rec=40, pad=0):
    """Create source and receiver positions for transmission geometry.

    Sources: evenly spaced along top (iz=pad).
    Receivers: evenly spaced along bottom (iz=nz-1-pad).

    Args:
        nx, nz: Grid dimensions.
        n_src: Number of sources.
        n_rec: Number of receivers.
        pad: Grid-point offset from domain edges (both x and z).
             E.g. pad=2 on a 201-grid gives x range [2, 198], matching
             the Gaussian lens JUDI example (20m offset on a 10m grid).
    """
    x_lo = pad
    x_hi = nx - 1 - pad

    src_ix = np.linspace(x_lo, x_hi, n_src, dtype=int)
    src_positions = [(int(ix), pad) for ix in src_ix]

    rec_ix = np.linspace(x_lo, x_hi, n_rec, dtype=int)
    rec_positions = [(int(ix), nz - 1 - pad) for ix in rec_ix]

    return src_positions, rec_positions
