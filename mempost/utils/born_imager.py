r"""Corrected zero-pad linearized-Born imager for the sparse rtm regeneration.

This vendors the ``PNF4seicmicImaging_gatech`` seismic pipeline
(``creating_dataset/setup_experiment.py`` + ``devito_wrapper.py``) for ONE job:
regenerate the migrated image :math:`\boldsymbol r = \boldsymbol J^{\!\top}
\boldsymbol d_{\rm obs}` of every genparihaka reflectivity patch under a sparser
acquisition, with the *correct* adjoint, building the Devito operator ONCE and
reusing it across all patches of a worker.

Math -> file map
----------------
* :class:`_ForwardBornZeroPad` -- the linearized-Born forward
  :math:`\boldsymbol J\,\mathrm{d}m` and its adjoint (migration)
  :math:`\boldsymbol J^{\!\top}\boldsymbol r`, as a ``torch.autograd.Function``.
  **forward**: ``solver.jacobian(pad(dm)[0, 0], src=src)[0].data``.
  **backward**: ``rec.data = grad_output``; ``u0 = solver.forward(src, save=True)
  [1]``; ``g = solver.jacobian_adjoint(rec, u0)[0].data[nb:-nb, nb:-nb]``.
* :class:`SparseBornImager` -- builds the Devito ``Model`` + acquisition geometry
  + Ricker wavelet ONCE for a fixed background ``m0`` and a sparse
  (``nsrc``, ``nrec``) geometry, then exposes :meth:`forward_obs` (one shot's
  ``d_obs = J dm + colored_noise``) and :meth:`migrate_dobs` (the
  source-averaged migration that *is* the regenerated ``rtm``).

THE ZERO-PAD ADJOINT (the whole reason this file exists)
--------------------------------------------------------
The PNF ``devito_wrapper.ForwardBorn`` pads the reflectivity with
``torch.nn.ReplicationPad2d(nbl)`` in the forward and crops ``g[nb:-nb, nb:-nb]``
in the backward. Cropping is the exact adjoint of ZERO-padding (a 0/1 selection
matrix and its transpose), but it is **not** the adjoint of replication-padding
(which fabricates a ``nbl``-cell boundary-reflectivity slab). So the PNF operator
is not a transpose pair: its dot-product adjoint test certifies ``~0.18``, not
``1.0``. The fix -- proven in this project's log "THE ADJOINT SAGA" and in
:mod:`pfvinf.forward.born` -- is to pad with ``torch.nn.ZeroPad2d(nbl)``;
everything else is byte-identical to PNF. Zero-pad is also physically correct
(reflectivity ``~0`` in the absorbing layer). With zero-pad
:meth:`SparseBornImager.adjoint_dot_ratio` returns ``~0.99999``.

:func:`_pad_layer` selects the pad at construction so the SAME class can build
the corrected (``"zero"``) operator used for generation and the buggy
(``"replication"``) operator the contract test uses to *prove* the distinction.

The rtm convention (mathematically identical to the PNF imaging loop)
---------------------------------------------------------------------
The PNF ``imaging.Imaging.rtm`` runs, per source :math:`s`,

.. math::
    d_{\rm pred} = J_s(\mathrm{d}m_0),\quad
    d_{\rm obs}  = J_s(\mathrm{d}m) + \varepsilon_s,\quad
    g_s = \nabla_{\mathrm{d}m_0}\|d_{\rm pred} - d_{\rm obs}\|^2,

and accumulates the running mean ``dm_est = (dm_est * s - g_s) / (s + 1)``. With
the background reflectivity ``dm0 = 0`` the demigration is exactly zero,
:math:`J_s(\boldsymbol 0) = \boldsymbol 0`, so
:math:`g_s = 2 J_s^{\!\top}(J_s(\boldsymbol 0) - d_{\rm obs}) = -2 J_s^{\!\top}
d_{\rm obs}` and ``-g_s = 2 J_s^T d_obs``. The running mean over the ``nsrc``
sources is therefore

.. math::
    \mathrm{rtm} = \frac{1}{n_{\rm src}}\sum_{s} (-g_s)
                 = \frac{2}{n_{\rm src}}\sum_{s} J_s^{\!\top} d_{{\rm obs},s},

i.e. the *source-averaged migration* of the noisy data, with the SAME factor of
two and the SAME ``1/nsrc`` average the PNF loop carries (so the regenerated
``rtm`` lives on exactly the original ``rtm`` scale). :meth:`migrate_dobs`
computes this directly with two solves per source (one forward ``J(dm)`` to make
``d_obs`` and one adjoint ``J^T d_obs``), skipping the wasteful third
``J(dm0) = 0`` solve of the PNF loop. The final image is passed through the same
``mute_op(end=mute_end, length=mute_length)`` the PNF loop applies at save time,
zeroing the shallow water column so the migrated image matches the original.

Determinism
-----------
The colored observation noise (:meth:`sample_noise`) is the PNF
wavelet-convolved Gaussian, but drawn from a per-call ``numpy.random.Generator``
seeded explicitly (``noise_seed + patch_idx``) rather than the PNF global
``numpy.random`` state, so a resumed / re-run patch reproduces its record
bit-for-bit. Devito's threaded C kernels are not bit-reproducible across thread
counts; the generation driver pins each worker to ``OMP_NUM_THREADS=1`` before
importing Devito.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal

from examples.seismic import AcquisitionGeometry, Model, RickerSource, TimeAxis
from examples.seismic.acoustic import AcousticWaveSolver
from devito import configuration

configuration["log-level"] = "WARNING"

_PAD_MODES = ("zero", "replication")


def _pad_layer(pad_mode: str, nbl: int) -> torch.nn.Module:
    """Return the boundary pad module for ``pad_mode`` (``"zero"``/``"replication"``).

    ``"zero"`` is the CORRECT choice: cropping ``g[nbl:-nbl, nbl:-nbl]`` in the
    adjoint is the exact transpose of zero-padding, making the migration the
    discrete adjoint of the demigration (dot-product ratio ``~1``).
    ``"replication"`` reproduces the buggy PNF operator (ratio ``~0.18``) and
    exists ONLY so the contract test can prove the distinction.
    """
    if pad_mode == "zero":
        return torch.nn.ZeroPad2d(int(nbl))
    if pad_mode == "replication":
        return torch.nn.ReplicationPad2d(int(nbl))
    raise ValueError(
        f"pad_mode must be one of {_PAD_MODES!r}; got {pad_mode!r}",
    )


class _ForwardBornZeroPad(torch.autograd.Function):
    """Linearized Born ``J dm`` with a configurable boundary pad (PNF-faithful).

    forward: pad the reflectivity into the ``nbl``-cell absorbing layer (zero-pad
    for the correct adjoint), then one Born/demigration solve
    ``solver.jacobian(padded[0, 0], src=src)``. backward: write the upstream data
    residual into the geometry receiver, recompute the background wavefield
    ``u0 = solver.forward(src, save=True)[1]``, migrate
    ``solver.jacobian_adjoint(rec, u0)``, and crop the boundary layer
    ``g[nb:-nb, nb:-nb]`` (the exact adjoint of the zero-pad).

    The pad module is passed in (``pad``) so the SAME autograd function serves the
    corrected zero-pad operator and the buggy replication-pad operator the test
    uses to certify the difference. ``src`` is the shared, per-source-positioned
    ``RickerSource``; the caller repositions it before calling and does not move
    it between this forward and its paired backward (the regeneration migrates
    each source immediately after demigrating it, so the position is stable --
    mirroring the PNF per-source loop).
    """

    @staticmethod
    def forward(ctx, dm, pad, src, model, geometry, solver, device):
        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device
        ctx.src = src
        # Pad the reflectivity into the absorbing layer. Zero-pad is the exact
        # adjoint of the backward crop and is physically correct (reflectivity
        # ~ 0 in the damping layer); replication-pad (the PNF default) fabricates
        # a boundary slab and breaks the transpose -- the whole bug being fixed.
        padded = pad(dm).detach().cpu().numpy()
        d_lin = solver.jacobian(padded[0, 0, :, :], src=src)[0].data
        return torch.from_numpy(np.asarray(d_lin)).to(device)

    @staticmethod
    def backward(ctx, grad_output):
        # Write the upstream residual into the geometry receiver, recompute the
        # background wavefield for THIS source, and migrate; crop the boundary
        # layer (the adjoint of the forward's zero-pad).
        rec = ctx.geometry.rec
        rec.data[:] = grad_output.detach().cpu().numpy()[:]
        u0 = ctx.solver.forward(src=ctx.src, save=True)[1]
        g = ctx.solver.jacobian_adjoint(rec, u0)[0].data
        nb = ctx.model.nbl
        g = torch.from_numpy(np.asarray(g[nb:-nb, nb:-nb])).to(ctx.device)
        return (g.view(1, 1, g.shape[0], g.shape[1]),
                None, None, None, None, None, None)


class SparseBornImager:
    r"""Build-once linearized-Born imager over a fixed background (sparse geometry).

    Reproduces the PNF ``SetupExperiment`` geometry (Devito ``Model`` +
    ``AcquisitionGeometry`` + Ricker source + receiver line + wavelet-colored
    noise + shallow mute) for a fixed background squared-slowness ``m0`` and a
    sparse ``(nsrc, nrec)`` acquisition, EXCEPT the boundary pad is selectable
    (``pad_mode="zero"`` fixes the PNF adjoint bug). The Devito ``Model`` /
    geometry / solver / wavelet depend only on the grid (``m0`` shape, spacing,
    ``space_order``, ``nbl``) and the acquisition -- all identical across patches
    since ``m0`` is constant -- so they are built ONCE here and reused for every
    patch of a worker; only the reflectivity ``dm`` is a runtime input to
    ``solver.jacobian`` and the per-source source x-coordinate is runtime data
    (no recompile).

    Args:
        m0: ``(nlat, ndep)`` background squared slowness (s^2/km^2); the
            background velocity is ``1/sqrt(m0)`` km/s. Orientation
            ``(lateral, depth)`` -- depth is axis 1, matching the genparihaka
            patches and the PNF model.
        spacing: ``(spacing_x, spacing_z)`` grid spacing in meters.
        nsrc: number of sources on the shallow line (sparse: 12).
        nrec: number of receivers on the shallow line (sparse: 24).
        tn: record length in milliseconds.
        f0: Ricker peak frequency (kHz; 0.030 = 30 Hz).
        space_order: spatial finite-difference order (PNF: 16).
        nbl: Devito damping-boundary width (PNF: 40).
        pad_mode: ``"zero"`` (correct adjoint; generation) or ``"replication"``
            (buggy PNF; test only).
        src_depth_cells, rec_depth_cells: source/receiver depth in grid cells
            (PNF: ``2 * spacing[1]`` -> 2 cells).
        device: torch device for returned tensors (``"cpu"``).
    """

    def __init__(
        self,
        m0: np.ndarray,
        spacing: tuple[float, float],
        *,
        nsrc: int,
        nrec: int,
        tn: float,
        f0: float,
        space_order: int = 16,
        nbl: int = 40,
        pad_mode: str = "zero",
        src_depth_cells: float = 2.0,
        rec_depth_cells: float = 2.0,
        device: str = "cpu",
    ) -> None:
        m0 = np.ascontiguousarray(m0, dtype=np.float32)
        if m0.ndim != 2:
            raise ValueError(f"m0 must be 2-D (nlat, ndep); got {m0.shape!r}")
        if int(nsrc) < 1 or int(nrec) < 1:
            raise ValueError(
                f"nsrc and nrec must be >= 1; got {nsrc!r}, {nrec!r}",
            )
        if pad_mode not in _PAD_MODES:
            raise ValueError(
                f"pad_mode must be one of {_PAD_MODES!r}; got {pad_mode!r}",
            )
        self.nlat, self.ndep = int(m0.shape[0]), int(m0.shape[1])
        self.spacing = (float(spacing[0]), float(spacing[1]))
        self.nsrc = int(nsrc)
        self.nrec = int(nrec)
        self.tn = float(tn)
        self.f0 = float(f0)
        self.space_order = int(space_order)
        self.nbl = int(nbl)
        self.pad_mode = str(pad_mode)
        self.device = torch.device(device)
        self._pad = _pad_layer(self.pad_mode, self.nbl)

        # Devito model: vp = 1/sqrt(m0) (km/s). space_order/nbl/bcs mirror PNF.
        self.model = Model(
            space_order=self.space_order, vp=1.0 / np.sqrt(m0),
            origin=(0.0, 0.0), shape=(self.nlat, self.ndep), dtype=np.float32,
            spacing=self.spacing, nbl=self.nbl, bcs="damp",
        )
        dt = float(self.model.critical_dt)
        time_range = TimeAxis(start=0.0, stop=self.tn, step=dt)
        self.nt = int(time_range.num)

        # Receivers on a shallow line spanning the full width (PNF: 2 cells deep).
        rec_coordinates = np.empty((self.nrec, 2), dtype=np.float32)
        rec_coordinates[:, 0] = np.linspace(
            0.0, self.model.domain_size[0], num=self.nrec,
        )
        rec_coordinates[:, 1] = rec_depth_cells * self.spacing[1]

        # Single Ricker source (one physical source at a time, like PNF's
        # non-sim-source path); repositioned per shot via create_op.
        self._src = RickerSource(
            name="src", grid=self.model.grid, f0=self.f0,
            time_range=time_range, npoint=1,
        )
        self.wavelet = np.array(self._src.data[:, 0:1])
        self._src.coordinates.data[:, 0] = 0.5 * self.model.domain_size[0]
        self._src.coordinates.data[:, -1] = src_depth_cells * self.spacing[1]
        # Per-shot lateral source positions span the full width (PNF: linspace
        # 0 .. domain_size over nsrc; src_idx picks one).
        self._src_x = np.linspace(0.0, self.model.domain_size[0], num=self.nsrc)

        self.geometry = AcquisitionGeometry(
            self.model, rec_coordinates, self._src.coordinates.data,
            t0=0.0, tn=self.tn, src_type="Ricker", f0=self.f0,
        )
        self.solver = AcousticWaveSolver(
            self.model, self.geometry, space_order=self.space_order,
        )

    # -- shape helpers ------------------------------------------------------ #

    @property
    def record_shape(self) -> tuple[int, int]:
        """Single-shot record shape ``(nt, nrec)``."""
        return (self.nt, self.nrec)

    def _as_1x1(self, dm: torch.Tensor) -> torch.Tensor:
        """Coerce a reflectivity patch to ``(1, 1, nlat, ndep)`` float32."""
        if dm.shape[-2:] != (self.nlat, self.ndep):
            raise ValueError(
                f"dm trailing dims must be ({self.nlat}, {self.ndep}); "
                f"got {tuple(dm.shape[-2:])}",
            )
        return dm.reshape(1, 1, self.nlat, self.ndep).to(torch.float32)

    # -- PNF utilities (faithful) ------------------------------------------ #

    def mute_op(
        self, dm: torch.Tensor, end: int = 10, length: int = 1,
    ) -> torch.Tensor:
        r"""Shallow taper that zeros the water column (PNF ``mute_op``).

        Multiplies the depth axis (axis ``-1`` of the ``(..., nlat, ndep)``
        image) by a mask that is ``0`` above ``start = end - length``, a raised
        half-sine ramp on ``[start, end)``, and ``1`` from ``end`` on. Applied to
        the migrated image exactly as PNF applies it at save time so the
        regenerated ``rtm`` matches the original water-zeroed convention.
        """
        start = end - length
        damp = torch.zeros(dm.shape[-2], dm.shape[-1])
        damp[..., end:] = 1.0
        damp[..., start:end] = (
            1.0 + torch.sin((np.pi / 2.0 * torch.arange(0, length)) / length)
        ) / 2.0
        return damp.to(dm) * dm

    def sample_noise(
        self, sigma: float, rng: np.random.Generator,
    ) -> np.ndarray:
        r"""Wavelet-colored Gaussian record noise (PNF ``sample_noise``, seeded).

        Draws ``e ~ sigma * N(0, I)`` of shape ``(nt, nrec)``, convolves it along
        time with the first 50 samples of the source wavelet
        (``signal.fftconvolve(..., mode="same")``), then renormalizes so the
        colored field has RMS ``sigma`` (``e *= sqrt(prod(shape)) * sigma /
        ||e||``). Identical to PNF except the draw comes from the supplied
        ``numpy.random.Generator`` (seeded ``noise_seed + patch_idx``) instead of
        the PNF global ``numpy.random`` state, so a resumed patch reproduces its
        record. Returns a ``(nt, nrec)`` float array.
        """
        shape = (self.nt, self.nrec)
        kernel = self.wavelet[:50].reshape(-1, 1)
        e = sigma * rng.standard_normal(shape)
        e = signal.fftconvolve(e, kernel, mode="same")
        e *= 1.0 / np.linalg.norm(e)
        e *= np.sqrt(np.prod(e.shape)) * sigma
        return e

    # -- the forward / adjoint matvecs -------------------------------------- #

    def _born_one(self, dm_1x1: torch.Tensor, src_idx: int) -> torch.Tensor:
        """One source's record ``J_s dm`` (differentiable), shape ``(nt, nrec)``.

        Repositions the shared source to ``src_idx`` (runtime data; no recompile)
        and applies the configurable-pad Born forward.
        """
        self._src.coordinates.data[:, 0] = self._src_x[src_idx]
        return _ForwardBornZeroPad.apply(
            dm_1x1, self._pad, self._src, self.model, self.geometry,
            self.solver, self.device,
        )

    def forward_obs(
        self, dm: torch.Tensor, src_idx: int, sigma: float,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        r"""One shot's observed record ``d_obs = J_s dm + colored_noise``.

        Computes the (differentiable) Born record for source ``src_idx`` and adds
        the seeded wavelet-colored noise (:meth:`sample_noise`). Returns a
        ``(nt, nrec)`` float32 tensor on :attr:`device`. The migration in
        :meth:`migrate_dobs` differentiates the SAME forward to obtain
        ``J_s^T d_obs``; the noise is a constant added after the differentiable
        record, so it does not enter the Jacobian.
        """
        x = self._as_1x1(dm)
        d_lin = self._born_one(x, src_idx)
        noise = torch.from_numpy(
            self.sample_noise(sigma, rng),
        ).to(d_lin.dtype).to(self.device)
        return d_lin + noise

    def _migrate_dobs_core(
        self, dm: torch.Tensor, sigma: float, rng: np.random.Generator,
        *, mute_end: int, mute_length: int, collect_raw: bool,
    ) -> tuple[torch.Tensor, np.ndarray | None]:
        r"""Source loop shared by :meth:`migrate_dobs` / :meth:`migrate_dobs_raw`.

        For each of the ``nsrc`` sources: reposition the shared source, build the
        differentiable Born record once, add the seeded colored noise to form
        ``d_obs_s``, optionally retain ``d_obs_s`` (the raw shot gather), then
        migrate ``J_s^T d_obs_s`` by autograd of the SAME record
        (``grad_{x} <J x, d_obs> = J^T d_obs``; the noise is detached so it does
        not enter the Jacobian). A fresh leaf clone of ``dm`` per source keeps the
        graph fresh so each adjoint migrates THIS source's residual. Accumulate
        ``(2 / nsrc) * sum_s J_s^T d_obs_s`` and apply the shallow
        :meth:`mute_op` -- the factor of two, the ``1/nsrc`` average and the mute
        reproduce the PNF imaging loop at ``dm0 = 0`` (where ``J(0) = 0``), so the
        image is on the original ``rtm`` scale. Returns ``(rtm, raw_or_None)``
        with ``rtm`` a ``(nlat, ndep)`` float32 CPU tensor and ``raw`` a
        ``(nsrc, nt, nrec)`` float32 array when ``collect_raw`` else ``None``.

        The ``rng`` is consumed once per source IN SOURCE ORDER, so a re-run with
        the same ``noise_seed + patch_idx`` reproduces both the image and the raw
        gathers bit-for-bit (under single-threaded Devito).
        """
        x = self._as_1x1(dm)
        acc = torch.zeros(self.nlat, self.ndep, dtype=torch.float64)
        raw = (np.empty((self.nsrc, self.nt, self.nrec), dtype=np.float32)
               if collect_raw else None)
        for s in range(self.nsrc):
            x_leaf = x.clone().requires_grad_(True)
            # PNF's effective forward is J(x) = forward_op(mute_op(x)) (mute the
            # reflectivity BEFORE the Born forward, setup_experiment.py:123), so
            # the modeled data uses M.dm and the migration backprops through the
            # (self-adjoint, diagonal) mute to give J^T d_obs = M(B^T d_obs). The
            # final-image mute below is kept (PNF also mutes at save time). NO-OP
            # on this dataset -- every dm already has its top mute_end depth cols
            # zero, so M.dm == dm -- but this makes J PNF-exact and robust if
            # reused on un-pre-muted reflectivity.
            x_muted = self.mute_op(
                x_leaf, end=int(mute_end), length=int(mute_length),
            )
            self._src.coordinates.data[:, 0] = self._src_x[s]
            d_lin = _ForwardBornZeroPad.apply(
                x_muted, self._pad, self._src, self.model, self.geometry,
                self.solver, self.device,
            )
            noise = torch.from_numpy(
                self.sample_noise(sigma, rng),
            ).to(d_lin.dtype).to(self.device)
            d_obs = (d_lin + noise).detach()
            if collect_raw:
                raw[s] = d_obs.cpu().numpy().astype(np.float32)
            # grad_{x} <J x, d_obs> = J^T d_obs (the migration of d_obs).
            jt_dobs = torch.autograd.grad(
                outputs=d_lin, inputs=x_leaf, grad_outputs=d_obs,
            )[0]
            acc += jt_dobs.reshape(self.nlat, self.ndep).to(torch.float64)
        rtm = (2.0 / self.nsrc) * acc
        rtm = self.mute_op(rtm, end=int(mute_end), length=int(mute_length))
        return rtm.to(torch.float32).cpu(), raw

    def migrate_dobs(
        self, dm: torch.Tensor, sigma: float, rng: np.random.Generator,
        *, mute_end: int = 10, mute_length: int = 1,
    ) -> torch.Tensor:
        r"""The regenerated ``rtm`` = source-averaged migration of ``d_obs``.

        Thin wrapper over :meth:`_migrate_dobs_core` that retains only the
        migrated image (two solves per source, versus PNF's three -- it also runs
        the redundant ``J(dm0) = 0`` forward; the result is identical because
        ``J(0) = 0`` exactly). Returns a ``(nlat, ndep)`` float32 CPU tensor.
        """
        rtm, _ = self._migrate_dobs_core(
            dm, sigma, rng, mute_end=mute_end, mute_length=mute_length,
            collect_raw=False,
        )
        return rtm

    def migrate_dobs_raw(
        self, dm: torch.Tensor, sigma: float, rng: np.random.Generator,
        *, mute_end: int = 10, mute_length: int = 1,
    ) -> tuple[torch.Tensor, np.ndarray]:
        r"""Like :meth:`migrate_dobs` but also returns the raw shot gathers.

        Computes the regenerated ``rtm`` AND the ``(nsrc, nt, nrec)`` raw
        observed records ``d_obs`` from the SAME seeded noise stream, so the
        saved raw gathers and the migrated image are mutually consistent for the
        patch. No extra solves beyond :meth:`migrate_dobs` (the records are
        already formed to migrate them). Returns ``(rtm, raw)``.
        """
        rtm, raw = self._migrate_dobs_core(
            dm, sigma, rng, mute_end=mute_end, mute_length=mute_length,
            collect_raw=True,
        )
        return rtm, raw

    def adjoint_dot_ratio(self, seed: int = 0) -> float:
        r"""Dot-product adjoint ratio ``<J x, y> / <x, J^T y>`` (the GATE).

        Draws a random reflectivity ``x`` and a random single-shot data field
        ``y``, forms ``d = J_0 x`` (forward) and ``g = J_0^T y`` (autograd
        adjoint) for source 0, and returns ``<d, y> / <x, g>``. For a true
        transpose pair this ratio is ``1``: zero-pad gives ``~0.99999`` (a true
        discrete adjoint). Replication-pad (the PNF bug) is NOT a transpose pair,
        so its ratio is far from ``1`` -- ``O(0.1)`` away or more, the exact value
        depending on the grid / ``nbl`` ratio and the acquisition (the historical
        ``~0.18`` was at a different geometry). The generation driver calls this
        with ``pad_mode="zero"`` as a hard gate before writing any patch, and the
        contract test calls it both ways to certify the distinction.
        """
        g_t = torch.Generator().manual_seed(int(seed))
        x = torch.randn(
            1, 1, self.nlat, self.ndep, generator=g_t, dtype=torch.float32,
        ).requires_grad_(True)
        y = torch.randn(
            self.nt, self.nrec, generator=g_t, dtype=torch.float32,
        )
        self._src.coordinates.data[:, 0] = self._src_x[0]
        d = _ForwardBornZeroPad.apply(
            x, self._pad, self._src, self.model, self.geometry,
            self.solver, self.device,
        )
        # Take the adjoint first (consumes the graph), then detach d for the
        # scalar dot readout (the dot itself carries no gradient meaning).
        jt_y = torch.autograd.grad(outputs=d, inputs=x, grad_outputs=y)[0]
        lhs = float(torch.dot(d.detach().reshape(-1), y.reshape(-1)))
        rhs = float(torch.dot(x.detach().reshape(-1), jt_y.reshape(-1)))
        return lhs / (rhs + 1e-30)


def parihaka_background_m0(
    nlat: int = 256, ndep: int = 256, water_zero_depth: int = 10,
) -> np.ndarray:
    r"""Fixed v(z)-gradient + water-layer background squared slowness (PNF ``run_imaging``).

    Reproduces the PNF ``run_imaging.py`` background EXACTLY (the SAME background
    for all patches): a constant ``2.5`` km/s velocity scaled by a linear depth
    gradient to ``4.5`` km/s at the bottom, with the top ``water_zero_depth``
    depth cells set to a ``1.5`` km/s water layer, returned as squared slowness
    ``m0 = (1/v)^2`` in ``(lateral, depth)`` orientation (depth is axis 1).

    .. code-block:: python

        v0 = ones((nlat, ndep)) * 2.5
        v0 *= linspace(1.0, 4.5 / v0.max(), ndep)   # along depth (axis 1)
        s = 1 / v0
        s[:, :water_zero_depth] = 1 / 1.5            # water layer
        m0 = s ** 2

    Args:
        nlat: lateral grid size (axis 0).
        ndep: depth grid size (axis 1).
        water_zero_depth: number of shallow depth cells set to the water layer.

    Returns:
        ``(nlat, ndep)`` float32 background squared slowness.
    """
    v0 = np.ones((int(nlat), int(ndep)), dtype=np.float32) * 2.5
    v0 *= np.linspace(1.0, 4.5 / v0.max(), int(ndep)).astype(np.float32)
    s = 1.0 / v0
    s[:, : int(water_zero_depth)] = 1.0 / 1.5
    return (s ** 2).astype(np.float32)
