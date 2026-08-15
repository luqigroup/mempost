# pylint: disable=E1102
# pylint: disable=invalid-name
"""Helmholtz FWI memorization experiment.

Trains an MLP-based DDPM on KL coefficients of velocity models, then performs
DPS posterior sampling through a nonlinear Helmholtz forward operator.  Shows
posterior collapse for small training sets (N = 50, 200, 1000).
"""
import os

import numpy as np
import torch
from copy import deepcopy
from projorg import checkpointsdir, plotsdir, setup_environment, upload_to_cloud
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mempost.models import NoiseScheduler

from grf_with_score import GaussianRandomField
from mempost.utils.helmholtz import HelmholtzSolver, make_src_rec
from mempost.utils.kl_prior import VelocityKLPrior
from mempost.utils.normalizer import Normalizer
from mempost.utils.memorization_metrics import (
    k_nearest_indices,
    nearest_neighbor_distances,
    ratio_mem_metric,
    ratio_mem_values,
)

import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1)
        emb = torch.exp(-emb * torch.arange(half, device=t.device))
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)


class ResBlock(nn.Module):
    """Residual block with LayerNorm (no BatchNorm)."""

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ScoreModel(nn.Module):
    """MLP score model for DDPM in R^d.

    Uses LayerNorm instead of BatchNorm. Output layer is linear (no
    activation) since we predict unbounded noise.
    """

    def __init__(self, input_size, hidden_dim=256, nlayers=4, emb_size=128,
                 dropout=0.1):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(emb_size)
        layers = [nn.Linear(input_size + emb_size, hidden_dim), nn.GELU()]
        for _ in range(nlayers):
            layers.append(ResBlock(hidden_dim, dropout=dropout))
        layers.append(nn.Linear(hidden_dim, input_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        return self.net(torch.cat((x, t_emb), dim=-1))


CONFIG_FILE = "helmholtz_fwi.json"
FIGS_DIR = os.path.join("figs", "helmholtz")


def make_gaussian_lens(grid_size=64, v_background=2.0,
                       v_anomaly=1.5, cx=0.5, cz=0.5,
                       sigma_x=0.25, sigma_z=0.15):
    """Create a Gaussian low-velocity lens true model.

    Defines the model directly on the grid (NOT projected to KL basis)
    so that the true model is not perfectly representable by the prior.

    Args:
        grid_size: Number of grid points per side.
        v_background: Background velocity (km/s).
        v_anomaly: Minimum velocity at the lens center (km/s).
        cx, cz: Lens center in normalized coordinates [0, 1].
        sigma_x, sigma_z: Lens width in normalized coordinates.

    Returns:
        m_true: Squared-slowness on (grid_size, grid_size).
        v_true: Velocity on (grid_size, grid_size).
    """
    x = np.linspace(0, 1, grid_size)
    z = np.linspace(0, 1, grid_size)
    X, Z = np.meshgrid(x, z, indexing="ij")

    gauss = np.exp(-((X - cx)**2 / (2 * sigma_x**2)
                     + (Z - cz)**2 / (2 * sigma_z**2)))
    v_true = v_background + (v_anomaly - v_background) * gauss
    m_true = 1.0 / v_true**2
    return m_true, v_true


class HelmholtzDDPM:
    """MLP-based DDPM in KL coefficient space for Helmholtz FWI.

    Trains a denoising diffusion model on KL expansion coefficients (R^d),
    then performs DPS posterior sampling through the nonlinear Helmholtz
    forward operator.
    """

    def __init__(self, args):
        if torch.cuda.is_available() and args.gpu_id > -1:
            self.device = torch.device("cuda:" + str(args.gpu_id))
        else:
            self.device = torch.device("cpu")

        # KL prior.
        self.kl_prior = VelocityKLPrior(
            K=args.kl_K,
            grid_size=args.grid_size,
            v_background=args.v_background,
            sigma_m=args.sigma_m,
        )
        self.d = self.kl_prior.d

        # Helmholtz solver.
        self.solver = HelmholtzSolver(
            nx=args.grid_size,
            nz=args.grid_size,
            dx=2.0 / args.grid_size,  # 2 km domain / grid_size
            npml=args.npml,
            pml_max=args.pml_max,
        )
        self.omega = 2.0 * np.pi * args.frequency
        self.src_positions, self.rec_positions = make_src_rec(
            args.grid_size, args.grid_size, args.n_src, args.n_rec,
        )

        # Precompute KL basis matrix for gradient projection.
        self.G = self.kl_prior.kl_basis_matrix()

        # GRF for sampling realistic velocity models.
        v_min = getattr(args, "v_min", 1.5)
        v_max = getattr(args, "v_max", 3.0)
        self.grf = GaussianRandomField(
            dim=2, size=args.grid_size,
            alpha=getattr(args, "grf_alpha", 3),
            tau=getattr(args, "grf_tau", 5),
            bounds=(1.0 / v_max**2, 1.0 / v_min**2),
        )

        # Generate training data: sample from GRF, project onto KL basis.
        np.random.seed(args.seed)
        m_train = self.grf.sample(args.num_train)
        z_train = self.kl_prior.project(m_train)
        self.z_train_raw = torch.tensor(z_train, dtype=torch.float32)

        # Validation set.
        np.random.seed(args.seed + 1)
        m_val = self.grf.sample(100)
        z_val = self.kl_prior.project(m_val)
        self.z_val_raw = torch.tensor(z_val, dtype=torch.float32)

        # Z-score normalization (per-dimension).
        self.normalizer = Normalizer(self.z_train_raw)
        self.z_train = self.normalizer.normalize(self.z_train_raw)
        self.z_val = self.normalizer.normalize(self.z_val_raw)

        print(f"Training set size: {self.z_train.shape[0]}, d = {self.d}")

        # Data loaders (normalized data).
        self.train_loader = DataLoader(
            TensorDataset(self.z_train),
            batch_size=args.batchsize,
            shuffle=True,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            TensorDataset(self.z_val),
            batch_size=2 * args.batchsize,
            shuffle=False,
            drop_last=False,
        )

        # MLP score model (LayerNorm, no BatchNorm, no dropout).
        self.score_model = ScoreModel(
            input_size=self.d,
            hidden_dim=args.hidden_dim,
            nlayers=args.nlayers,
            emb_size=args.emb_size,
            dropout=0.0,
        ).to(self.device)

        # EMA model.
        self.ema_model = deepcopy(self.score_model)
        self.ema_decay = 0.9999

        # Noise scheduler.
        self.noise_scheduler = NoiseScheduler(
            nt=args.nt,
            beta_schedule="linear",
            device=self.device,
        )

        # Optimizer with cosine decay.
        self.optimizer = torch.optim.Adam(
            self.score_model.parameters(), lr=args.lr,
        )
        lr_ratio = args.lr_final / args.lr
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.max_epochs, eta_min=args.lr_final,
        )

        self.train_obj = []
        self.val_obj = []

        n_params = sum(p.numel() for p in self.score_model.parameters())
        print(f"Score model parameters: {n_params:,}")

    def load_checkpoint(self, args, resume=False):
        file_to_load = os.path.join(
            checkpointsdir(args.experiment),
            "checkpoint_" + str(args.testing_epoch) + ".pth",
        )
        if os.path.isfile(file_to_load):
            checkpoint = torch.load(
                file_to_load, map_location=self.device, weights_only=False,
            )
            self.score_model.load_state_dict(checkpoint["model_state_dict"])
            if "ema_state_dict" in checkpoint:
                self.ema_model.load_state_dict(checkpoint["ema_state_dict"])
            if resume:
                self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
                if "lr_scheduler_state_dict" in checkpoint:
                    self.lr_scheduler.load_state_dict(
                        checkpoint["lr_scheduler_state_dict"]
                    )
            self.train_obj = checkpoint["train_obj"]
            self.val_obj = checkpoint["val_obj"]
            if not args.testing_epoch == checkpoint["epoch"]:
                raise ValueError("Inconsistent filename and loaded checkpoint.")
        else:
            raise ValueError(f"Checkpoint does not exist: {file_to_load}")

    def train(self, args, start_epoch=0):
        for epoch in tqdm(
            range(start_epoch, args.max_epochs),
            unit="epoch", colour="#B5F2A9", dynamic_ncols=True,
            desc="Training progress",
        ):
            # Validation.
            if epoch % 10 == 0:
                self.score_model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for (z_val,) in self.val_loader:
                        z_val = z_val.to(self.device)
                        noise = torch.randn_like(z_val)
                        t = torch.randint(
                            0, len(self.noise_scheduler),
                            (z_val.shape[0],), device=self.device,
                        ).long()
                        z_t = self.noise_scheduler.add_noise(z_val, noise, t)
                        noise_pred = self.score_model(z_t, t)
                        obj = torch.norm(noise_pred - noise)**2
                        val_loss += obj.item() / z_val.shape[0]
                    val_loss /= max(len(self.val_loader), 1)
                self.val_obj.append(val_loss)

            # Training.
            self.score_model.train()
            for (z_train,) in self.train_loader:
                z_train = z_train.to(self.device)
                noise = torch.randn_like(z_train)
                t = torch.randint(
                    0, len(self.noise_scheduler),
                    (z_train.shape[0],), device=self.device,
                ).long()
                z_t = self.noise_scheduler.add_noise(z_train, noise, t)
                noise_pred = self.score_model(z_t, t)
                obj = torch.norm(noise_pred - noise)**2 / z_train.shape[0]

                obj.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.score_model.parameters(), max_norm=1.0,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update EMA.
                with torch.no_grad():
                    for ema_p, p in zip(
                        self.ema_model.parameters(),
                        self.score_model.parameters(),
                    ):
                        ema_p.mul_(self.ema_decay).add_(
                            p.data, alpha=1 - self.ema_decay,
                        )

                self.train_obj.append(obj.item())

            self.lr_scheduler.step()

            # Save checkpoints.
            if epoch % args.save_freq == 0 or epoch == args.max_epochs - 1:
                torch.save(
                    {
                        "model_state_dict": self.score_model.state_dict(),
                        "ema_state_dict": self.ema_model.state_dict(),
                        "optim_state_dict": self.optimizer.state_dict(),
                        "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "train_obj": self.train_obj,
                        "val_obj": self.val_obj,
                        "normalizer_mean": self.normalizer.mean,
                        "normalizer_std": self.normalizer.std,
                    },
                    os.path.join(
                        checkpointsdir(args.experiment),
                        "checkpoint_" + str(epoch) + ".pth",
                    ),
                )

    @torch.no_grad()
    def _raw_sample(self, num_samples):
        """Generate KL coefficient samples via DDIM (deterministic).

        Uses DDIM update to avoid noise injection that compounds errors
        over many steps.  Returns unnormalized (physical) KL coefficients.
        """
        self.ema_model.eval()
        alpha_cumprod = self.noise_scheduler.alphas_cumprod
        z = torch.randn(num_samples, self.d, device=self.device)
        for t in reversed(range(self.noise_scheduler.nt)):
            t_batch = torch.full(
                (num_samples,), t, device=self.device, dtype=torch.long,
            )
            eps_pred = self.ema_model(z, t_batch)

            # Tweedie estimate of x0.
            a_t = alpha_cumprod[t]
            x0_hat = (z - (1 - a_t).sqrt() * eps_pred) / a_t.sqrt()

            if t > 0:
                a_prev = alpha_cumprod[t - 1]
                z = a_prev.sqrt() * x0_hat + (1 - a_prev).sqrt() * eps_pred
            else:
                z = x0_hat

        # Unnormalize to physical KL space.
        z = self.normalizer.unnormalize(z.cpu())
        return z

    def test(self, args, epoch=-1, n_gen=384, ratio_threshold=0.5):
        if epoch == -1:
            self.load_checkpoint(args)
            epoch = args.testing_epoch

        # Generate many samples in batches.
        batch_sz = 64
        all_gen = []
        for start in range(0, n_gen, batch_sz):
            n = min(batch_sz, n_gen - start)
            all_gen.append(self._raw_sample(n))
        generated = torch.cat(all_gen, dim=0)

        # Memorization metrics in raw z-space.
        gen_flat = generated.reshape(generated.shape[0], -1)
        train_flat = self.z_train_raw.reshape(self.z_train_raw.shape[0], -1)

        k = min(50, args.num_train)
        ratios = ratio_mem_values(train_flat, gen_flat, k_neighbors=k)
        _, nn_indices = nearest_neighbor_distances(train_flat, gen_flat)

        memorized_mask = ratios < ratio_threshold
        n_memorized = memorized_mask.sum().item()

        print(f"[Epoch {epoch}] Generated {n_gen} samples")
        print(f"[Epoch {epoch}] Ratio d1/avg(d2..d{k}): "
              f"mean={ratios.mean():.4f}, min={ratios.min():.4f}")
        print(f"[Epoch {epoch}] Memorized (ratio<{ratio_threshold}): "
              f"{n_memorized}/{n_gen} = {n_memorized/n_gen:.1%}")
        for r in [0.1, 0.3, 0.5, 0.7]:
            frac = (ratios < r).float().mean().item()
            print(f"[Epoch {epoch}] Frac memorized (r={r}): {frac:.4f}")

        # Plot memorized samples with their nearest training neighbors.
        _plot_memorization(
            generated, self.z_train_raw, nn_indices, ratios,
            memorized_mask, self.kl_prior, ratio_threshold,
            os.path.join(plotsdir(args.experiment),
                         f"memorization_{epoch:04d}.png"),
        )

        # Ratio histogram.
        _plot_ratio_histogram(
            ratios, ratio_threshold,
            os.path.join(plotsdir(args.experiment),
                         f"ratio_histogram_{epoch:04d}.png"),
        )

        # KL scatter: first 2 components, training vs generated.
        _plot_kl_scatter(
            generated, self.z_train_raw, args.num_train,
            os.path.join(plotsdir(args.experiment),
                         f"kl_scatter_{epoch:04d}.png"),
        )

        # Training loss plot.
        _plot_loss(
            self.train_obj, self.val_obj, epoch,
            os.path.join(plotsdir(args.experiment), "log.png"),
        )

    def dps_sample(self, args, d_obs, sigma_d, num_samples=16,
                   step_size=0.3):
        """DPS posterior sampling in KL space through Helmholtz forward model.

        Implements DPS (Chung et al., 2023) with DDIM reverse steps and
        per-sample gradient normalization.

        At each reverse step t:
            1. Predict noise, compute Tweedie estimate z0_hat
            2. Likelihood gradient via adjoint + chain rule through Tweedie
            3. Normalize gradient per-sample (step_size controls magnitude)
            4. DDIM reverse step: z_{t-1} from z_t
            5. Correction: z_{t-1} -= step_size * grad / ||grad||

        Args:
            args: Experiment arguments.
            d_obs: Observed data (n_src, n_rec), complex numpy array.
            sigma_d: Data noise standard deviation.
            num_samples: Number of posterior samples.
            step_size: DPS step size (controls update magnitude after
                       gradient normalization).

        Returns:
            z_samples: Posterior samples in KL space (num_samples, d),
                       unnormalized (physical).
            snapshots: List of (timestep, z_raw) tuples for convergence viz.
        """
        self.ema_model.eval()
        nt = self.noise_scheduler.nt
        alpha_cumprod = self.noise_scheduler.alphas_cumprod

        # Initialize from background velocity (z=0 gives v=v_bg everywhere).
        z_init_raw = torch.zeros(self.d, dtype=torch.float32)
        z_init_norm = self.normalizer.normalize(z_init_raw).to(self.device)
        # Noise to t=T level: z_T = sqrt(alpha_T)*z0 + sqrt(1-alpha_T)*eps.
        a_T = alpha_cumprod[nt - 1]
        z_t = (a_T.sqrt() * z_init_norm.unsqueeze(0).expand(num_samples, -1)
               + (1 - a_T).sqrt() * torch.randn(num_samples, self.d,
                                                  device=self.device))
        print(f"DPS init: z=0 (v_bg={self.kl_prior.v_background}), "
              f"z_init_norm={z_init_norm.norm():.2f}, "
              f"a_T={a_T:.4f}, z_T_norm={z_t.norm(dim=-1).mean():.2f}")

        # Save snapshots at log-spaced intervals for convergence viz.
        snapshot_times = set([nt - 1, int(0.75 * nt), int(0.5 * nt),
                             int(0.25 * nt), int(0.1 * nt), 0])
        snapshots = []

        # Snapshot of initialization (constant background before diffusion).
        snapshots.append(("init", z_init_raw.unsqueeze(0).expand(
            num_samples, -1).clone()))

        for t in tqdm(
            reversed(range(nt)), total=nt,
            desc="DPS sampling", dynamic_ncols=True,
        ):
            t_batch = torch.full(
                (num_samples,), t, device=self.device, dtype=torch.long,
            )

            with torch.no_grad():
                eps_pred = self.ema_model(z_t, t_batch)

            # Tweedie estimate of z0 (in normalized space).
            a_t = alpha_cumprod[t]
            z0_hat = (z_t - (1 - a_t).sqrt() * eps_pred) / a_t.sqrt()

            # Unnormalize Tweedie estimate to physical KL space.
            z0_raw = self.normalizer.unnormalize(z0_hat.detach().cpu())
            z0_np = z0_raw.numpy()

            # Save snapshot of Tweedie estimate.
            if t in snapshot_times:
                snapshots.append((t, z0_raw.clone()))

            # Compute likelihood gradient for each sample via adjoint.
            grad_z_batch = np.zeros((num_samples, self.d), dtype=np.float32)

            for i in range(num_samples):
                m_i = self.kl_prior.reconstruct(z0_np[i])
                grad_m = self.solver.adjoint_gradient(
                    m_i, self.omega,
                    self.src_positions, self.rec_positions,
                    d_obs, sigma_d,
                )
                grad_z_raw = self.G.T @ grad_m.ravel()
                # Chain rule through normalizer and Tweedie.
                norm_std = self.normalizer.std.numpy() + self.normalizer.eps
                grad_z_batch[i] = (
                    grad_z_raw * norm_std / a_t.sqrt().item()
                )

            grad_z = torch.tensor(
                grad_z_batch, dtype=torch.float32, device=self.device,
            )

            # Per-sample gradient normalization.
            grad_norms = grad_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            grad_z_normalized = grad_z / grad_norms

            # DDIM reverse step (deterministic, no noise injection).
            with torch.no_grad():
                if t > 0:
                    a_prev = alpha_cumprod[t - 1]
                    z_t = (a_prev.sqrt() * z0_hat
                           + (1 - a_prev).sqrt() * eps_pred)
                else:
                    z_t = z0_hat

            # DPS likelihood correction (after reverse step).
            z_t = z_t - step_size * grad_z_normalized

            if t % 50 == 0:
                z_norm = self.normalizer.unnormalize(
                    z_t.detach().cpu()).norm(dim=-1).mean().item()
                # Report data misfit for sample 0.
                m_diag = self.kl_prior.reconstruct(z0_np[0])
                d_diag = self.solver.forward(
                    m_diag, self.omega,
                    self.src_positions, self.rec_positions,
                )
                misfit_diag = 0.5 * np.sum(
                    np.abs(d_diag - d_obs)**2) / sigma_d**2
                print(f"  t={t}: z_norm={z_norm:.2f}, "
                      f"grad_norm={grad_norms.mean().item():.2e}, "
                      f"misfit[0]={misfit_diag:.1f}")

        z_out = self.normalizer.unnormalize(z_t.detach().cpu())
        snapshots.append((-1, z_out.clone()))  # final result
        return z_out, snapshots


def _dense_src_rec(grid_size, n_src=50, n_rec=200):
    """Create dense source/receiver geometry matching SVGD paper.

    50 sources at the top, 200 receivers at the bottom.
    """
    return make_src_rec(grid_size, grid_size, n_src, n_rec)


def run_dps_experiment(example, args):
    """Run DPS posterior sampling and create visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use dense source/receiver geometry (50 src, 200 rec) for inversion,
    # matching the SVGD paper setup.  Training checkpoints are unaffected
    # since the DDPM operates in KL space.
    example.src_positions, example.rec_positions = _dense_src_rec(
        args.grid_size,
    )

    # True model: Gaussian lens (NOT projected to KL basis).
    m_true, v_true = make_gaussian_lens(
        grid_size=args.grid_size,
        v_background=args.v_background,
        v_anomaly=getattr(args, "v_min", 1.5),
    )

    # Forward data from true model (on the full grid, not KL-reconstructed).
    d_clean = example.solver.forward(
        m_true, example.omega,
        example.src_positions, example.rec_positions,
    )
    noise_level = args.noise_rel * np.abs(d_clean).max()
    rng = np.random.default_rng(args.seed)
    d_obs = d_clean + noise_level * (
        rng.standard_normal(d_clean.shape)
        + 1j * rng.standard_normal(d_clean.shape)
    ) / np.sqrt(2)

    print(f"Data shape: {d_obs.shape}, noise level: {noise_level:.6f}")
    print(f"True velocity range: [{v_true.min():.3f}, {v_true.max():.3f}] km/s")

    # DPS.
    z_samples, snapshots = example.dps_sample(
        args, d_obs, noise_level,
        num_samples=args.dps_num_samples,
        step_size=args.dps_step_size,
    )

    # Memorization metrics (in raw z-space).
    train_flat = example.z_train_raw.reshape(example.z_train_raw.shape[0], -1)
    dps_flat = z_samples.reshape(z_samples.shape[0], -1)

    k = min(50, args.num_train)
    ratios = ratio_mem_values(train_flat, dps_flat, k_neighbors=k)
    print(f"DPS Mem ratio d1/avg(d2..d{k}): "
          f"mean={ratios.mean():.4f}, min={ratios.min():.4f}")
    for r in [0.3, 0.5, 0.7]:
        rmem = ratio_mem_metric(train_flat, dps_flat, ratio=r, k_neighbors=k)
        print(f"DPS Frac memorized (r={r}): {rmem:.4f}")

    nn_dists, nn_indices = nearest_neighbor_distances(train_flat, dps_flat)
    print(f"Mean NN distance: {nn_dists.mean():.4f}")

    # Save results.
    results = {
        "m_true": m_true,
        "v_true": v_true,
        "d_obs": d_obs,
        "z_samples": z_samples.numpy(),
        "nn_indices": nn_indices.numpy(),
        "nn_dists": nn_dists.numpy(),
        "noise_level": noise_level,
    }
    torch.save(
        results,
        os.path.join(checkpointsdir(args.experiment), "dps_results.pth"),
    )

    # Convergence visualization: Tweedie estimates throughout DPS + nearest NN.
    _plot_dps_convergence(
        snapshots, example.z_train_raw, example.kl_prior,
        m_true, v_true, args,
    )

    # Velocity images: true, DPS samples, nearest training.
    _plot_dps_results(
        m_true, v_true, z_samples, example.z_train_raw, nn_indices,
        example.kl_prior, args,
    )

    # Data fit plot.
    _plot_data_fit(
        z_samples, d_obs, example, args,
    )

    return results


def _plot_memorization(z_gen, z_train, nn_indices, ratios, memorized_mask,
                       kl_prior, threshold, save_path):
    """Plot memorized generated samples alongside nearest training neighbors.

    Only shows samples identified as memorized (ratio < threshold).
    Each column shows: generated (top), nearest train neighbor (bottom),
    with the memorization ratio annotated.  Up to 8 columns.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only show memorized samples (ratio < threshold).
    mem_indices = torch.where(memorized_mask)[0]
    if len(mem_indices) == 0:
        print("No memorized samples to plot.")
        return

    # Sort by ratio (lowest first) and take up to 8.
    mem_ratios = ratios[mem_indices]
    sorted_order = mem_ratios.argsort()
    show_idx = mem_indices[sorted_order[:8]]
    n_show = len(show_idx)

    fig, axes = plt.subplots(
        2, n_show, figsize=(2.2 * n_show, 4.8),
        gridspec_kw={"wspace": 0.06, "hspace": 0.18},
    )
    if n_show == 1:
        axes = axes[:, np.newaxis]

    # Compute vmin/vmax from the data with 10% padding for sharper contrast.
    all_v = []
    for idx in show_idx:
        z_g = z_gen[idx.item()].numpy()
        z_nn = z_train[nn_indices[idx.item()]].numpy()
        all_v.append(kl_prior.to_velocity(kl_prior.reconstruct(z_g)))
        all_v.append(kl_prior.to_velocity(kl_prior.reconstruct(z_nn)))
    v_all = np.concatenate([v.ravel() for v in all_v])
    v_range = v_all.max() - v_all.min()
    vmin_global = v_all.min() + 0.15 * v_range
    vmax_global = v_all.max() - 0.15 * v_range

    for col, idx in enumerate(show_idx):
        idx = idx.item()
        z_g = z_gen[idx].numpy()
        z_nn = z_train[nn_indices[idx]].numpy()
        r = ratios[idx].item()

        v_gen = kl_prior.to_velocity(kl_prior.reconstruct(z_g))
        v_nn = kl_prior.to_velocity(kl_prior.reconstruct(z_nn))

        axes[0, col].imshow(
            v_gen.T, cmap="terrain", vmin=vmin_global, vmax=vmax_global,
            aspect="auto", origin="upper",
        )
        axes[1, col].imshow(
            v_nn.T, cmap="terrain", vmin=vmin_global, vmax=vmax_global,
            aspect="auto", origin="upper",
        )

        axes[0, col].set_title(f"r = {r:.3f}", fontsize=10, fontweight="bold",
                               color="green")
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("green")
                spine.set_linewidth(2)

    axes[0, 0].set_ylabel("Generated", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Nearest train", fontsize=13, fontweight="bold")

    n_mem = memorized_mask.sum().item()
    n_total = len(ratios)
    fig.suptitle(
        f"Memorized: {n_mem}/{n_total} samples with "
        f"$r < {threshold}$  (sorted by ratio)",
        fontsize=13, y=1.02,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _plot_ratio_histogram(ratios, threshold, save_path):
    """Histogram of per-sample memorization ratios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(ratios.numpy(), bins=40, color="steelblue", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"threshold = {threshold}")
    n_mem = (ratios < threshold).sum().item()
    ax.set_xlabel("$d_1 / \\mathrm{avg}(d_2 \\ldots d_k)$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Memorization ratio distribution  "
                 f"({n_mem}/{len(ratios)} memorized)", fontsize=11)
    ax.legend(fontsize=10)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _plot_kl_scatter(z_gen, z_train, n_train, save_path):
    """Scatter plot of first 2 KL components: training (blue) vs generated (red)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_gen_np = z_gen.numpy() if hasattr(z_gen, 'numpy') else z_gen
    z_train_np = z_train.numpy() if hasattr(z_train, 'numpy') else z_train

    # Subsample training points if more than generated.
    n_gen = len(z_gen_np)
    if len(z_train_np) > n_gen:
        idx = np.random.default_rng(42).choice(len(z_train_np), n_gen, replace=False)
        z_train_np = z_train_np[idx]

    s_train = 20
    s_gen = 25

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(z_train_np[:, 0], z_train_np[:, 1],
               color="#1f77b4", s=s_train, alpha=0.35,
               label="Training", zorder=1)
    ax.scatter(z_gen_np[:, 0], z_gen_np[:, 1],
               facecolors="none", edgecolors="#d62728", s=s_gen,
               linewidths=0.8, alpha=0.7, label="Generated", zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_title(f"$N = {n_train}$", fontsize=12)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _plot_dps_convergence(snapshots, z_train, kl_prior, m_true, v_true, args):
    """Plot DPS convergence: Tweedie estimates at different timesteps.

    Shows sample 0's velocity field at each snapshot timestep, plus the
    nearest training example to the final result.

    Args:
        snapshots: List of (timestep, z_raw) from dps_sample.
        z_train: Training KL coefficients (N, d).
        kl_prior: VelocityKLPrior instance.
        m_true: True squared-slowness (grid_size, grid_size).
        v_true: True velocity (grid_size, grid_size).
        args: Experiment arguments.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_snap = len(snapshots)
    # Show: true model | snapshots | nearest training
    n_cols = n_snap + 2
    fig, axes = plt.subplots(
        1, n_cols, figsize=(2.0 * n_cols, 2.4),
        gridspec_kw={"wspace": 0.06},
    )

    vmin, vmax = 1.5, 2.5

    # True model.
    axes[0].imshow(
        v_true.T, cmap="jet", vmin=vmin, vmax=vmax,
        aspect="auto", origin="upper",
    )
    axes[0].set_title("True", fontsize=7)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Snapshots.
    for col, (t, z_raw) in enumerate(snapshots):
        v_snap = kl_prior.to_velocity(kl_prior.reconstruct(z_raw[0].numpy()))
        ax = axes[col + 1]
        ax.imshow(
            v_snap.T, cmap="jet", vmin=vmin, vmax=vmax,
            aspect="auto", origin="upper",
        )
        if t == "init":
            label = "Init"
        elif isinstance(t, int) and t >= 0:
            label = f"t={t}"
        else:
            label = "Final"
        ax.set_title(label, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    # Nearest training example to final sample.
    final_z = snapshots[-1][1]
    train_flat = z_train.reshape(z_train.shape[0], -1)
    final_flat = final_z[0:1].reshape(1, -1)
    _, nn_idx = nearest_neighbor_distances(train_flat, final_flat)
    v_nn = kl_prior.to_velocity(
        kl_prior.reconstruct(z_train[nn_idx[0]].numpy())
    )
    axes[-1].imshow(
        v_nn.T, cmap="jet", vmin=vmin, vmax=vmax,
        aspect="auto", origin="upper",
    )
    axes[-1].set_title("Nearest train", fontsize=7)
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])

    os.makedirs(FIGS_DIR, exist_ok=True)
    save_path = os.path.join(
        FIGS_DIR, f"helmholtz_dps_convergence_N{args.num_train}.png",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    # Also save to plots dir.
    plt.savefig(
        os.path.join(plotsdir(args.experiment),
                     f"dps_convergence_N{args.num_train}.png"),
        dpi=300, bbox_inches="tight", pad_inches=0.02,
    )
    plt.close(fig)
    print(f"Saved convergence plot to {save_path}")


def _plot_dps_results(m_true, v_true, z_dps, z_train, nn_indices,
                      kl_prior, args):
    """Plot DPS posterior samples: true, samples, nearest training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(4, z_dps.shape[0])

    vmin, vmax = 1.5, 2.5

    fig, axes = plt.subplots(
        3, n_show + 1, figsize=(3 * (n_show + 1), 9),
        gridspec_kw={"wspace": 0.05, "hspace": 0.15},
    )

    # Row 0: true + DPS samples.
    axes[0, 0].imshow(
        v_true.T, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto",
        origin="upper",
    )
    axes[0, 0].set_title("Ground truth", fontsize=10)
    axes[0, 0].axis("off")

    for i in range(n_show):
        v_dps = kl_prior.to_velocity(
            kl_prior.reconstruct(z_dps[i].numpy())
        )
        axes[0, i + 1].imshow(
            v_dps.T, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto",
            origin="upper",
        )
        axes[0, i + 1].set_title(f"DPS sample {i+1}", fontsize=10)
        axes[0, i + 1].axis("off")

    # Row 1: nearest training examples.
    axes[1, 0].axis("off")
    for i in range(n_show):
        v_nn = kl_prior.to_velocity(
            kl_prior.reconstruct(z_train[nn_indices[i]].numpy())
        )
        axes[1, i + 1].imshow(
            v_nn.T, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto",
            origin="upper",
        )
        axes[1, i + 1].set_title("Nearest train", fontsize=10)
        axes[1, i + 1].axis("off")

    # Row 2: difference (DPS - NN).
    axes[2, 0].axis("off")
    for i in range(n_show):
        v_dps = kl_prior.to_velocity(
            kl_prior.reconstruct(z_dps[i].numpy())
        )
        v_nn = kl_prior.to_velocity(
            kl_prior.reconstruct(z_train[nn_indices[i]].numpy())
        )
        diff = v_dps - v_nn
        max_abs = max(np.abs(diff).max(), 1e-6)
        axes[2, i + 1].imshow(
            diff.T, cmap="seismic", vmin=-max_abs, vmax=max_abs,
            aspect="auto", origin="upper",
        )
        axes[2, i + 1].set_title("Difference", fontsize=10)
        axes[2, i + 1].axis("off")

    os.makedirs(FIGS_DIR, exist_ok=True)
    save_path = os.path.join(
        plotsdir(args.experiment), f"dps_results_N{args.num_train}.png",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    # Also save to brainstorming figs.
    fig.savefig(
        os.path.join(FIGS_DIR, f"helmholtz_dps_N{args.num_train}.png"),
        dpi=300, bbox_inches="tight", pad_inches=0.02,
    )
    plt.close(fig)


def _plot_data_fit(z_samples, d_obs, example, args):
    """Plot predicted vs observed data for DPS samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(4, z_samples.shape[0])
    n_src = d_obs.shape[0]

    fig, axes = plt.subplots(
        n_show, n_src, figsize=(3 * n_src, 2.5 * n_show),
        gridspec_kw={"wspace": 0.2, "hspace": 0.3},
    )
    if n_show == 1:
        axes = axes[np.newaxis, :]
    if n_src == 1:
        axes = axes[:, np.newaxis]

    for i in range(n_show):
        m_i = example.kl_prior.reconstruct(z_samples[i].numpy())
        d_pred = example.solver.forward(
            m_i, example.omega,
            example.src_positions, example.rec_positions,
        )
        for s in range(n_src):
            ax = axes[i, s]
            ax.plot(np.abs(d_obs[s]), "k-", alpha=0.7, label="Observed")
            ax.plot(np.abs(d_pred[s]), "r--", alpha=0.7, label="Predicted")
            if i == 0:
                ax.set_title(f"Source {s+1}", fontsize=10)
            if s == 0:
                ax.set_ylabel(f"Sample {i+1}", fontsize=10)
            if i == 0 and s == 0:
                ax.legend(fontsize=8)
            ax.set_xticks([])

    save_path = os.path.join(
        plotsdir(args.experiment), f"data_fit_N{args.num_train}.png",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _laplacian_reg(m, m_bg, dx):
    """Compute Laplacian smoothing regularization and its gradient.

    reg = 0.5 * sum(|grad_x m|^2 + |grad_z m|^2) * dx^2
    Encourages spatial smoothness without biasing toward m_bg.

    Returns:
        reg_val: Scalar regularization value.
        grad_reg: Gradient w.r.t. m, same shape as m.
    """
    dmx = np.diff(m, axis=0) / dx
    dmz = np.diff(m, axis=1) / dx
    reg_val = 0.5 * (np.sum(dmx**2) + np.sum(dmz**2)) * dx**2

    # Gradient: negative discrete Laplacian applied to m.
    grad_reg = np.zeros_like(m)
    # x-direction: d/dm_ij of sum (m_{i+1,j}-m_{i,j})^2 / dx^2 * dx^2
    grad_reg[:-1, :] -= dmx
    grad_reg[1:, :] += dmx
    grad_reg[:, :-1] -= dmz
    grad_reg[:, 1:] += dmz
    return reg_val, grad_reg


def run_baselines(example, args):
    """Run MAP baselines in m-space with physical bounds.

    Baselines:
        1. Smooth MAP: data misfit + Laplacian smoothing (standard FWI)
        2. GRF MAP: data misfit + GRF log-prior (informative prior)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize as sp_minimize

    nx = args.grid_size
    nz = args.grid_size
    dx = 2.0 / nx  # 2 km domain

    # Use dense source/receiver geometry (50 src, 200 rec).
    example.src_positions, example.rec_positions = _dense_src_rec(
        args.grid_size,
    )

    # True model: same Gaussian lens as DPS.
    m_true, v_true = make_gaussian_lens(
        grid_size=args.grid_size,
        v_background=args.v_background,
        v_anomaly=getattr(args, "v_min", 1.5),
    )

    # Forward data from true model.
    d_clean = example.solver.forward(
        m_true, example.omega,
        example.src_positions, example.rec_positions,
    )
    noise_level = args.noise_rel * np.abs(d_clean).max()
    rng = np.random.default_rng(args.seed)
    d_obs = d_clean + noise_level * (
        rng.standard_normal(d_clean.shape)
        + 1j * rng.standard_normal(d_clean.shape)
    ) / np.sqrt(2)

    n_data = d_obs.shape[0] * d_obs.shape[1]
    print(f"Data shape: {d_obs.shape} ({n_data} complex), "
          f"noise level: {noise_level:.6f}")

    # Bounds matching GRF domain: v in [v_min, v_max] from config.
    v_min_cfg = getattr(args, "v_min", 1.5)
    v_max_cfg = getattr(args, "v_max", 3.0)
    m_lo = 1.0 / v_max_cfg**2
    m_hi = 1.0 / v_min_cfg**2
    bounds_m = [(m_lo, m_hi)] * (nx * nz)
    m_bg = 1.0 / args.v_background**2

    def make_objective(reg_mode):
        """Return (loss, grad) callable for L-BFGS in m-space."""
        call_count = [0]

        def objective(m_flat):
            m = m_flat.reshape(nx, nz)

            d_pred = example.solver.forward(
                m, example.omega,
                example.src_positions, example.rec_positions,
            )
            misfit = 0.5 * np.sum(np.abs(d_pred - d_obs)**2) / noise_level**2

            grad_m = example.solver.adjoint_gradient(
                m, example.omega,
                example.src_positions, example.rec_positions,
                d_obs, noise_level,
            )
            grad = grad_m.ravel().astype(np.float64)
            loss = float(misfit)

            if reg_mode == "smooth":
                # Laplacian smoothing: penalizes spatial gradients of m.
                alpha = 1e6
                reg_val, grad_reg = _laplacian_reg(m, m_bg, dx)
                loss += alpha * reg_val
                grad += alpha * grad_reg.ravel()
            elif reg_mode == "grf":
                # GRF log-prior (acts as informative prior).
                prior_weight = 50.0
                score_m = example.grf.score(m)
                log_prior = float(example.grf.log_prob(m))
                loss -= prior_weight * log_prior
                grad -= prior_weight * score_m.ravel()

            call_count[0] += 1
            if call_count[0] % 20 == 1:
                v = 1.0 / np.sqrt(np.clip(m, 1e-6, None))
                print(f"    eval {call_count[0]}: loss={loss:.1f}, "
                      f"misfit={misfit:.1f}, "
                      f"v=[{v.min():.2f}, {v.max():.2f}]")

            return loss, grad

        return objective

    m0 = np.full(nx * nz, m_bg, dtype=np.float64)

    results = {}
    labels = [("smooth", "smooth"), ("grf_prior", "grf")]
    for method, reg_mode in labels:
        print(f"\n=== Baseline: {method} (m-space, "
              f"v bounds=[{v_min_cfg}, {v_max_cfg}]) ===")
        obj_fn = make_objective(reg_mode)

        res = sp_minimize(
            obj_fn, m0.copy(), method="L-BFGS-B", jac=True,
            bounds=bounds_m,
            options={"maxiter": 2000, "maxfun": 5000, "ftol": 1e-14},
        )
        m_result = res.x.reshape(nx, nz)
        v_result = 1.0 / np.sqrt(np.clip(m_result, 1e-6, None))
        print(f"  converged: {res.success}, nfev={res.nfev}, "
              f"loss={res.fun:.1f}, "
              f"v=[{v_result.min():.2f}, {v_result.max():.2f}]")
        results[method] = m_result

    # Plot.
    fig, axes = plt.subplots(
        1, 3, figsize=(6.6, 2.4),
        gridspec_kw={"wspace": 0.06},
    )
    vmin, vmax = 1.5, 2.5
    titles = ["True", "Smooth MAP", "GRF MAP"]
    fields = [v_true,
              1.0 / np.sqrt(np.clip(results["smooth"], 1e-6, None)),
              1.0 / np.sqrt(np.clip(results["grf_prior"], 1e-6, None))]

    for ax, title, v in zip(axes, titles, fields):
        ax.imshow(v.T, cmap="jet", vmin=vmin, vmax=vmax,
                  aspect="auto", origin="upper")
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(FIGS_DIR, exist_ok=True)
    save_path = os.path.join(FIGS_DIR, "helmholtz_baselines.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(
        os.path.join(plotsdir(args.experiment), "baselines.png"),
        dpi=300, bbox_inches="tight", pad_inches=0.02,
    )
    plt.close(fig)
    print(f"\nSaved baseline figure to {save_path}")

    return results


def _plot_loss(train_obj, val_obj, epoch, save_path):
    """Plot training and validation loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure("training logs", figsize=(7, 4))
    plt.plot(
        np.linspace(0, epoch + 1, len(train_obj)),
        train_obj, color="orange", alpha=1.0, label="training loss",
    )
    if val_obj:
        val_epochs = np.arange(len(val_obj)) * 10
        plt.plot(val_epochs, val_obj, color="k", alpha=0.8,
                 label="validation loss")
    plt.ticklabel_format(axis="y", style="sci", useMathText=True)
    plt.title("Training loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(save_path, format="png", bbox_inches="tight",
                dpi=400, pad_inches=0.02)
    plt.close(fig)


if "__main__" == __name__:
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=[
            "experiment_name",
            "gpu_id",
            "phase",
            "noise_rel",
            "dps_step_size",
            "dps_num_samples",
            "grf_alpha",
            "grf_tau",
            "v_min",
            "v_max",
            "upload",
        ],
    )

    # Add testing_epoch if not present.
    if not hasattr(args, "testing_epoch"):
        args.testing_epoch = args.max_epochs - 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    resume_epoch = getattr(args, "testing_epoch", -1)

    if args.testing_epoch == -1:
        args.testing_epoch = args.max_epochs - 1

    example = HelmholtzDDPM(args)

    if args.phase == "train":
        start_epoch = 0
        if resume_epoch >= 0 and resume_epoch < args.max_epochs - 1:
            try:
                example.load_checkpoint(args, resume=True)
                start_epoch = resume_epoch + 1
                args.testing_epoch = args.max_epochs - 1
            except ValueError:
                print("No checkpoint found, training from scratch.")
        example.train(args, start_epoch=start_epoch)
        example.test(args, args.max_epochs - 1)
    elif args.phase == "dps":
        example.load_checkpoint(args)
        run_dps_experiment(example, args)
    elif args.phase == "baselines":
        run_baselines(example, args)
    else:
        example.load_checkpoint(args)
        example.test(args, args.testing_epoch)

    if hasattr(args, "upload") and args.upload:
        upload_to_cloud(args)
