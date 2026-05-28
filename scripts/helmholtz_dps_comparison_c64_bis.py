"""DPS comparison across N = {50, 200, 1000} for Helmholtz FWI.

Mirrors helmholtz_dps_cs.py but replaces the compressed sensing forward
operator with the frequency-domain Helmholtz equation.  Everything else
is identical: OOD true model, terrain colormap, loss tracking, memorization
bar chart, individual panel saving.

Usage:
    python scripts/helmholtz_dps_comparison.py [--gpu_id 0] [--seed 123]
"""
import os
import subprocess

import numpy as np
import torch

from copy import deepcopy
from multiprocessing import Pool
from tqdm import tqdm
from projorg import (checkpointsdir, plotsdir, setup_environment,
                     upload_to_cloud)

from mempost.models import NoiseScheduler
from grf_with_score import GaussianRandomField
from mempost.utils.helmholtz_c64 import HelmholtzSolver, make_src_rec
from mempost.utils.kl_prior import VelocityKLPrior
from mempost.utils.normalizer import Normalizer
from mempost.utils.memorization_metrics import (
    nearest_neighbor_distances,
    ratio_mem_values,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.abspath(__file__))

FIGS_DIR = os.path.join(os.path.dirname(_script_dir), "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

# Per-N model configs: (hidden_dim, nlayers, max_epochs, save_freq)
MODEL_CONFIGS = {
    50:   (2048, 12, 50000, 5000),
    200:  (2048, 12, 12500, 1250),
    1000: (2048, 12,  2500,  250),
}
# Checkpoint directory basenames for each N (must match training output).
_CKPT_DIRS = {
    50:   "helmholtz_fwi_sigma_w_2.5_kl_K-10_grid_size-64_v_background-2.0_sigma_m-0.02_npml-10_pml_max-100.0_n_src-5_n_rec-40_frequency-4.0_hidden_dim-2048_num_train-50_max_epochs-50000_save_freq-5000_holdout-0.1_n_clusters-5",
    200:  "helmholtz_fwi_sigma_w_2.5_kl_K-10_grid_size-64_v_background-2.0_sigma_m-0.02_npml-10_pml_max-100.0_n_src-5_n_rec-40_frequency-4.0_hidden_dim-2048_num_train-200_max_epochs-12500_save_freq-1250_holdout-0.1_n_clusters-5",
    1000: "helmholtz_fwi_sigma_w_2.5_kl_K-10_grid_size-64_v_background-2.0_sigma_m-0.02_npml-10_pml_max-100.0_n_src-5_n_rec-40_frequency-4.0_hidden_dim-2048_num_train-1000_max_epochs-2500_save_freq-250_holdout-0.1_n_clusters-5",
}
N_VALUES_DEFAULT = [50, 200, 1000]

# --- Multiprocessing worker for adjoint gradient ---
_worker_state = {}


def _init_worker(solver, omega, src_positions, rec_positions, d_obs, sigma_d,
                 kl_prior, G):
    _worker_state["solver"] = solver
    _worker_state["omega"] = omega
    _worker_state["src_positions"] = src_positions
    _worker_state["rec_positions"] = rec_positions
    _worker_state["d_obs"] = d_obs
    _worker_state["sigma_d"] = sigma_d
    _worker_state["kl_prior"] = kl_prior
    _worker_state["G"] = G


def _compute_grad(args):
    z0_i, norm_std, a_t_sqrt = args
    s = _worker_state
    m_i = s["kl_prior"].reconstruct(z0_i)
    grad_m = s["solver"].adjoint_gradient(
        m_i, s["omega"], s["src_positions"], s["rec_positions"],
        s["d_obs"], s["sigma_d"],
    )
    grad_z_raw = s["G"].T @ grad_m.ravel()
    return (grad_z_raw * norm_std / a_t_sqrt).astype(np.float32)


def load_model(n_train, device):
    from helmholtz_fwi import ScoreModel

    hidden_dim, nlayers, max_epochs, _ = MODEL_CONFIGS[n_train]
    exp_dir = checkpointsdir(_CKPT_DIRS[n_train], mkdir=False)
    ckpt_path = os.path.join(exp_dir, f"checkpoint_{max_epochs - 1}.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ScoreModel(
        input_size=100, hidden_dim=hidden_dim, nlayers=nlayers,
        emb_size=256, dropout=0.0,
    ).to(device)
    ema_model = deepcopy(model)
    if "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
    else:
        ema_model.load_state_dict(checkpoint["model_state_dict"])

    kl_prior = VelocityKLPrior(K=10, grid_size=64, v_background=2.0,
                               sigma_m=0.02)
    grf = GaussianRandomField(
        dim=2, size=64, alpha=3, tau=5,
        bounds=(1.0 / 3.0**2, 1.0 / 1.5**2),
    )
    np.random.seed(42)
    m_train = grf.sample(n_train)
    z_train = kl_prior.project(m_train)
    z_train_t = torch.tensor(z_train, dtype=torch.float32)
    normalizer = Normalizer(z_train_t)

    print(f"Loaded N={n_train} from {os.path.basename(exp_dir)}")
    return ema_model, normalizer, z_train_t


def dps_helmholtz(ema_model, normalizer, noise_scheduler, kl_prior, solver,
                  omega, src_positions, rec_positions, G, d_obs, sigma_d,
                  num_samples=128, step_size=0.15, n_lik_steps=1,
                  device="cpu", pool=None, normalize_grad=True):
    """DPS posterior sampling with Helmholtz forward operator.

    Returns z_samples (num_samples, d) and loss history dict.
    """
    ema_model.eval()
    nt = noise_scheduler.nt
    alpha_cumprod = noise_scheduler.alphas_cumprod
    d = kl_prior.d

    norm_std = normalizer.std.numpy() + normalizer.eps

    # Loss tracking.
    hist = {"likelihood": []}

    def record(z_raw_np):
        # Compute average likelihood over a few samples (expensive).
        n_eval = min(4, num_samples)
        nll_sum = 0.0
        for i in range(n_eval):
            m_i = kl_prior.reconstruct(z_raw_np[i])
            d_pred = solver.forward(m_i, omega, src_positions, rec_positions)
            residual = d_pred - d_obs
            nll_sum += 0.5 * np.sum(np.abs(residual)**2) / sigma_d**2
        nll = nll_sum / n_eval
        hist["likelihood"].append(nll)

    # Start from pure noise.
    z_t = torch.randn(num_samples, d, device=device)

    for t in tqdm(
        reversed(range(nt)), total=nt,
        desc="DPS-Helmholtz", dynamic_ncols=True,
    ):
        t_batch = torch.full(
            (num_samples,), t, device=device, dtype=torch.long,
        )

        with torch.no_grad():
            eps_pred = ema_model(z_t, t_batch)

        a_t = alpha_cumprod[t]
        z0_hat = (z_t - (1 - a_t).sqrt() * eps_pred) / a_t.sqrt()

        # DDIM reverse step.
        with torch.no_grad():
            if t > 0:
                a_prev = alpha_cumprod[t - 1]
                z_t = (a_prev.sqrt() * z0_hat
                       + (1 - a_prev).sqrt() * eps_pred)
            else:
                z_t = z0_hat

        # Multiple likelihood gradient steps per diffusion step.
        a_t_sqrt = a_t.sqrt().item()
        for _ in range(n_lik_steps):
            # Tweedie estimate at current z_t.
            with torch.no_grad():
                eps_cur = ema_model(z_t, t_batch)
                a_cur = alpha_cumprod[max(t - 1, 0)]
                z0_hat_cur = (z_t - (1 - a_cur).sqrt() * eps_cur) / a_cur.sqrt()
                z0_raw = normalizer.unnormalize(z0_hat_cur.cpu())
                z0_np = z0_raw.numpy()

            # Likelihood gradient via adjoint (parallelized).
            if pool is not None:
                args = [(z0_np[i], norm_std, a_t_sqrt)
                        for i in range(num_samples)]
                grad_z_batch = np.array(pool.map(_compute_grad, args))
            else:
                grad_z_batch = np.zeros((num_samples, d), dtype=np.float32)
                for i in range(num_samples):
                    grad_z_batch[i] = _compute_grad(
                        (z0_np[i], norm_std, a_t_sqrt))

            grad_z = torch.tensor(
                grad_z_batch, dtype=torch.float32, device=device,
            )
            if normalize_grad:
                grad_norms = grad_z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                z_t = z_t - step_size * grad_z / grad_norms
            else:
                z_t = z_t - step_size * grad_z
            if t in (199, 150, 100, 50, 10, 0):
                gn = grad_z.norm(dim=-1).mean().item()
                print(f"  t={t:3d} ||grad_z||={gn:.2e} "
                      f"step*||grad||={step_size*gn:.2e}")

        # Record losses periodically (forward solve is expensive).
        if t % 10 == 0:
            z0_raw_rec = normalizer.unnormalize(z0_hat.detach().cpu())
            record(z0_raw_rec.numpy())

        if t % 50 == 0:
            print(f"  t={t}: nll={hist['likelihood'][-1]:.1f}")

    z_out = normalizer.unnormalize(z_t.detach().cpu())
    return z_out, hist


CONFIG_FILE = "helmholtz_dps_comparison_c64_bis.json"


def main():
    cli = setup_environment(
        CONFIG_FILE,
        sequence_args_and_types=[("num_train_values", int)],
        ignore_arg_list=[
            "gpu_id",
            "plot_only",
            "n_workers",
            "upload",
            "no_grad_norm",
            "omp_num_threads",
            "cmap_velocity",
            "cmap_data",
            "mem_threshold",
            "num_train_values",
            "kl_K",
            "grid_size",
            "plot_grid",
            "v_background",
            "sigma_m",
            "grf_train_alpha",
            "grf_train_tau",
            "grf_ood_alpha",
            "grf_ood_tau",
            "npml",
            "pml_max",
            "frequency",
            "nt",
            "hidden_dim",
            "nlayers",
            "emb_size",
            "z_lim_min",
            "z_lim_max",
            "true_model_path",
            "ood",
            "dx",
            "pad",
        ],
    )
    # Convert int flags to bool where needed.
    cli.plot_only = bool(cli.plot_only)
    cli.no_grad_norm = bool(cli.no_grad_norm)
    cli.upload = bool(cli.upload)
    cli.ood = bool(getattr(cli, "ood", 1))
    z_lim_min = getattr(cli, "z_lim_min", None)
    z_lim_max = getattr(cli, "z_lim_max", None)
    cli.z_lim = (z_lim_min, z_lim_max) if z_lim_min is not None else None
    N_VALUES = cli.num_train_values

    if torch.cuda.is_available() and cli.gpu_id >= 0:
        device = torch.device(f"cuda:{cli.gpu_id}")
    else:
        device = torch.device("cpu")

    d = 100
    gs = cli.grid_size
    kl_prior = VelocityKLPrior(K=10, grid_size=gs, v_background=2.0,
                               sigma_m=0.02)
    kl_prior_plot = VelocityKLPrior(K=10, grid_size=cli.plot_grid,
                                     v_background=2.0, sigma_m=0.02)
    dx = getattr(cli, "dx", 2.0 / gs)
    solver = HelmholtzSolver(nx=gs, nz=gs, dx=dx, npml=cli.npml,
                             pml_max=cli.pml_max)
    omega = 2.0 * np.pi * cli.frequency
    pad = getattr(cli, "pad", 0)
    src_positions, rec_positions = make_src_rec(gs, gs, cli.n_src, cli.n_rec,
                                                pad=pad)
    G = kl_prior.kl_basis_matrix()

    noise_scheduler = NoiseScheduler(nt=200, beta_schedule="linear",
                                     device=device)

    # --- True model ---
    grf_train = GaussianRandomField(
        dim=2, size=gs, alpha=3, tau=5,
        bounds=(1.0 / 3.0**2, 1.0 / 1.5**2),
    )

    true_model = getattr(cli, "true_model", "ood" if cli.ood else "indist")

    if true_model == "file":
        # Load pre-computed true model from npz file.
        true_model_path = getattr(cli, "true_model_path", "")
        if true_model_path:
            file_path = true_model_path
        else:
            file_path = os.path.join(
                os.path.dirname(_script_dir), "data", "ood_true_model.npz")
        loaded = np.load(file_path)
        z_true = torch.tensor(loaded["z_true"], dtype=torch.float32)
        print(f"True model: loaded from {file_path} "
              f"(min_dist={float(loaded['min_dist']):.2f}, "
              f"recon_rmse={float(loaded['recon_rmse']):.4f})")

    elif true_model == "gausslens":
        # Load actual Gaussian lens from JUDI.jl benchmark (201x201, dx=10m).
        import h5py
        lens_path = os.path.join(
            os.path.dirname(_script_dir), "data", "GaussLens.jld")
        with h5py.File(lens_path, "r") as fh:
            m_lens_201 = np.array(fh["m"], dtype=np.float32)
        # Crop/resample to grid_size.
        m_lens = m_lens_201[:gs, :gs]
        z_true = torch.tensor(
            kl_prior.project(m_lens[np.newaxis])[0], dtype=torch.float32,
        )
        v_orig = 1.0 / np.sqrt(m_lens)
        print(f"True model: Gaussian lens from JUDI.jl "
              f"(v=[{v_orig.min():.2f}, {v_orig.max():.2f}])")

    elif true_model == "ood":
        grf_ood = GaussianRandomField(
            dim=2, size=gs,
            alpha=cli.grf_ood_alpha, tau=cli.grf_ood_tau,
            bounds=(1.0 / 3.0**2, 1.0 / 1.5**2),
        )
        np.random.seed(42)
        m_train_50 = grf_train.sample(50)
        z_train_50 = torch.tensor(
            kl_prior.project(m_train_50), dtype=torch.float32,
        )
        np.random.seed(9999)
        n_candidates = 2000
        m_candidates = grf_ood.sample(n_candidates)
        z_candidates = torch.tensor(
            kl_prior.project(m_candidates), dtype=torch.float32,
        )
        dists = torch.cdist(z_candidates, z_train_50)
        min_dists = dists.min(dim=1).values
        best_idx = min_dists.argmax().item()
        z_true = z_candidates[best_idx]
        print(f"True model: OOD GRF candidate {best_idx} "
              f"(min dist to N=50 train: {min_dists[best_idx]:.2f})")

    else:
        np.random.seed(777)
        m_true_raw = grf_train.sample(1)
        z_true = torch.tensor(
            kl_prior.project(m_true_raw)[0], dtype=torch.float32,
        )
        print("True model: in-distribution held-out GRF (seed=777)")

    m_true = kl_prior.reconstruct(z_true.numpy())
    v_true = kl_prior_plot.to_velocity(kl_prior_plot.reconstruct(z_true.numpy()))
    print(f"v=[{v_true.min():.2f}, {v_true.max():.2f}]")

    # --- Forward data ---
    d_clean = solver.forward(m_true, omega, src_positions, rec_positions)
    noise_level = cli.noise_rel * np.abs(d_clean).max()
    rng_noise = np.random.default_rng(cli.seed + 1000)
    d_obs = d_clean + noise_level * (
        rng_noise.standard_normal(d_clean.shape)
        + 1j * rng_noise.standard_normal(d_clean.shape)
    ) / np.sqrt(2)
    print(f"Data shape: {d_obs.shape}, noise level: {noise_level:.6f}")

    results_path = os.path.join(checkpointsdir(cli.experiment), "results.pth")
    panel_dir = plotsdir(cli.experiment)

    if cli.plot_only:
        # --- Reload saved results ---
        print(f"Loading saved results from {results_path}")
        save_data = torch.load(results_path, map_location="cpu", weights_only=False)
        v_true = save_data["v_true"]
        d_obs = save_data["d_obs"]
        all_results = {}
        for N in N_VALUES:
            nn_dists_key = f"nn_dists_{N}"
            if nn_dists_key in save_data:
                nn_dists_val = torch.from_numpy(save_data[nn_dists_key])
            else:
                # Backward compat: recompute from z_samples and z_train.
                z_s = torch.from_numpy(save_data[f"z_samples_{N}"])
                z_t = torch.from_numpy(save_data[f"z_train_{N}"])
                nn_dists_val = torch.cdist(z_s, z_t).min(dim=1).values
            all_results[N] = {
                "z_samples": torch.from_numpy(save_data[f"z_samples_{N}"]),
                "z_train": torch.from_numpy(save_data[f"z_train_{N}"]),
                "ratios": torch.from_numpy(save_data[f"ratios_{N}"]),
                "nn_indices": torch.from_numpy(save_data[f"nn_indices_{N}"]),
                "nn_dists": nn_dists_val,
                "errors": save_data[f"errors_{N}"],
                "dps_hist": save_data[f"dps_hist_{N}"],
            }
    else:
        # --- Run DPS for each N ---
        n_workers = min(cli.n_workers, cli.num_samples)
        print(f"Using {n_workers} workers for parallel adjoint gradients")
        pool = Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(solver, omega, src_positions, rec_positions,
                      d_obs, noise_level, kl_prior, G),
        )

        all_results = {}
        for N in N_VALUES:
            print(f"\n{'='*50}")
            print(f"  N = {N}")
            print(f"{'='*50}")

            ema_model, normalizer, z_train = load_model(N, device)

            z_samples, dps_hist = dps_helmholtz(
                ema_model, normalizer, noise_scheduler, kl_prior, solver,
                omega, src_positions, rec_positions, G,
                d_obs, noise_level,
                num_samples=cli.num_samples,
                step_size=cli.step_size,
                n_lik_steps=cli.n_lik_steps,
                device=device,
                pool=pool,
                normalize_grad=not cli.no_grad_norm,
            )

            # Memorization metrics.
            train_flat = z_train.reshape(z_train.shape[0], -1)
            dps_flat = z_samples.reshape(z_samples.shape[0], -1)
            k = min(50, N)
            ratios = ratio_mem_values(train_flat, dps_flat, k_neighbors=k)
            nn_dists, nn_indices = nearest_neighbor_distances(train_flat, dps_flat)

            # RMSE to true model.
            errors = []
            for i in range(cli.num_samples):
                v_i = kl_prior_plot.to_velocity(
                    kl_prior_plot.reconstruct(z_samples[i].numpy()))
                errors.append(np.sqrt(np.mean((v_i - v_true) ** 2)))
            errors = np.array(errors)

            print(f"  RMSE: mean={errors.mean():.4f}, min={errors.min():.4f}")
            print(f"  Mem ratio: mean={ratios.mean():.4f}, min={ratios.min():.4f}")
            for r in [0.5, 0.7]:
                print(f"  r<{r}: {(ratios < r).float().mean().item():.1%}")

            all_results[N] = {
                "z_samples": z_samples,
                "z_train": z_train,
                "ratios": ratios,
                "nn_indices": nn_indices,
                "nn_dists": nn_dists,
                "errors": errors,
                "dps_hist": {k: np.array(v) for k, v in dps_hist.items()},
            }

            # Save results incrementally after each N.
            save_data = {
                "v_true": v_true,
                "z_true": z_true.numpy(),
                "d_obs": d_obs,
                "noise_level": noise_level,
                "seed": cli.seed,
                "step_size": cli.step_size,
                "noise_rel": cli.noise_rel,
                "plot_grid": cli.plot_grid,
            }
            for N_done in all_results:
                r = all_results[N_done]
                save_data[f"z_samples_{N_done}"] = r["z_samples"].numpy()
                save_data[f"z_train_{N_done}"] = r["z_train"].numpy()
                save_data[f"ratios_{N_done}"] = r["ratios"].numpy()
                save_data[f"nn_indices_{N_done}"] = r["nn_indices"].numpy()
                save_data[f"nn_dists_{N_done}"] = r["nn_dists"].numpy()
                save_data[f"errors_{N_done}"] = r["errors"]
                save_data[f"dps_hist_{N_done}"] = r["dps_hist"]
            torch.save(save_data, results_path)
            print(f"\nSaved results to {results_path}")

        pool.close()
        pool.join()

    # Combined figure.
    dps_hists = {N: all_results[N]["dps_hist"] for N in N_VALUES}
    plot_combined_helmholtz(all_results, v_true, kl_prior_plot, N_VALUES,
                            dps_hists, d_obs=d_obs, cli=cli,
                            results_path=results_path, panel_dir=panel_dir)

    # Per-N diagnostic figures (same layout as sanity_check_dps.py).
    z_true_np = z_true.numpy()
    for N in N_VALUES:
        nll_hist = all_results[N]["dps_hist"].get("likelihood", [])
        plot_diagnostic_per_N(
            N, all_results[N], z_true_np, kl_prior_plot, v_true,
            nll_hist, cli.num_samples, cli.step_size, panel_dir,
            z_lim=cli.z_lim,
        )

    # Summary table.
    print(f"\n{'='*60}")
    print("DPS Helmholtz Memorization Summary")
    print(f"(noise_rel={cli.noise_rel}, step={cli.step_size})")
    print(f"{'='*60}")
    print(f"{'N':>6s}  {'RMSE':>8s}  {'mean ratio':>10s}  "
          f"{'r<0.5':>6s}  {'r<0.7':>6s}")
    for N in N_VALUES:
        r = all_results[N]
        ratios = r["ratios"]
        print(f"{N:6d}  {r['errors'].mean():8.4f}  "
              f"{ratios.mean():10.4f}  "
              f"{(ratios < 0.5).float().mean().item():6.1%}  "
              f"{(ratios < 0.7).float().mean().item():6.1%}")

    # Upload all plots (combined + diagnostics + individual panels).
    if cli.upload:
        upload_to_cloud(cli)


def plot_diagnostic_per_N(N, results, z_true_np, kl_prior, v_true,
                          nll_history, num_samples, step_size, panel_dir,
                          z_lim=None):
    """Per-N diagnostic figure matching sanity_check_dps.py layout (4x5)."""
    z_samples = results["z_samples"]
    z_train = results["z_train"]
    ratios = results["ratios"]
    nn_indices = results["nn_indices"]
    nn_dists = results["nn_dists"]

    # Velocity fields for all samples.
    v_samples = np.stack([
        kl_prior.to_velocity(kl_prior.reconstruct(z_samples[i].numpy()))
        for i in range(num_samples)
    ])
    v_mean = v_samples.mean(axis=0)
    v_std = v_samples.std(axis=0)
    v_error = v_mean - v_true

    v_range = float(v_true.max()) - float(v_true.min())
    vmin = float(v_true.min()) + 0.1 * v_range
    vmax = float(v_true.max()) - 0.1 * v_range

    n_show = 5
    fig, axes = plt.subplots(4, n_show, figsize=(3.2 * n_show, 12.5))

    # Row 0: True | Posterior mean | Posterior std | Error | NLL curve.
    axes[0, 0].imshow(v_true, cmap="terrain", vmin=vmin, vmax=vmax,
                       origin="lower")
    axes[0, 0].set_title("True (KL)")
    axes[0, 0].axis("off")

    rmse = np.sqrt((v_error**2).mean())
    axes[0, 1].imshow(v_mean, cmap="terrain", vmin=vmin, vmax=vmax,
                       origin="lower")
    axes[0, 1].set_title(f"Post. mean (RMSE={rmse:.3f})")
    axes[0, 1].axis("off")

    im_std = axes[0, 2].imshow(v_std, cmap="magma", origin="lower")
    axes[0, 2].set_title("Post. std")
    axes[0, 2].axis("off")
    fig.colorbar(im_std, ax=axes[0, 2], fraction=0.046, pad=0.04)

    err_lim = max(abs(v_error.min()), abs(v_error.max()))
    im_err = axes[0, 3].imshow(v_error, cmap="seismic", vmin=-err_lim,
                                vmax=err_lim, origin="lower")
    axes[0, 3].set_title("Error")
    axes[0, 3].axis("off")
    fig.colorbar(im_err, ax=axes[0, 3], fraction=0.046, pad=0.04)

    steps = np.arange(len(nll_history))
    axes[0, 4].plot(steps, nll_history, 'b-', linewidth=1)
    axes[0, 4].set_xlabel("Reverse step", fontsize=9)
    axes[0, 4].set_ylabel("NLL", fontsize=9)
    axes[0, 4].set_title("Neg log-likelihood")
    axes[0, 4].set_yscale("log")
    axes[0, 4].tick_params(labelsize=8)

    # Row 1-2: 5 most memorized samples and their NNs.
    worst_idx = ratios.argsort()[:n_show]
    for col, si in enumerate(worst_idx):
        si = si.item()
        axes[1, col].imshow(v_samples[si], cmap="terrain", vmin=vmin,
                            vmax=vmax, origin="lower")
        axes[1, col].set_title(f"#{si} (r={ratios[si]:.2f})", fontsize=9)
        axes[1, col].axis("off")

    for col, si in enumerate(worst_idx):
        si = si.item()
        ni = nn_indices[si].item()
        m_nn = kl_prior.reconstruct(z_train[ni].numpy())
        v_nn = kl_prior.to_velocity(m_nn)
        axes[2, col].imshow(v_nn, cmap="terrain", vmin=vmin, vmax=vmax,
                            origin="lower")
        axes[2, col].set_title(f"NN {ni} (d={nn_dists[si]:.1f})", fontsize=9)
        axes[2, col].axis("off")

    # Row 3: KL scatter + histogram + NN bar.
    z_out_np = z_samples.numpy()
    z_train_np = z_train.numpy()

    if z_lim is not None:
        zl = z_lim
    else:
        all_z = np.concatenate([z_train_np[:, :4], z_out_np[:, :4],
                                z_true_np[:4].reshape(1, -1)], axis=0)
        z_min, z_max = all_z.min(), all_z.max()
        z_pad = (z_max - z_min) * 0.05
        zl = (z_min - z_pad, z_max + z_pad)

    ax_s1 = axes[3, 0]
    ax_s1.scatter(z_train_np[:, 0], z_train_np[:, 1],
                  c="#1f77b4", s=15, alpha=0.4, label="Training", zorder=1)
    ax_s1.scatter(z_out_np[:, 0], z_out_np[:, 1],
                  facecolors="none", edgecolors="#d62728", s=20,
                  linewidths=0.6, alpha=0.6, label="Posterior", zorder=2)
    ax_s1.scatter([z_true_np[0]], [z_true_np[1]],
                  c="black", s=80, marker="*", zorder=3, label="True")
    ax_s1.set_xlabel("KL dim 0", fontsize=9)
    ax_s1.set_ylabel("KL dim 1", fontsize=9)
    ax_s1.legend(fontsize=7, loc="upper right")
    ax_s1.set_xlim(zl)
    ax_s1.set_ylim(zl)
    ax_s1.set_aspect("equal")
    ax_s1.set_title("KL scatter (dims 0-1)", fontsize=9)
    ax_s1.tick_params(labelsize=8)

    ax_s2 = axes[3, 1]
    ax_s2.scatter(z_train_np[:, 2], z_train_np[:, 3],
                  c="#1f77b4", s=15, alpha=0.4, zorder=1)
    ax_s2.scatter(z_out_np[:, 2], z_out_np[:, 3],
                  facecolors="none", edgecolors="#d62728", s=20,
                  linewidths=0.6, alpha=0.6, zorder=2)
    ax_s2.scatter([z_true_np[2]], [z_true_np[3]],
                  c="black", s=80, marker="*", zorder=3)
    ax_s2.set_xlabel("KL dim 2", fontsize=9)
    ax_s2.set_ylabel("KL dim 3", fontsize=9)
    ax_s2.set_xlim(zl)
    ax_s2.set_ylim(zl)
    ax_s2.set_aspect("equal")
    ax_s2.set_title("KL scatter (dims 2-3)", fontsize=9)
    ax_s2.tick_params(labelsize=8)

    # Ratio histogram.
    ratios_np = ratios.numpy()
    ax_hist = axes[3, 2]
    ax_hist.hist(ratios_np, bins=30, color="#2ca02c", alpha=0.7,
                 edgecolor="black", linewidth=0.5)
    ax_hist.axvline(0.7, color="red", linestyle="--", linewidth=1.5,
                    label="r=0.7")
    ax_hist.set_xlabel("Memorization ratio", fontsize=9)
    ax_hist.set_ylabel("Count", fontsize=9)
    n_mem = int((ratios < 0.7).sum().item())
    ax_hist.set_title(f"Ratios (mem: {n_mem}/{num_samples})", fontsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.tick_params(labelsize=8)

    # Unique NN bar chart.
    nn_idx_np = nn_indices.numpy()
    unique_nn = len(set(nn_idx_np.tolist()))
    ax_nn = axes[3, 3]
    nn_counts = {}
    for idx_val in nn_idx_np:
        nn_counts[idx_val] = nn_counts.get(idx_val, 0) + 1
    sorted_nn = sorted(nn_counts.items(), key=lambda x: -x[1])[:10]
    labels = [str(k) for k, _ in sorted_nn]
    counts = [v for _, v in sorted_nn]
    ax_nn.barh(labels, counts, color="#ff7f0e", edgecolor="black",
               linewidth=0.5)
    ax_nn.set_xlabel("Count", fontsize=9)
    ax_nn.set_title(f"Top NNs ({unique_nn} unique)", fontsize=9)
    ax_nn.tick_params(labelsize=8)
    ax_nn.invert_yaxis()

    axes[3, 4].axis("off")

    # Row labels.
    axes[0, 0].set_ylabel("Summary", fontsize=11)
    axes[1, 0].set_ylabel("Posterior samples", fontsize=11)
    axes[2, 0].set_ylabel("Nearest neighbor", fontsize=11)
    axes[3, 0].set_ylabel("Diagnostics", fontsize=11)

    fig.suptitle(f"DPS Helmholtz: N={N}, {num_samples} samples, "
                 f"step={step_size}, "
                 f"mem(r<0.7)={n_mem}/{num_samples}, "
                 f"mean r={ratios.mean():.3f}",
                 fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(panel_dir,
                            f"helmholtz_ddpm_diagnostic_N{N}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic: {out_path}")

    # --- Save individual panels for paper figures ---

    # KL scatter (dims 0-1).
    fs, axs = plt.subplots(1, 1, figsize=(3.0, 3.0))
    axs.scatter(z_train_np[:, 0], z_train_np[:, 1],
                c="#1f77b4", s=15, alpha=0.4, label="Training", zorder=1)
    axs.scatter(z_out_np[:, 0], z_out_np[:, 1],
                facecolors="none", edgecolors="#d62728", s=20,
                linewidths=0.6, alpha=0.6, label="Posterior", zorder=2)
    axs.scatter([z_true_np[0]], [z_true_np[1]],
                c="black", s=80, marker="*", zorder=3, label="True")
    axs.set_xlabel("KL dim 0", fontsize=9)
    axs.set_ylabel("KL dim 1", fontsize=9)
    axs.legend(fontsize=7, loc="upper right")
    if z_lim is not None:
        axs.set_xlim(z_lim)
        axs.set_ylim(z_lim)
    axs.set_aspect("equal")
    axs.tick_params(labelsize=8)
    scatter_path = os.path.join(panel_dir, f"scatter_01_N{N}.png")
    fs.savefig(scatter_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fs)
    print(f"  Saved: {scatter_path}")

    # Ratio histogram.
    fh, axh = plt.subplots(1, 1, figsize=(3.0, 2.5))
    axh.hist(ratios_np, bins=30, color="#2ca02c", alpha=0.7,
             edgecolor="black", linewidth=0.5)
    axh.axvline(0.7, color="red", linestyle="--", linewidth=1.5, label="r=0.7")
    axh.set_xlabel("Memorization ratio", fontsize=9)
    axh.set_ylabel("Count", fontsize=9)
    axh.set_title(f"$N={N}$: {n_mem}/{num_samples} memorized", fontsize=9)
    axh.legend(fontsize=8)
    axh.tick_params(labelsize=8)
    hist_path = os.path.join(panel_dir, f"histogram_N{N}.png")
    fh.savefig(hist_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fh)
    print(f"  Saved: {hist_path}")


def plot_combined_helmholtz(all_results, v_true, kl_prior, N_VALUES,
                            dps_hists=None, d_obs=None, cli=None,
                            results_path=None, panel_dir=None):
    """Combined figure matching CS layout.

    Layout (4 rows x 4 cols):
      Row 0: col0=True, cols1-3=DPS sample per N
      Row 1: col0=Observed data, cols1-3=Nearest train neighbor
      Row 2: col0=Mem. ratio bar, cols1-3=Difference
      Row 3: cols1-3=Likelihood+Prior curves
    """
    n_n = len(N_VALUES)
    v_range = float(v_true.max()) - float(v_true.min())
    vmin = float(v_true.min()) + 0.1 * v_range
    vmax = float(v_true.max()) - 0.1 * v_range

    has_losses = (dps_hists is not None
                  and any(v is not None for v in dps_hists.values()))

    # Pick a memorized sample (ratio < 0.7) if one exists.
    mem_threshold = 0.7
    sample_idx = {}
    mem_counts = {}
    for N in N_VALUES:
        ratios = all_results[N]["ratios"]
        n_total = len(ratios)
        memorized_mask = ratios < mem_threshold
        n_mem = memorized_mask.sum().item()
        mem_counts[N] = (n_mem, n_total)
        if memorized_mask.any():
            memorized_indices = memorized_mask.nonzero(as_tuple=True)[0]
            best_among_mem = ratios[memorized_indices].argmin()
            sample_idx[N] = memorized_indices[best_among_mem].item()
            tag = "MEMORIZED"
        else:
            sample_idx[N] = ratios.argmin().item()
            tag = "not memorized"
        print(f"  N={N}: sample {sample_idx[N]} "
              f"(ratio={ratios[sample_idx[N]]:.3f}, "
              f"RMSE={all_results[N]['errors'][sample_idx[N]]:.4f}) "
              f"[{tag}, {n_mem}/{n_total} memorized]")

    n_rows = 4 if has_losses else 3
    height_ratios = [1, 1, 1, 0.8] if has_losses else [1, 1, 1]
    fig = plt.figure(figsize=(2.6 * (n_n + 1), 2.5 * n_rows))
    gs = fig.add_gridspec(
        n_rows, n_n + 1,
        height_ratios=height_ratios,
        wspace=0.06, hspace=0.25,
    )
    axes = np.array([[fig.add_subplot(gs[r, c])
                      for c in range(n_n + 1)] for r in range(3)])

    if has_losses:
        ax_losses = []
        for col in range(n_n):
            ax = fig.add_subplot(gs[3, col + 1],
                                 sharey=ax_losses[0] if ax_losses else None)
            ax_losses.append(ax)
        ax_loss_empty = fig.add_subplot(gs[3, 0])
        ax_loss_empty.axis("off")

    def to_velocity(z):
        m = kl_prior.reconstruct(np.asarray(z))
        return kl_prior.to_velocity(m)

    # --- Row 0, col 0: True model ---
    axes[0, 0].imshow(v_true.T, cmap="terrain", vmin=vmin, vmax=vmax,
                      aspect="auto", origin="upper")
    axes[0, 0].set_title("True", fontsize=10, fontweight="bold")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # --- Row 1, col 0: Observed data |d_obs| (n_src x n_rec) ---
    if d_obs is not None:
        axes[1, 0].imshow(np.abs(d_obs), cmap="jet", aspect="auto")
        axes[1, 0].set_xlabel("Receivers", fontsize=8)
        axes[1, 0].set_ylabel("Sources", fontsize=8)
        axes[1, 0].tick_params(labelsize=6)
    else:
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
    axes[1, 0].set_title("$|\\mathbf{d}_{\\mathrm{obs}}|$", fontsize=10, fontweight="bold")

    # --- Row 2, col 0: Fraction memorized bar plot ---
    colors_bar = {50: "#d62728", 200: "#ff7f0e", 1000: "#2ca02c"}
    bar_x = np.arange(n_n)
    frac_mem = [mem_counts[N][0] / mem_counts[N][1] for N in N_VALUES]
    bar_colors = [colors_bar.get(N, "#333333") for N in N_VALUES]
    axes[2, 0].bar(bar_x, frac_mem, color=bar_colors, width=0.6, edgecolor="k",
                   linewidth=0.5)
    axes[2, 0].set_xticks(bar_x)
    axes[2, 0].set_xticklabels([str(N) for N in N_VALUES], fontsize=8)
    axes[2, 0].set_ylabel("Frac. memorized", fontsize=9)
    axes[2, 0].set_xlabel("$N$", fontsize=9)
    axes[2, 0].set_ylim(0, 1.05)
    for i, N in enumerate(N_VALUES):
        n_mem, n_total = mem_counts[N]
        axes[2, 0].text(i, frac_mem[i] + 0.03, f"{n_mem}/{n_total}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")
    axes[2, 0].tick_params(labelsize=8)

    # Precompute fields.
    dps_fields, nn_fields, diff_fields = {}, {}, {}
    for N in N_VALUES:
        r = all_results[N]
        idx = sample_idx[N]
        dps_fields[N] = to_velocity(r["z_samples"][idx])
        nn_fields[N] = to_velocity(r["z_train"][r["nn_indices"][idx]])
        diff_fields[N] = dps_fields[N] - nn_fields[N]

    max_abs_global = max(np.abs(diff_fields[N]).max() for N in N_VALUES)
    max_abs_global = max(max_abs_global, 0.01)

    for col, N in enumerate(N_VALUES):
        c = col + 1
        axes[0, c].imshow(dps_fields[N].T, cmap="terrain", vmin=vmin, vmax=vmax,
                          aspect="auto", origin="upper")
        n_mem, n_total = mem_counts[N]
        axes[0, c].set_title(f"$N = {N}$  ({n_mem}/{n_total} mem.)",
                              fontsize=9, fontweight="bold")

        axes[1, c].imshow(nn_fields[N].T, cmap="terrain", vmin=vmin, vmax=vmax,
                          aspect="auto", origin="upper")

        axes[2, c].imshow(diff_fields[N].T, cmap="seismic",
                          vmin=-max_abs_global, vmax=max_abs_global,
                          aspect="auto", origin="upper")

        for row in range(3):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    axes[0, 0].set_ylabel("DPS sample", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Nearest train", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Difference", fontsize=10, fontweight="bold")

    # Likelihood + prior curves.
    if has_losses:
        for col, N in enumerate(N_VALUES):
            ax = ax_losses[col]
            h = dps_hists.get(N)
            if h is None:
                continue
            steps = np.arange(len(h["likelihood"]))
            ax.semilogy(steps, h["likelihood"], label="Likelihood",
                        color="#d62728", linewidth=1.2, alpha=0.8)
            ax.set_xlabel("Iteration", fontsize=9)
            ax.set_title(f"$N = {N}$", fontsize=9)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("Neg. log", fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

    labels = "abcdefghijklmnop"
    for row in range(3):
        for col in range(n_n + 1):
            if row == 2 and col == 0:
                continue
            idx = row * (n_n + 1) + col
            if idx >= len(labels):
                continue
            axes[row, col].text(
                0.04, 0.93, f"({labels[idx]})",
                transform=axes[row, col].transAxes,
                fontsize=11, fontweight="bold", va="top", color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                          alpha=0.6, edgecolor="none"),
            )

    save_path = os.path.join(panel_dir, "helmholtz_dps_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved: {save_path}")

    # --- Save individual panels ---
    if panel_dir is None:
        panel_dir = os.path.join(FIGS_DIR, "helmholtz_panels_c64")
    os.makedirs(panel_dir, exist_ok=True)

    def save_panel(data, fname, cmap_name="terrain", vmin_=None, vmax_=None):
        f, a = plt.subplots(1, 1, figsize=(2.5, 2.5))
        a.imshow(data.T, cmap=cmap_name, vmin=vmin_, vmax=vmax_,
                 aspect="auto", origin="upper")
        a.set_xticks([]); a.set_yticks([])
        f.savefig(os.path.join(panel_dir, fname), dpi=300,
                  bbox_inches="tight", pad_inches=0.02)
        plt.close(f)

    save_panel(v_true, "true.png", vmin_=vmin, vmax_=vmax)

    # Observed data |d_obs| (n_src x n_rec).
    if d_obs is not None:
        f, a = plt.subplots(1, 1, figsize=(2.5, 2.5))
        a.imshow(np.abs(d_obs), cmap="jet", aspect="auto")
        a.set_xlabel("Receivers", fontsize=8)
        a.set_ylabel("Sources", fontsize=8)
        a.tick_params(labelsize=6)
        f.savefig(os.path.join(panel_dir, "observed.png"), dpi=300,
                  bbox_inches="tight", pad_inches=0.02)
        plt.close(f)

    for N in N_VALUES:
        save_panel(dps_fields[N], f"dps_N{N}.png", vmin_=vmin, vmax_=vmax)
        save_panel(nn_fields[N], f"nn_N{N}.png", vmin_=vmin, vmax_=vmax)
        save_panel(diff_fields[N], f"diff_N{N}.png", cmap_name="seismic",
                   vmin_=-max_abs_global, vmax_=max_abs_global)

    # Bar plot.
    fb, ab = plt.subplots(1, 1, figsize=(2.5, 2.0))
    ab.bar(bar_x, frac_mem, color=bar_colors, width=0.6, edgecolor="k", lw=0.5)
    ab.set_xticks(bar_x)
    ab.set_xticklabels([str(N) for N in N_VALUES], fontsize=9)
    ab.set_ylabel("Frac. memorized", fontsize=9)
    ab.set_xlabel("$N$", fontsize=9)
    ab.set_ylim(0, 1.05)
    for i, N in enumerate(N_VALUES):
        n_mem, n_total = mem_counts[N]
        ab.text(i, frac_mem[i] + 0.03, f"{n_mem}/{n_total}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")
    ab.tick_params(labelsize=8)
    fb.savefig(os.path.join(panel_dir, "bar_memorized.png"), dpi=300,
               bbox_inches="tight", pad_inches=0.02)
    plt.close(fb)

    # Loss curves per N.
    if has_losses:
        for N in N_VALUES:
            h = dps_hists.get(N)
            if h is None:
                continue
            fl, al = plt.subplots(1, 1, figsize=(2.5, 2.0))
            steps = np.arange(len(h["likelihood"]))
            al.semilogy(steps, h["likelihood"], label="Likelihood",
                        color="#d62728", linewidth=1.2)
            al.set_xlabel("Iteration", fontsize=9)
            al.set_ylabel("Neg. log", fontsize=9)
            al.legend(fontsize=7)
            al.grid(True, alpha=0.3)
            fl.savefig(os.path.join(panel_dir, f"loss_N{N}.png"), dpi=300,
                       bbox_inches="tight", pad_inches=0.02)
            plt.close(fl)

    # --- Posterior mean, std, error, and calibration ---
    # First pass: compute all fields to get shared colorbar limits.
    post_stats = {}
    global_std_max = 0.0
    global_err_max = 0.0
    for N in N_VALUES:
        z_samps = all_results[N]["z_samples"].numpy()
        v_fields = []
        for i in range(z_samps.shape[0]):
            v_i = kl_prior.to_velocity(kl_prior.reconstruct(z_samps[i]))
            v_fields.append(v_i)
        v_fields = np.stack(v_fields, axis=0)
        v_mean = v_fields.mean(axis=0)
        v_std = v_fields.std(axis=0)
        v_err = np.abs(v_mean - v_true)
        post_stats[N] = {"v_mean": v_mean, "v_std": v_std, "v_err": v_err}
        global_std_max = max(global_std_max, v_std.max())
        global_err_max = max(global_err_max, v_err.max())
    global_std_max = max(global_std_max, 1e-6)
    global_err_max = max(global_err_max, 1e-6)

    def save_panel_cbar(data, fname, cmap_name, vmin_, vmax_):
        f, a = plt.subplots(1, 1, figsize=(2.5, 2.5))
        im = a.imshow(data.T, cmap=cmap_name, aspect="auto", origin="upper",
                       vmin=vmin_, vmax=vmax_)
        a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        f.savefig(os.path.join(panel_dir, fname), dpi=300,
                  bbox_inches="tight", pad_inches=0.02)
        plt.close(f)

    # True model with colorbar (same vmin/vmax as means).
    save_panel_cbar(v_true, "true_cbar.png", "terrain", vmin, vmax)

    # Second pass: plot with shared limits.
    for N in N_VALUES:
        v_mean = post_stats[N]["v_mean"]
        v_std = post_stats[N]["v_std"]
        v_err = post_stats[N]["v_err"]

        # Posterior mean with colorbar (same vmin/vmax as true model).
        save_panel_cbar(v_mean, f"mean_N{N}.png", "terrain", vmin, vmax)
        # Posterior std (OrRd, shared limits).
        save_panel_cbar(v_std, f"std_N{N}.png", "OrRd", 0, global_std_max)
        # Error |mean - true| (magma, shared limits).
        save_panel_cbar(v_err, f"error_N{N}.png", "magma", 0, global_err_max)

        # UQ calibration: scatter plot of pointwise std vs |error|.
        f, a = plt.subplots(1, 1, figsize=(2.5, 2.5))
        a.scatter(v_std.ravel(), v_err.ravel(), s=1, alpha=0.3, c="steelblue")
        lim = max(global_std_max, global_err_max) * 1.05
        a.plot([0, lim], [0, lim], "k--", linewidth=1, label="$y = x$")
        a.set_xlabel("Pointwise std", fontsize=9)
        a.set_ylabel("|Mean $-$ True|", fontsize=9)
        a.set_xlim(0, lim); a.set_ylim(0, lim)
        a.set_aspect("equal")
        a.legend(fontsize=7)
        a.tick_params(labelsize=7)
        f.savefig(os.path.join(panel_dir, f"calibration_N{N}.png"), dpi=300,
                  bbox_inches="tight", pad_inches=0.02)
        plt.close(f)

    print(f"Saved individual panels to {panel_dir}/")


if __name__ == "__main__":
    main()
