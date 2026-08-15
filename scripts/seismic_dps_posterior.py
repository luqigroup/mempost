#!/usr/bin/env python
"""Diffusion Posterior Sampling (DPS) on the seismic linearized-Born operator, for the MEMORIZATION
study: condition a trained prior (N training images) on held-out shot data and read how strongly the
posterior COLLAPSES onto the training set as N shrinks.

The memorization read (mempost.seismic.memorization) is the r = d1/mean(d2..dk) nearest-neighbor
ratio on the posterior samples, the same metric used on the prior.

Guidance chain (normalized model space -> physical -> data), frozen score (eps detached):
  eps=UNet(x_t,t).detach();  x0=(x_t - sqrt(1-abar) eps)/sqrt(abar)              [normalized]
  dm = x0*(std+1e-5)+mean;  dm~=mute(dm, end=16, len=6)                          [PHYSICAL]
  r_s = d_obs[s] - born_one(dm~, s);  rnorm=sqrt(sum_s ||r_s||^2)               [per chain]
  x_{t-1} = ancestral_step(eps,t,x_t) - AdaGrad(xi * grad rnorm)                 [Chung DPS + AdaGrad]

Devito Born solves are CPU-only; the frozen UNet score runs on GPU if free. Run selfcheck first,
under OMP_NUM_THREADS=1:

  OMP_NUM_THREADS=1 python scripts/seismic_dps_posterior.py --phase selfcheck
  OMP_NUM_THREADS=1 python scripts/seismic_dps_posterior.py --phase smoke --N 2048
  OMP_NUM_THREADS=1 python scripts/seismic_dps_posterior.py --phase run   --N 2048
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import h5py
import numpy as np
import torch
from diffusers import DDPMScheduler

from mempost.models.unet2d import make_unet
from mempost.utils.seismic import water_mute, MUTE_END, TAPER, N
from mempost.utils.born_imager import SparseBornImager, parihaka_background_m0
from mempost.seismic.sampling import load_prior
from mempost.seismic.data import load_images, resolve_data_path
from mempost.seismic.memorization import memorization_report, ratios_vs_reference, standardize_flat

# --- acquisition (configs/seismic_laundering.json, verified) ---
NSRC, NREC, TN_MS, F0_KHZ = 12, 24, 2000.0, 0.03
SIGMA, SPACING, SPACE_ORDER, NBL, WATER_ZERO_DEPTH = 4000.0, (20.0, 12.5), 16, 40, 10
CKPT_ROOT = "data/checkpoints"
CKPT_GLOB = ("seismic_prior_parihaka_oracle_data_key-broadband_dm_n_train-{N}_n_val-8_base-64_lr-0.0002_"
             "weight_decay-0.0_beta_schedule-squaredcos_cap_v2_num_train_timesteps-1000_batch_size-8_"
             "n_steps-50000_seed-0")
EVAL_H5, TRAIN_H5 = "data/seismic/dataset_eval.h5", "data/seismic/dataset_train.h5"


def build_imager():
    m0 = parihaka_background_m0(nlat=N, ndep=N, water_zero_depth=WATER_ZERO_DEPTH)
    return SparseBornImager(m0, spacing=SPACING, nsrc=NSRC, nrec=NREC, tn=TN_MS, f0=F0_KHZ,
                            space_order=SPACE_ORDER, nbl=NBL, pad_mode="zero", device="cpu")


def find_ckpt(Nt):
    hits = glob.glob(os.path.join(CKPT_ROOT, CKPT_GLOB.format(N=Nt), "ckpt.pth"))
    if not hits:
        raise FileNotFoundError(f"no ckpt for N={Nt} under {CKPT_ROOT} (glob {CKPT_GLOB.format(N=Nt)})")
    return hits[0]


def load_prior_dps(ckpt, unet_dev):
    """Return (unet on unet_dev, mean(1,1,N,N), std(1,1,N,N), normalizer). denorm = x*(std+1e-5)+mean."""
    unet, normalizer, _ = load_prior(ckpt, 64, unet_dev)
    mean = normalizer.mean.reshape(1, 1, N, N).float()
    std = normalizer.std.reshape(1, 1, N, N).float()
    return unet, mean, std, normalizer


def denorm(x0_norm, mean, std):
    """normalized model space -> PHYSICAL reflectivity (matches sample_prior / Normalizer eps=1e-5)."""
    return x0_norm * (std + 1e-5) + mean


def load_truth_shots(row):
    """d_obs = eval shots[row] (NSRC,nt,nrec); truth = water-muted eval broadband_dm[row] (N*N,)."""
    with h5py.File(resolve_data_path(EVAL_H5), "r") as f:
        d_obs = torch.tensor(np.asarray(f["shots"][row]), dtype=torch.float32)
    truth = load_images(EVAL_H5, "broadband_dm", row, row + 1)[0].reshape(N * N).numpy()
    return d_obs, truth


TRUTH_NOISE_SEED = 12345   # fixed so the synthetic d_obs is IDENTICAL across N (all N share the truth)


def make_train_truth(im, idx, snr):
    """In-distribution truth = a SINGLE real training image (physical reflectivity) at index ``idx``.
    Because the training sets are nested (broadband_dm[0:N]), any idx < smallest-N is in EVERY prior's
    training set, so a prior that memorized this image can fit the data by reproducing it. NO averaging
    (an average of reflectivities is not physical geology). Synthetic shots d_obs = Born(mute(truth)) +
    noise, calibrated so the per-sample data SNR = ``snr`` (the global SIGMA is tuned to the sharper
    real eval shots); the true model then sits at misfit/noise_floor ~ 1. Returns (d_obs, truth,
    sigma_eff)."""
    truth_img = load_images(TRAIN_H5, "broadband_dm", idx, idx + 1)[0].float()   # (N,N) physical
    truth_t = im.mute_op(truth_img.reshape(1, 1, N, N), end=MUTE_END, length=TAPER)
    with torch.no_grad():
        clean = torch.stack([im._born_one(truth_t, s) for s in range(NSRC)])    # (NSRC,nt,nrec)
    sigma = float((clean ** 2).mean().item() / snr) ** 0.5                       # noise calibrated to target SNR
    g = torch.Generator().manual_seed(TRUTH_NOISE_SEED)
    d_obs = (clean + sigma * torch.randn(clean.shape, generator=g)).float()
    print(f"[truth] train-img idx={idx} | synthetic d_obs, target SNR={snr} -> sigma={sigma:.1f}", flush=True)
    return d_obs, truth_img.reshape(-1).numpy(), sigma


# ---------------------------------------------------------------------------------------------------
# the differentiable data misfit (CPU): denorm -> mute -> Born(J_s) -> residual vs observed shots
# ---------------------------------------------------------------------------------------------------
def seismic_rnorm(x0_norm, im, mean, std, d_obs, shot_ids):
    """Per-chain rnorm = sqrt(sum_s ||d_obs[s] - J_s(mute(denorm(x0_c)))||^2), differentiable wrt x0_norm."""
    dm = denorm(x0_norm, mean, std)                              # (nchain,1,N,N) physical
    dm = im.mute_op(dm, end=MUTE_END, length=TAPER)             # water mute BEFORE Born (self-adjoint diag)
    out = []
    for c in range(dm.shape[0]):
        acc = torch.zeros((), dtype=torch.float64)             # float64 kills sum cancellation
        dm_c = dm[c:c + 1]
        for s in shot_ids:
            r = d_obs[int(s)] - im._born_one(dm_c, int(s))     # (nt,nrec) differentiable
            acc = acc + (r.double() ** 2).sum()
        out.append(torch.sqrt(acc.clamp_min(1e-30)).float())
    return torch.stack(out)                                     # (nchain,)


# ---------------------------------------------------------------------------------------------------
# self-checks -- hard correctness gate (run before ANY expensive compute; OMP_NUM_THREADS=1)
# ---------------------------------------------------------------------------------------------------
def self_checks(im, mean, std, d_obs, x0_base):
    print("=== SELF-CHECKS (correctness gate; run with OMP_NUM_THREADS=1) ===", flush=True)
    ok = True
    x = torch.randn(2, 1, N, N)
    rt = (denorm(x, mean, std) - mean) / (std + 1e-5)
    e1 = float((rt - x).abs().max())
    print(f"[1] round-trip denorm: max|err|={e1:.2e}  {'PASS' if e1 < 1e-4 else 'FAIL'}", flush=True); ok &= e1 < 1e-4

    adr = im.adjoint_dot_ratio()
    print(f"[2a] Born adjoint_dot_ratio={adr:.6f}  {'PASS' if abs(adr - 1) < 1e-3 else 'FAIL'}", flush=True); ok &= abs(adr - 1) < 1e-3
    x0 = x0_base.clone().detach().requires_grad_(True)
    ids = [0, 5]
    r0 = float(seismic_rnorm(x0.detach(), im, mean, std, d_obs, ids).item())
    r = seismic_rnorm(x0, im, mean, std, d_obs, ids)
    g = torch.autograd.grad(r.sum(), x0)[0]
    torch.manual_seed(1); h = torch.randn(1, 1, N, N); h = h / h.norm()
    eps = float(os.environ.get("FD_EPS", "1e-2"))
    with torch.no_grad():
        rp = float(seismic_rnorm((x0 + eps * h).detach(), im, mean, std, d_obs, ids).item())
        rm = float(seismic_rnorm((x0 - eps * h).detach(), im, mean, std, d_obs, ids).item())
    fd = (rp - rm) / (2 * eps); ana = float((g * h).sum()); ratio = fd / (ana + 1e-30)
    soft = (fd * ana > 0) and (0.5 < abs(ratio) < 2.0)
    print(f"[2b] FD sanity (float32-Born-limited) @ rnorm={r0:.3e}: fd={fd:.3e} ana={ana:.3e} ratio={ratio:.2f}  "
          f"{'OK' if soft else 'SUSPECT'}", flush=True); ok &= soft

    dm = denorm(x0.detach().requires_grad_(False), mean, std).detach().requires_grad_(True)
    dmm = im.mute_op(dm, end=MUTE_END, length=TAPER)
    acc = torch.zeros(())
    for s in ids:
        rr = d_obs[s] - im._born_one(dmm[0:1], int(s)); acc = acc + (rr ** 2).sum()
    rn = torch.sqrt(acc.clamp_min(1e-30))
    g_dm = torch.autograd.grad(rn, dm)[0]
    g_expect = (std + 1e-5) * g_dm
    e3 = float((g - g_expect).abs().max() / (g.abs().max() + 1e-30))
    print(f"[3] scaling identity: rel max|err|={e3:.2e}  {'PASS' if e3 < 1e-3 else 'FAIL'}", flush=True); ok &= e3 < 1e-3
    print(f"=== SELF-CHECKS {'ALL PASS' if ok else 'FAILED -- DO NOT RUN COMPUTE'} ===", flush=True)
    return ok


# ---------------------------------------------------------------------------------------------------
# the DPS reverse loop (ancestral DDPM, frozen score, AdaGrad-preconditioned Chung guidance)
# ---------------------------------------------------------------------------------------------------
def dps_posterior(unet, unet_dev, im, mean, std, d_obs, nchain, nrev, xi, nsub, guide_from, uchunk,
                  precond="adagrad", noise_floor_shot=None, Ztrain=None, truth_phys=None):
    """Returns (post (nchain, N*N), diag). diag holds per-guided-step convergence traces:
    data_fid (shot misfit / noise floor), prior_d1 (dist of running x0 to nearest TRAIN image, the
    memorized-prior NLL proxy), corr_truth (scale-free corr of x0 to the true model), and the
    prior-move vs data-move magnitudes -- the direct read on whether the likelihood drowns the prior."""
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    sched.set_timesteps(nrev)
    abars = sched.alphas_cumprod
    x = torch.randn(nchain, 1, N, N)
    Gacc = torch.zeros(nchain, 1, N, N)
    adg_eps = 1e-8
    steps = sched.timesteps
    n_skip = int(guide_from * len(steps))
    perm = np.random.permutation(NSRC)
    diag = {k: [] for k in ("t", "data_fid", "prior_d1", "corr_truth", "prior_move", "data_move")}
    for k, t in enumerate(steps):
        ab = float(abars[t])
        eps = torch.empty(nchain, 1, N, N)
        with torch.no_grad():
            for i in range(0, nchain, uchunk):
                xb = x[i:i + uchunk].to(unet_dev)
                eps[i:i + uchunk] = unet(xb, t.expand(xb.shape[0]).to(unet_dev)).sample.detach().cpu()
        if k < n_skip:
            with torch.no_grad():
                x = sched.step(eps, t, x).prev_sample
            continue
        x = x.detach().requires_grad_(True)
        x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)         # eps detached -> graph linear in x
        j = (k - n_skip) * nsub
        shot_ids = [int(perm[(j + s) % NSRC]) for s in range(nsub)]
        rnorm = seismic_rnorm(x0, im, mean, std, d_obs, shot_ids)
        g = torch.autograd.grad(rnorm.sum(), x)[0]
        with torch.no_grad():
            xprev = sched.step(eps, t, x.detach()).prev_sample
            if precond == "adagrad":
                Gacc = Gacc + g ** 2
                upd = xi * g / (torch.sqrt(Gacc) + adg_eps)
            else:                                              # plain Chung DPS zeta/||r|| step
                upd = (xi / rnorm.view(-1, 1, 1, 1)) * g
            prior_move = (xprev - x.detach())                  # what the score/ancestral step moves
            x = xprev - upd
            # ---- convergence diagnostics (all in normalized model space) ----
            x0f = x0.detach().reshape(nchain, -1)
            diag["t"].append(int(t))
            if noise_floor_shot is not None:
                diag["data_fid"].append(float((rnorm.detach() ** 2).mean()) / (len(shot_ids) * noise_floor_shot))
            if Ztrain is not None:
                diag["prior_d1"].append(float(torch.cdist(x0f, Ztrain).min(dim=1).values.mean()))
            if truth_phys is not None:
                dm0 = denorm(x0.detach(), mean, std).reshape(nchain, -1)     # physical reflectivity
                a = dm0 - dm0.mean(1, keepdim=True); b = truth_phys - truth_phys.mean()
                diag["corr_truth"].append(float(((a * b).sum(1) / (a.norm(dim=1) * b.norm() + 1e-12)).mean()))
            diag["prior_move"].append(float(prior_move.reshape(nchain, -1).norm(dim=1).mean()))
            diag["data_move"].append(float(upd.reshape(nchain, -1).norm(dim=1).mean()))
    dm = denorm(x.detach(), mean, std).squeeze(1)
    return dm.reshape(nchain, N * N).numpy(), {k: np.asarray(v) for k, v in diag.items()}


# ---------------------------------------------------------------------------------------------------
# memorization-in-the-posterior read (same r-metric + nulls as the prior eval)
# ---------------------------------------------------------------------------------------------------
def score_memorization(post, normalizer, Nt, train_imgs=None):
    samples = torch.from_numpy(post.reshape(-1, N, N).astype(np.float32))
    if train_imgs is None:
        train_imgs = load_images(TRAIN_H5, "broadband_dm", 0, Nt)
    k = min(50, Nt)
    report, r_samp, d1, nn_idx = memorization_report(train_imgs, samples, normalizer,
                                                     k_neighbors=k, thresholds=(0.3, 0.5, 0.7))
    r_loo, _, _ = ratios_vs_reference(train_imgs, train_imgs, normalizer, k_neighbors=k, exclude_self=True)
    heldout = load_images(TRAIN_H5, "broadband_dm", 4000 - 128, 4000)
    r_held, _, _ = ratios_vs_reference(heldout, train_imgs, normalizer, k_neighbors=k)
    return dict(r_samp=r_samp.numpy(), nn_idx=nn_idx.numpy(), d1=d1.numpy(),
                r_loo=r_loo.numpy(), r_held=r_held.numpy(), report=report, train_imgs=train_imgs.numpy())


def posterior_misfit(post, im, d_obs):
    pm = torch.tensor(post.mean(0).reshape(1, 1, N, N), dtype=torch.float32)
    pmm = im.mute_op(pm, end=MUTE_END, length=TAPER)
    with torch.no_grad():
        return float(sum(((d_obs[s] - im._born_one(pmm, int(s))) ** 2).sum() for s in range(NSRC)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["selfcheck", "smoke", "run"], default="selfcheck")
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--row", type=int, default=300)            # held-out eval truth
    ap.add_argument("--unet-device", default="auto")
    ap.add_argument("--nchain", type=int, default=32)
    ap.add_argument("--nrev", type=int, default=30)
    ap.add_argument("--xi", type=float, default=0.5)
    ap.add_argument("--nsub", type=int, default=1)
    ap.add_argument("--guide-from", type=float, default=0.25)
    ap.add_argument("--uchunk", type=int, default=8)
    ap.add_argument("--precond", choices=["adagrad", "none"], default="adagrad")
    ap.add_argument("--truth-mode", choices=["eval-row", "train-img"], default="eval-row")
    ap.add_argument("--truth-idx", type=int, default=0)        # train-img: which training image is the truth
    ap.add_argument("--snr", type=float, default=10.0)         # train-img: target per-sample data SNR
    ap.add_argument("--seed", type=int, default=0)             # DPS init + shot order (reproducible)
    ap.add_argument("--skip-mem", action="store_true")         # skip memorization scoring (saves samples+corr; memory-light)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    unet_dev = args.unet_device
    if unet_dev == "auto":
        unet_dev = "cpu"
        if torch.cuda.is_available():
            try:
                unet_dev = "cuda" if torch.cuda.mem_get_info()[0] / 1e9 > 3.0 else "cpu"
            except Exception:
                unet_dev = "cpu"
    print(f"[dps] phase={args.phase} N={args.N} unet_device={unet_dev} (Born always CPU) | "
          f"nchain={args.nchain} nrev={args.nrev} xi={args.xi} nsub={args.nsub} "
          f"truth={args.truth_mode} seed={args.seed}", flush=True)

    im = build_imager()
    if args.truth_mode == "train-img":
        d_obs, truth, sigma = make_train_truth(im, args.truth_idx, args.snr)
        truth_tag = f"trainidx{args.truth_idx}"
    else:
        d_obs, truth = load_truth_shots(args.row)
        sigma = SIGMA
        truth_tag = f"row{args.row}"
    unet, mean, std, normalizer = load_prior_dps(find_ckpt(args.N), unet_dev)

    if args.phase == "selfcheck":
        x0_base = 0.3 * (torch.tensor(truth.reshape(1, 1, N, N), dtype=torch.float32) - mean) / (std + 1e-5)
        self_checks(im, mean, std, d_obs, x0_base)
        return

    noise_floor = NSRC * im.nt * im.nrec * sigma ** 2
    noise_floor_shot = im.nt * im.nrec * sigma ** 2
    train_imgs = load_images(TRAIN_H5, "broadband_dm", 0, args.N)
    Ztrain = standardize_flat(train_imgs, normalizer).float()                       # (N, N*N) normalized
    truth_phys = torch.tensor(truth.reshape(-1), dtype=torch.float32)               # physical reflectivity
    t0 = time.time()
    post, diag = dps_posterior(unet, unet_dev, im, mean, std, d_obs, args.nchain, args.nrev, args.xi,
                               args.nsub, args.guide_from, args.uchunk, args.precond,
                               noise_floor_shot=noise_floor_shot, Ztrain=Ztrain, truth_phys=truth_phys)
    mis = posterior_misfit(post, im, d_obs)
    rc = float(np.corrcoef(post.mean(0), truth)[0, 1])
    mv, dv = float(diag["prior_move"].mean()), float(diag["data_move"].mean())
    print(f"[{args.phase}] {time.time()-t0:.0f}s | posterior-mean misfit/noise_floor = {mis/noise_floor:.2f} "
          f"(want ~1) | corr(post-mean, truth) = {rc:.3f}", flush=True)
    print(f"[balance] mean prior_move={mv:.3f} data_move={dv:.3f}  data/prior={dv/(mv+1e-12):.1f}x "
          f"(>>1 => likelihood dominates) | prior_d1 {diag['prior_d1'][0]:.0f}->{diag['prior_d1'][-1]:.0f} "
          f"| corr_truth {diag['corr_truth'][0]:.2f}->{diag['corr_truth'][-1]:.2f}", flush=True)

    os.makedirs("plots/seismic_dps", exist_ok=True)
    if args.phase == "smoke":
        dg = os.path.join("plots/seismic_dps",
                          f"dps_diag_N{args.N}_{truth_tag}_xi{args.xi}_gf{args.guide_from}_nc{args.nchain}.npz")
        np.savez(dg, misfit_ratio=mis / noise_floor, corr=rc, xi=args.xi, guide_from=args.guide_from,
                 N=args.N, **{f"diag_{k}": v for k, v in diag.items()})
        print(f"[smoke] saved diag -> {dg}", flush=True)
        return

    if args.skip_mem:                          # light path: samples + corr only (for landscapes/summary), skip mem scoring
        out = os.path.join("plots/seismic_dps", f"dps_N{args.N}_{truth_tag}.npz")
        np.savez_compressed(out, samples=post.reshape(-1, N, N).astype(np.float32),
                            post_mean=post.mean(0).reshape(N, N).astype(np.float32),
                            post_std=post.std(0).reshape(N, N).astype(np.float32),
                            truth=truth.reshape(N, N).astype(np.float32),
                            misfit_ratio=mis / noise_floor, corr=rc, N=args.N, row=args.row,
                            xi=args.xi, nrev=args.nrev, nchain=args.nchain,
                            **{f"diag_{k}": v for k, v in diag.items()})
        print(f"[run] N={args.N}: corr={rc:.3f} misfit/floor={mis/noise_floor:.2f} saved (skip-mem) -> {out}", flush=True)
        return

    mem = score_memorization(post, normalizer, args.N, train_imgs=train_imgs)
    r = mem["r_samp"]
    print(f"[run] N={args.N}: r_med={np.median(r):.3f} r_min={np.min(r):.3f} "
          f"frac(r<0.3)={np.mean(r < 0.3):.3f} frac(r<0.5)={np.mean(r < 0.5):.3f} | "
          f"null_loo={np.median(mem['r_loo']):.3f} null_held={np.median(mem['r_held']):.3f}", flush=True)
    outdir = "plots/seismic_dps"
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"dps_N{args.N}_{truth_tag}.npz")
    np.savez_compressed(out, samples=post.reshape(-1, N, N).astype(np.float32),
                        post_mean=post.mean(0).reshape(N, N).astype(np.float32),
                        post_std=post.std(0).reshape(N, N).astype(np.float32),
                        truth=truth.reshape(N, N).astype(np.float32),
                        misfit_ratio=mis / noise_floor, corr=rc, N=args.N, row=args.row,
                        xi=args.xi, nrev=args.nrev, nchain=args.nchain,
                        r_samp=mem["r_samp"], nn_idx=mem["nn_idx"], d1=mem["d1"],
                        r_loo=mem["r_loo"], r_held=mem["r_held"], train_imgs=mem["train_imgs"],
                        **{f"diag_{k}": v for k, v in diag.items()})
    print(f"[run] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
