"""Train the Parihaka seismic DDPM prior (unconditional, 256 x 256 reflectivity).

The network is a four-scale
``diffusers.UNet2DModel`` (``mempost.models.unet2d.make_unet``) trained with
epsilon-prediction, a cosine-beta DDPM over 1000 timesteps, per-pixel
``Normalizer`` standardization, AdamW + OneCycleLR, gradient clipping at 1.0, and
NO EMA.  The scheduler runs with ``clip_sample=False``: the images are per-pixel
standardized (roughly ``[-17, 19]``, not ``[-1, 1]``), so the default clamp of the
predicted image would flatten every reflector.

Training images come from the Parihaka broadband-reflectivity archive.  For this
project we train the ORACLE arm only -- ``data_key="broadband_dm"`` (the truth
reflectivity), never ``lsrtm5`` (the legacy migration).  The checkpoint stores
``{model, norm_mean, norm_std, hist, data_key}`` so samples come back in physical units.

Config: ``seismic_prior_ddpm.json``
    python scripts/seismic_prior_ddpm.py --gpu_id 0
    python scripts/seismic_prior_ddpm.py --phase visualization --gpu_id 0
"""
from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from projorg import checkpointsdir, plotsdir, setup_environment, upload_to_cloud

from mempost.models.unet2d import make_unet
from mempost.seismic.data import load_images
from mempost.utils.normalizer import Normalizer
from mempost.utils.seismic import N, water_mute

CONFIG_FILE = "seismic_prior_ddpm.json"


def _save_ckpt(args, unet, normalizer, hist) -> str:
    """Atomically write the checkpoint (write .tmp then os.replace) so a concurrent eval never
    reads a half-written file -- lets us monitor memorization on the in-progress model."""
    path = os.path.join(checkpointsdir(args.experiment), "ckpt.pth")
    tmp = path + ".tmp"
    torch.save({"model": unet.state_dict(), "norm_mean": normalizer.mean,
                "norm_std": normalizer.std, "hist": hist, "data_key": args.data_key}, tmp)
    os.replace(tmp, path)
    return path


@torch.no_grad()
def sample_prior(unet, sched, normalizer: Normalizer, n_sample: int, n_steps: int,
                 device) -> np.ndarray:
    """Draw ``n_sample`` unconditional prior samples in physical units (ancestral DDPM)."""
    sched.set_timesteps(n_steps, device=device)
    x = torch.randn(n_sample, 1, N, N, device=device)
    unet.eval()
    for t in sched.timesteps:
        x = sched.step(unet(x, t.expand(n_sample)).sample, t, x).prev_sample
    return normalizer.unnormalize(x.squeeze(1).cpu()).numpy()


def _plot(args, hist: dict, real: np.ndarray, samp: np.ndarray) -> None:
    """Loss curves plus real-vs-sample reflectivity panels -> plotsdir/samples.png."""
    ncol = int(min(4, samp.shape[0], real.shape[0]))
    fig = plt.figure(figsize=(3.0 + 2.5 * ncol, 7))
    gs = fig.add_gridspec(2, 1 + ncol, width_ratios=[1.5] + [1] * ncol,
                          hspace=0.12, wspace=0.06)
    axl = fig.add_subplot(gs[:, 0])
    axl.plot(hist["step"], hist["train"], label="train")
    axl.plot(hist["step"], hist["val"], label="val")
    axl.set_yscale("log")
    axl.set_xlabel("step")
    axl.set_ylabel("denoising MSE")
    axl.legend(frameon=False)
    axl.grid(alpha=0.3)
    axl.set_title(f"{args.data_key} (Normalizer, no clamp, no EMA)")
    v = 800.0         # fixed display window: vmin=-800, vmax=800 (unnormalized units)
    for r, (name, imgs) in enumerate([("real", real), ("DDPM sample", samp)]):
        for j in range(ncol):
            ax = fig.add_subplot(gs[r, j + 1])
            im = np.asarray(imgs[j], np.float32); im = im - im.mean()          # de-mean (DC drift)
            ax.imshow(water_mute(im).T, cmap="gray", vmin=-v, vmax=v, aspect="auto")   # mute water LAST
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
    out = os.path.join(plotsdir(args.experiment), "samples.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {out}", flush=True)


def train(args, device) -> None:
    """Fit the prior and write ``checkpointsdir/ckpt.pth`` plus a sample figure."""
    Xtr = load_images(args.data_h5, args.data_key, 0, args.n_train)
    Xva = load_images(args.data_h5, args.data_key, args.n_train, args.n_train + args.n_val)
    normalizer = Normalizer(Xtr)                                   # per-pixel, fit on train only
    # Keep the (up to ~1 GB) train/val tensors on CPU and move each batch to the device per
    # step.  The CUDA index RNG stream is unchanged (indices are still drawn on-device), so
    # training stays bit-faithful to the source recipe while removing the full-dataset-resident
    # OOM risk on a 16 GB card.
    Xtr = normalizer.normalize(Xtr).unsqueeze(1)
    Xva = normalizer.normalize(Xva).unsqueeze(1)
    print(f"[{args.experiment}] normalized train {tuple(Xtr.shape)} | per-pixel std in "
          f"[{float(normalizer.std.min()):.1f}, {float(normalizer.std.max()):.1f}] | normed range "
          f"[{float(Xtr.min()):.1f}, {float(Xtr.max()):.1f}]", flush=True)

    unet = make_unet(args.base).to(device)
    sched = DDPMScheduler(num_train_timesteps=args.num_train_timesteps,
                          beta_schedule=args.beta_schedule, clip_sample=False)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.n_steps,
                                              pct_start=0.05)
    nT = sched.config.num_train_timesteps
    hist = {"step": [], "train": [], "val": []}
    t0 = time.time()

    @torch.no_grad()
    def val_loss() -> float:
        """Denoising MSE on the held-out images at a fixed noise and timestep draw."""
        tot, cnt = 0.0, 0
        g = torch.Generator(device=device).manual_seed(0)
        for s in range(0, Xva.shape[0], 8):
            x0 = Xva[s:s + 16].to(device)
            noise = torch.randn(x0.shape, generator=g, device=device)
            t = torch.randint(0, nT, (x0.shape[0],), generator=g, device=device)
            tot += F.mse_loss(unet(sched.add_noise(x0, noise, t), t).sample, noise,
                              reduction="sum").item()
            cnt += noise.numel()
        return tot / max(cnt, 1)

    unet.train()
    for step in range(args.n_steps):
        idx = torch.randint(0, Xtr.shape[0], (args.batch_size,), device=device)
        x0 = Xtr[idx.cpu()].to(device)
        noise = torch.randn_like(x0)
        t = torch.randint(0, nT, (args.batch_size,), device=device)
        loss = F.mse_loss(unet(sched.add_noise(x0, noise, t), t).sample, noise)   # epsilon target
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 200 == 0 or step == args.n_steps - 1:
            unet.eval()
            vl = val_loss()
            unet.train()
            hist["step"].append(step)
            hist["train"].append(float(loss.item()))
            hist["val"].append(vl)
            print(f"  step {step:5d}/{args.n_steps}  train {loss.item():.4f}  val {vl:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if (step + 1) % args.save_freq == 0 and step + 1 < args.n_steps:
            _save_ckpt(args, unet, normalizer, hist)                # snapshot for live monitoring
            print(f"  [snapshot @ step {step + 1}]", flush=True)
            if getattr(args, "upload", 0):                          # progress-safe: mirror each snapshot to Dropbox
                try:
                    upload_to_cloud(args)                           # rclone -> MyDropbox:mempost/{data,plots}/<exp>/
                    print(f"  [uploaded snapshot @ step {step + 1}]", flush=True)
                except Exception as exc:                            # a network hiccup must never kill a long run
                    print(f"  [upload warning @ step {step + 1}] {exc}", flush=True)

    ckpt = _save_ckpt(args, unet, normalizer, hist)
    print(f"saved checkpoint -> {ckpt}", flush=True)

    real = normalizer.unnormalize(Xtr[:args.n_sample].squeeze(1).cpu()).numpy()
    samp = sample_prior(unet, sched, normalizer, args.n_sample, args.n_sample_steps, device)
    _plot(args, hist, real, samp)


def visualize(args, device) -> None:
    """Reload the checkpoint and re-draw the sample figure (no retraining)."""
    ckpt = os.path.join(checkpointsdir(args.experiment), "ckpt.pth")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    unet = make_unet(args.base).to(device)
    unet.load_state_dict(ck["model"])
    unet.eval()
    normalizer = Normalizer.from_stats(ck["norm_mean"], ck["norm_std"])   # CPU stats, matches sampler
    sched = DDPMScheduler(num_train_timesteps=args.num_train_timesteps,
                          beta_schedule=args.beta_schedule, clip_sample=False)
    real = load_images(args.data_h5, ck["data_key"], 0, args.n_sample).numpy()
    samp = sample_prior(unet, sched, normalizer, args.n_sample, args.n_sample_steps, device)
    _plot(args, ck["hist"], real, samp)


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload", "data_h5",
                         "n_sample", "n_sample_steps", "save_freq"],
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_id}" if (torch.cuda.is_available() and args.gpu_id >= 0) else "cpu"
    )
    if args.phase == "train":
        train(args, device)
    else:
        visualize(args, device)
    if getattr(args, "upload", 0):
        upload_to_cloud(args)
