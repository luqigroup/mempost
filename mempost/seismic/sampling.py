"""Unconditional sampling from a trained seismic DDPM prior (DDIM or DDPM).

Loads a checkpoint written by ``scripts/seismic_prior_ddpm.py`` and draws samples in physical
reflectivity units.  Two samplers:

``ddim``   deterministic (eta = 0): the reverse ODE maps each noise seed to the nearest learned
           mode, so a memorized prior lands ON training images -- the cleanest memorization probe.
``ddpm``   ancestral / stochastic: injects per-step noise, adding spread (a realism cross-check).

Both run with ``clip_sample=False`` (the images are per-pixel standardized, not [-1, 1]; the default
clamp would flatten every reflector).  Reused by the memorization eval and, later, the Born DPS.
"""
from __future__ import annotations

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from mempost.models.unet2d import make_unet
from mempost.utils.normalizer import Normalizer
from mempost.utils.seismic import N


def load_prior(ckpt_path: str, base: int, device) -> tuple:
    """Return ``(unet, normalizer, checkpoint_dict)`` for a trained prior.

    Normalization statistics are kept on CPU so ``normalizer.unnormalize`` matches the CPU sampler.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    unet = make_unet(base).to(device)
    unet.load_state_dict(ck["model"])
    unet.eval()
    normalizer = Normalizer.from_stats(ck["norm_mean"], ck["norm_std"])
    return unet, normalizer, ck


@torch.no_grad()
def sample_prior(unet, normalizer: Normalizer, n: int, sampler: str, n_steps: int,
                 num_train_timesteps: int, beta_schedule: str, device,
                 chunk: int = 8, seed: int | None = None) -> np.ndarray:
    """Draw ``n`` unconditional samples in physical units -> (n, N, N) float32 array.

    Args:
        sampler: ``"ddim"`` (deterministic) or ``"ddpm"`` (ancestral).
        n_steps: Number of reverse (inference) steps.
        seed: If given, seeds a device generator so the noise seeds are reproducible.
    """
    if sampler not in ("ddim", "ddpm"):
        raise ValueError(f"sampler must be 'ddim' or 'ddpm'; got {sampler!r}")
    Sched = DDIMScheduler if sampler == "ddim" else DDPMScheduler
    sched = Sched(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
                  clip_sample=False)
    sched.set_timesteps(n_steps, device=device)
    unet.eval()
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    out = []
    for i in range(0, n, chunk):
        b = min(chunk, n - i)
        x = torch.randn(b, 1, N, N, device=device, generator=gen)
        for t in sched.timesteps:
            # generator=gen makes the DDPM ancestral noise reproducible under `seed`
            # (DDIM eta=0 ignores it).
            x = sched.step(unet(x, t.expand(b)).sample, t, x, generator=gen).prev_sample
        out.append(normalizer.unnormalize(x.squeeze(1).cpu()).numpy())
    return np.concatenate(out, 0)
