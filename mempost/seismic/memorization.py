"""Score memorization of a seismic prior with the paper's nearest-neighbor ratio, WITH controls.

For each query image, ``r = d1 / mean(d2..dk)`` -- distance to the nearest reference image over the
mean distance to the next ``k-1`` -- in **per-pixel-standardized** image space (the space the model
operates in, the image analogue of the paper's normalized-KL metric).  ``r`` near 0 means the query
sits on top of one reference image.  We reuse the estimator core from
``mempost.utils.memorization_metrics`` (the KL experiment's), flattening images to vectors in
**float64** (the 65536-dim distances lose accuracy in float32 exactly where ``d1 -> 0``).

A memorization *rate* is uninterpretable alone: adjacent Parihaka patches are near-duplicates, so a
low ``r`` for generated samples could be data redundancy rather than model memorization.  The claim
therefore rests on a CONTRAST computed in the same metric space:

  - **samples-vs-train**   -- the quantity of interest.
  - **leave-one-out train null** (``exclude_self=True``) -- how memorized the training set already is
    against itself (the data-redundancy floor).
  - **held-out-reals null** -- how close never-trained real reflectivities sit to the training set
    (the generalization floor).

Memorization is claimed only when the sample ``r``-distribution separates BELOW both nulls.
"""
from __future__ import annotations

import torch

from mempost.utils.memorization_metrics import k_nearest_indices


def standardize_flat(images: torch.Tensor, normalizer) -> torch.Tensor:
    """Per-pixel standardize ``(n, N, N)`` physical images -> ``(n, N*N)`` float64.

    float64 keeps the nearest-neighbor distance accurate in the memorized regime (``d1 -> 0``); at
    65536 dims the float32 ``||a||^2 + ||b||^2 - 2ab`` path loses ~0.1 absolute even for near-identical
    vectors.
    """
    z = normalizer.normalize(images.float())
    return z.reshape(z.shape[0], -1).double()


def ratios_vs_reference(query_imgs: torch.Tensor, reference_imgs: torch.Tensor, normalizer,
                        k_neighbors: int = 50, exclude_self: bool = False) -> tuple:
    """Per-query ``r = d1 / mean(d2..dk)`` against a reference image set.

    Args:
        query_imgs: ``(n_q, N, N)`` physical images to score.
        reference_imgs: ``(n_ref, N, N)`` physical reference set (e.g. the training images).
        exclude_self: If the query IS the reference set, drop the zero-distance self-match so ``d1``
            is the nearest OTHER image (leave-one-out).

    Returns:
        ``(ratios, d1, nn_idx)`` -- ``ratios`` is NaN where the denominator is empty (too few
        references); ``d1``/``nn_idx`` are the nearest-reference distance and index per query.
    """
    q = standardize_flat(query_imgs, normalizer)
    ref = standardize_flat(reference_imgs, normalizer)
    off = 1 if exclude_self else 0
    k = min(k_neighbors + off, ref.shape[0])
    n_q = q.shape[0]
    if k <= off:                                        # not enough references to define d1
        nan = torch.full((n_q,), float("nan"), dtype=torch.float64)
        return nan, nan, torch.zeros(n_q, dtype=torch.long)

    dists, idx = k_nearest_indices(ref, q, k_neighbors=k)   # ascending (n_q, k)
    d1 = dists[:, off]
    nn_idx = idx[:, off]
    denom = dists[:, off + 1:]
    if denom.shape[1] >= 1:
        ratios = d1 / denom.mean(dim=1)
    else:
        ratios = torch.full((n_q,), float("nan"), dtype=d1.dtype)
    return ratios, d1, nn_idx


def _stats(ratios: torch.Tensor) -> dict:
    r = ratios[torch.isfinite(ratios)]
    if r.numel() == 0:
        return {"mean": None, "median": None, "min": None}
    return {"mean": float(r.mean()), "median": float(r.median()), "min": float(r.min())}


def rates(ratios: torch.Tensor, thresholds: tuple[float, ...]) -> dict:
    """Fraction of finite ratios below each threshold (JSON-safe: None when undefined)."""
    r = ratios[torch.isfinite(ratios)]
    if r.numel() == 0:
        return {str(t): None for t in thresholds}
    return {str(t): float((r < t).float().mean()) for t in thresholds}


def bootstrap_rate_ci(ratios: torch.Tensor, threshold: float = 0.5, n_boot: int = 1000,
                      seed: int = 0) -> list:
    """95% bootstrap CI for the memorized-rate at ``threshold`` (resampling the finite ratios)."""
    r = ratios[torch.isfinite(ratios)]
    if r.numel() == 0:
        return [None, None]
    g = torch.Generator().manual_seed(seed)
    n = r.numel()
    boot = torch.empty(n_boot)
    for i in range(n_boot):
        sel = torch.randint(0, n, (n,), generator=g)
        boot[i] = (r[sel] < threshold).double().mean()
    return [float(torch.quantile(boot, 0.025)), float(torch.quantile(boot, 0.975))]


def summarize(ratios: torch.Tensor, d1: torch.Tensor, thresholds: tuple[float, ...],
              n_ref: int, k_neighbors: int) -> dict:
    """JSON-safe summary block for one query set (ratio stats, rates, bootstrap CI, d1)."""
    d1f = d1[torch.isfinite(d1)]
    return {
        "n_query": int(ratios.shape[0]),
        "n_ref": int(n_ref),
        "k": int(min(k_neighbors, n_ref)),
        "ratio": _stats(ratios),
        "rates": rates(ratios, thresholds),
        "rate_ci_0.5": bootstrap_rate_ci(ratios, 0.5),
        "d1_mean": float(d1f.mean()) if d1f.numel() else None,
        "d1_min": float(d1f.min()) if d1f.numel() else None,
    }


def memorization_report(train_imgs: torch.Tensor, samples: torch.Tensor, normalizer,
                        k_neighbors: int = 50,
                        thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)) -> tuple:
    """Samples-vs-train memorization summary (the quantity of interest).

    Returns ``(report_dict, ratios, d1, nn_idx)``.  Compute the nulls separately with
    ``ratios_vs_reference`` (leave-one-out train, held-out reals) for the interpretive contrast.
    """
    ratios, d1, nn_idx = ratios_vs_reference(samples, train_imgs, normalizer, k_neighbors=k_neighbors)
    report = summarize(ratios, d1, thresholds, n_ref=train_imgs.shape[0], k_neighbors=k_neighbors)
    return report, ratios, d1, nn_idx
