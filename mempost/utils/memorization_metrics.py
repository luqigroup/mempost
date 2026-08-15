import torch


def ratio_mem_metric(
    x_train: torch.Tensor,
    samples: torch.Tensor,
    ratio: float = 0.5,
    k_neighbors: int = 50,
    batch_size: int = 500,
) -> float:
    """Nearest-neighbor ratio memorization metric.

    For each generated sample, computes d1 / avg(d2..dk) where d1 is the
    distance to the closest training sample and d2..dk are distances to the
    2nd through k-th closest. A sample is considered memorized when this
    ratio is below the threshold.

    Args:
        x_train: Flattened training data (N_train, D).
        samples: Flattened generated samples (N_gen, D).
        ratio: Threshold ratio below which a sample is memorized.
        k_neighbors: Number of nearest neighbors to average over.
        batch_size: Process training data in chunks to avoid OOM.

    Returns:
        Fraction of generated samples considered memorized.
    """
    n_gen = samples.shape[0]
    k = min(k_neighbors, x_train.shape[0])
    topk_dists = torch.full((n_gen, k), float("inf"), device=samples.device)

    for i in range(0, x_train.shape[0], batch_size):
        batch = x_train[i : i + batch_size]
        dists = torch.cdist(samples, batch)
        combined = torch.cat([topk_dists, dists], dim=1)
        topk_dists = combined.topk(k=k, dim=1, largest=False).values

    d1 = topk_dists[:, 0]
    avg_dk = topk_dists[:, 1:].mean(dim=1)
    memorized = d1 <= ratio * avg_dk
    return memorized.float().mean().item()


def ratio_mem_values(
    x_train: torch.Tensor,
    samples: torch.Tensor,
    k_neighbors: int = 50,
    batch_size: int = 500,
) -> torch.Tensor:
    """Per-sample d1/avg(d2..dk) ratio values.

    Args:
        x_train: Flattened training data (N_train, D).
        samples: Flattened generated samples (N_gen, D).
        k_neighbors: Number of nearest neighbors.
        batch_size: Chunk size for distance computation.

    Returns:
        Ratio values (N_gen,). Lower = more memorized.
    """
    n_gen = samples.shape[0]
    k = min(k_neighbors, x_train.shape[0])
    topk_dists = torch.full((n_gen, k), float("inf"), device=samples.device)

    for i in range(0, x_train.shape[0], batch_size):
        batch = x_train[i : i + batch_size]
        dists = torch.cdist(samples, batch)
        combined = torch.cat([topk_dists, dists], dim=1)
        topk_dists = combined.topk(k=k, dim=1, largest=False).values

    d1 = topk_dists[:, 0]
    avg_dk = topk_dists[:, 1:].mean(dim=1)
    return d1 / avg_dk


def k_nearest_indices(
    x_train: torch.Tensor,
    samples: torch.Tensor,
    k_neighbors: int = 50,
    batch_size: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find k-nearest training indices and distances for each sample.

    Args:
        x_train: Flattened training data (N_train, D).
        samples: Flattened generated samples (N_gen, D).
        k_neighbors: Number of nearest neighbors.
        batch_size: Chunk size for distance computation.

    Returns:
        Tuple of (distances (N_gen, k), indices (N_gen, k)) sorted by distance.
    """
    n_gen = samples.shape[0]
    k = min(k_neighbors, x_train.shape[0])
    topk_dists = torch.full((n_gen, k), float("inf"), device=samples.device)
    topk_indices = torch.zeros(
        (n_gen, k), dtype=torch.long, device=samples.device
    )

    for i in range(0, x_train.shape[0], batch_size):
        batch = x_train[i : i + batch_size]
        dists = torch.cdist(samples, batch)
        # Offset indices by batch start position.
        batch_indices = torch.arange(
            i, i + batch.shape[0], device=samples.device
        ).unsqueeze(0).expand(n_gen, -1)
        combined_dists = torch.cat([topk_dists, dists], dim=1)
        combined_indices = torch.cat([topk_indices, batch_indices], dim=1)
        _, sel = combined_dists.topk(k=k, dim=1, largest=False)
        topk_dists = combined_dists.gather(1, sel)
        topk_indices = combined_indices.gather(1, sel)

    return topk_dists, topk_indices


def nearest_neighbor_distances(
    x_train: torch.Tensor,
    samples: torch.Tensor,
    batch_size: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute nearest-neighbor distances and indices.

    Args:
        x_train: Flattened training data (N_train, D).
        samples: Flattened generated samples (N_gen, D).
        batch_size: Process training data in chunks to avoid OOM.

    Returns:
        Tuple of (distances, indices) for nearest training sample per
        generated sample.
    """
    n_gen = samples.shape[0]
    min_dists = torch.full((n_gen,), float("inf"), device=samples.device)
    min_indices = torch.zeros(n_gen, dtype=torch.long, device=samples.device)

    for i in range(0, x_train.shape[0], batch_size):
        batch = x_train[i : i + batch_size]
        dists = torch.cdist(samples, batch)
        batch_min_dists, batch_min_idx = dists.min(dim=1)
        update_mask = batch_min_dists < min_dists
        min_dists[update_mask] = batch_min_dists[update_mask]
        min_indices[update_mask] = batch_min_idx[update_mask] + i

    return min_dists, min_indices
