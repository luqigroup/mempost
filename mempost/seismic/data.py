"""Load water-muted Parihaka reflectivity images from the archive h5.

Canonical loader shared by the eval / DPS paths (the training script keeps its own copy).  Selection
is deterministic: rows are the completed images (``done`` flag, or ``recon_done`` for ``lsrtm5``),
sorted, sliced ``[a, b)`` -- so the eval reloads exactly the images a training run of size ``n_train``
was shown by calling ``load_images(h5, key, 0, n_train)``.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from mempost.utils.seismic import MUTE_END, TAPER, water_mute

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_data_path(path: str) -> str:
    """Local path for a data h5; downloads it on first use if it is a released artifact."""
    from mempost.download import ensure, REGISTRY

    path = os.path.expanduser(path)
    rel = os.path.relpath(path, _REPO) if os.path.isabs(path) else path
    if rel in REGISTRY:
        return ensure(rel)
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def load_images(h5_path: str, data_key: str, a: int, b: int) -> torch.Tensor:
    """Rows ``[a, b)`` of the completed ``data_key`` images, water-muted -> (n, N, N) float32 tensor.

    ``lsrtm5`` needs ``recon_done`` (the imaging pass may be partial); the broadband truth needs only
    ``done``.  Physical reflectivity units; identical selection to the training loader.
    """
    with h5py.File(resolve_data_path(h5_path), "r") as f:
        flag = "recon_done" if (data_key == "lsrtm5" and "recon_done" in f) else "done"
        idx = np.where(f[flag][:] == 1)[0] if flag in f else np.arange(f[data_key].shape[0])
        sel = np.sort(idx[a:b])
        X = f[data_key][sel].astype(np.float32)
    return torch.from_numpy(np.stack([water_mute(x, MUTE_END, TAPER) for x in X]))
