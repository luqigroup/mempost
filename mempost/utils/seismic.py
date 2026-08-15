"""Seismic reflectivity utilities for the Parihaka DDPM prior.

The prior is trained on 256 x 256 broadband reflectivity images derived from the
Parihaka-3D survey.  The only
preprocessing the trainer needs from here is the water-column mute, which zeroes
the shallow water layer and tapers back to full amplitude exactly as the imaging
operator mutes it -- so the training images carry the same muted water column as
everything the forward operator produces.
"""
from __future__ import annotations

import numpy as np

N = 256                                  # image side, cells
WATER_ZERO_DEPTH = 10                    # cells of fully-muted water column
TAPER = 6                                # cells of half-sine ramp below the water column
MUTE_END = WATER_ZERO_DEPTH + TAPER      # first fully-unmuted depth cell (= 16)


def water_mute(img: np.ndarray, end: int = MUTE_END, length: int = TAPER) -> np.ndarray:
    """Tapered water mute (raised half-sine) along the depth axis (-1).

    Zero above ``end - length``, a half-sine ramp on ``[end - length, end)``, and
    one from ``end`` on.  Matches the imager's ``mute_op``.

    Args:
        img: Image with the depth axis last, e.g. ``(N, N)``.
        end: First fully-unmuted depth cell.
        length: Number of ramp cells below the water column.

    Returns:
        The muted image, ``float32``, same shape as ``img``.
    """
    damp = np.zeros(img.shape[-1], np.float32)
    damp[end:] = 1.0
    damp[end - length:end] = (1.0 + np.sin((np.pi / 2.0 * np.arange(length)) / length)) / 2.0
    return (img * damp).astype(np.float32)
