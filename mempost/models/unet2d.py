"""UNet2D score network for the seismic reflectivity DDPM prior.

A four-scale ``diffusers.UNet2DModel`` with attention at the coarsest two scales,
operating on single-channel 256 x 256 per-pixel-standardized reflectivity images.
Epsilon-prediction, cosine-beta DDPM, no EMA.  The training recipe
lives in ``scripts/seismic_prior_ddpm.py``.

``diffusers`` is an optional dependency: this module is imported directly by the
seismic scripts (never re-exported through ``mempost.models``), so the KL /
Helmholtz pipeline does not need ``diffusers`` installed.
"""
from __future__ import annotations

from diffusers import UNet2DModel

from mempost.utils.seismic import N

BASE = 64        # UNet base width of the deployed prior (~25 M params)


def make_unet(base: int = BASE) -> UNet2DModel:
    """The prior's score network: four scales, attention at the coarsest two.

    Args:
        base: Width of the shallowest scale; deeper scales are ``2x``, ``4x``, ``4x``.

    Returns:
        An epsilon-predicting ``UNet2DModel`` over ``1 x N x N`` images.
    """
    return UNet2DModel(
        sample_size=N, in_channels=1, out_channels=1, layers_per_block=2,
        block_out_channels=(base, base * 2, base * 4, base * 4),
        norm_num_groups=min(32, base),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"))
