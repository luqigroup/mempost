"""1D convolutional UNet for diffusion denoising.

Mirrors the diffusers ``UNet2DModel`` architecture but operates on 1D
sequences with ``Conv1d`` layers.  Key design choices match diffusers:

- SiLU activations throughout (not GELU).
- Time embedding: sinusoidal → Linear → SiLU → Linear, projected into
  each ResBlock after a SiLU gate.
- ResBlock: norm → act → conv → +temb → norm → act → dropout → conv,
  with a ``conv_shortcut`` when in/out channels differ (skip concat).
- Self-attention: GroupNorm → Linear Q/K/V → multi-head scaled dot
  product → Linear out (no positional encoding, matching diffusers).
- Skip connections: concatenate encoder features with decoder features,
  first ResBlock in each decoder level takes doubled input channels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock1d(nn.Module):
    """Residual block matching diffusers ``ResnetBlock2D``.

    When ``in_channels != out_channels``, a 1x1 ``conv_shortcut`` is
    used on the skip path (matching diffusers behaviour for decoder
    blocks that receive concatenated skip connections).
    """

    def __init__(self, in_channels: int, out_channels: int, t_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               padding=1)
        self.t_proj = nn.Linear(t_dim, out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv1d(in_channels, out_channels,
                                           kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # diffusers applies SiLU to temb before projection
        h = h + self.t_proj(F.silu(t_emb))[:, :, None]
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class SelfAttention1d(nn.Module):
    """Multi-head self-attention matching diffusers ``Attention``.

    GroupNorm → reshape to (B, L, C) tokens → Linear Q/K/V →
    scaled dot-product attention → Linear out → residual.
    No positional encoding (same as diffusers).
    """

    def __init__(self, channels: int, head_dim: int = 8):
        super().__init__()
        self.n_heads = channels // head_dim
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = self.norm(x)
        # (B, C, L) → (B, L, C) — treat spatial positions as tokens
        h = h.permute(0, 2, 1)
        q = self.to_q(h).reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3)
        k = self.to_k(h).reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3)
        v = self.to_v(h).reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3)
        h = F.scaled_dot_product_attention(q, k, v)  # (B, heads, L, d)
        h = h.permute(0, 2, 1, 3).reshape(B, L, C)   # (B, L, C)
        h = self.to_out(h)
        h = h.permute(0, 2, 1)  # (B, C, L)
        return x + h


class Downsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2,
                              padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet1d
# ---------------------------------------------------------------------------

class UNet1d(nn.Module):
    """1D convolutional UNet for diffusion models.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for a scalar field like Vp(z)).
    base_channels : int
        Channel count at the first level.
    channel_mults : tuple of int
        Multipliers for each level.  E.g. ``(1, 2, 4, 8)`` gives four
        levels with ``base_channels * m`` channels each.
    n_res_blocks : int
        Number of residual blocks per level.
    t_dim : int
        Dimension of the timestep embedding.
    dropout : float
        Dropout probability in residual blocks.
    attn_levels : tuple of int
        Which encoder/decoder levels get self-attention (e.g. ``(3,)``).
        The bottleneck always has attention.
    attn_head_dim : int
        Dimension per attention head.  Number of heads = channels / head_dim.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        n_res_blocks: int = 2,
        t_dim: int = 256,
        dropout: float = 0.0,
        attn_levels: tuple[int, ...] = (),
        attn_head_dim: int = 8,
    ):
        super().__init__()
        self.n_levels = len(channel_mults)
        self.attn_levels = set(attn_levels)

        # --- timestep embedding (matches diffusers TimestepEmbedding) -----
        # Sinusoidal → Linear(t_dim → 4*t_dim) → SiLU → Linear(4*t_dim → 4*t_dim)
        inner_dim = t_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )
        t_emb_dim = inner_dim  # ResBlocks receive this dimension

        # --- input projection ---------------------------------------------
        ch0 = base_channels * channel_mults[0]
        self.in_conv = nn.Conv1d(in_channels, ch0, kernel_size=3, padding=1)

        # --- encoder (downsampling path) ----------------------------------
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for level, mult in enumerate(channel_mults):
            ch = base_channels * mult
            block = nn.ModuleList()
            for _ in range(n_res_blocks):
                block.append(ResBlock1d(ch, ch, t_emb_dim, dropout))
            self.down_blocks.append(block)
            if level in self.attn_levels:
                self.down_attns.append(SelfAttention1d(ch, attn_head_dim))
            else:
                self.down_attns.append(nn.Identity())
            if level < self.n_levels - 1:
                ch_next = base_channels * channel_mults[level + 1]
                self.downsamplers.append(
                    nn.Sequential(
                        Downsample1d(ch),
                        nn.Conv1d(ch, ch_next, 1),
                    )
                )

        # --- bottleneck ---------------------------------------------------
        ch_bot = base_channels * channel_mults[-1]
        self.mid_block1 = ResBlock1d(ch_bot, ch_bot, t_emb_dim, dropout)
        self.mid_attn = (
            SelfAttention1d(ch_bot, attn_head_dim)
            if len(attn_levels) > 0
            else nn.Identity()
        )
        self.mid_block2 = ResBlock1d(ch_bot, ch_bot, t_emb_dim, dropout)

        # --- decoder (upsampling path) ------------------------------------
        # Matches diffusers: first ResBlock in each level takes
        # concatenated channels (skip + current), subsequent take ch.
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in reversed(range(self.n_levels)):
            ch = base_channels * channel_mults[level]
            if level < self.n_levels - 1:
                ch_prev = base_channels * channel_mults[level + 1]
                self.upsamplers.append(
                    nn.Sequential(
                        Upsample1d(ch_prev),
                        nn.Conv1d(ch_prev, ch, 1),
                    )
                )
            block = nn.ModuleList()
            # First ResBlock takes concatenated skip (2*ch → ch).
            block.append(ResBlock1d(ch * 2, ch, t_emb_dim, dropout))
            for _ in range(n_res_blocks - 1):
                block.append(ResBlock1d(ch, ch, t_emb_dim, dropout))
            self.up_blocks.append(block)
            if level in self.attn_levels:
                self.up_attns.append(SelfAttention1d(ch, attn_head_dim))
            else:
                self.up_attns.append(nn.Identity())

        # --- output projection --------------------------------------------
        self.out_norm = nn.GroupNorm(min(32, ch0), ch0)
        self.out_conv = nn.Conv1d(ch0, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy input and timestep.

        Parameters
        ----------
        x : (B, C, L) or (B, L)
            Noisy 1D signal.  If 2-D, a channel dimension is added.
        t : (B,)
            Integer diffusion timesteps.

        Returns
        -------
        (B, C, L) or (B, L)
            Predicted noise, same shape as input.
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)  # (B, 1, L)

        orig_len = x.shape[-1]

        # pad to multiple of 2^(n_levels-1)
        factor = 2 ** (self.n_levels - 1)
        pad_total = (factor - orig_len % factor) % factor
        if pad_total > 0:
            x = F.pad(x, (0, pad_total), mode="reflect")

        t_emb = self.time_embed(t)  # (B, 4*t_dim)

        # --- encoder ------------------------------------------------------
        h = self.in_conv(x)
        skips = [h]

        for level in range(self.n_levels):
            for res_block in self.down_blocks[level]:
                h = res_block(h, t_emb)
            h = self.down_attns[level](h)
            skips.append(h)
            if level < self.n_levels - 1:
                h = self.downsamplers[level](h)

        # --- bottleneck ---------------------------------------------------
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # --- decoder ------------------------------------------------------
        for level_idx in range(self.n_levels):
            if level_idx > 0:
                h = self.upsamplers[level_idx - 1](h)
            skip = skips.pop()
            # align lengths after upsampling
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            # Concatenate skip and pass through ResBlocks (first one
            # has conv_shortcut to handle 2*ch → ch, like diffusers).
            h = torch.cat([h, skip], dim=1)
            for res_block in self.up_blocks[level_idx]:
                h = res_block(h, t_emb)
            h = self.up_attns[level_idx](h)

        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)

        # crop back to original length
        h = h[..., :orig_len]

        if squeeze:
            h = h.squeeze(1)
        return h
