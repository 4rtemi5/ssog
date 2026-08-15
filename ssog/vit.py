"""A minimal vision transformer that runs with either attention flavor.

Deliberately small: patch embedding, a learned position embedding, pre-norm
blocks (attention + MLP), mean pooling, linear head. The only interesting
knob is ``attn="ssog"`` vs ``attn="dot"`` — everything else is standard.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from flax import linen as nn

from ssog.attention import DotAttention, SSOGAttention


class Block(nn.Module):
    """Pre-norm transformer block: LN → attention → residual, LN → MLP → residual."""

    dim: int
    num_heads: int
    mlp_ratio: float = 2.0
    dropout_rate: float = 0.1
    attn: Literal["ssog", "dot"] = "ssog"
    grid_h: int = 8
    grid_w: int = 8
    num_atoms: int = 4

    @nn.compact
    def __call__(self, x, train: bool = False):
        y = nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(x.dtype)
        if self.attn == "ssog":
            y = SSOGAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                num_atoms=self.num_atoms,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
            )(y)
        else:
            y = DotAttention(dim=self.dim, num_heads=self.num_heads)(y)
        y = nn.Dropout(self.dropout_rate, deterministic=not train)(y)
        x = x + y

        y = nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(x.dtype)
        hidden = int(self.dim * self.mlp_ratio)
        y = nn.Dense(hidden)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.dim)(y)
        y = nn.Dropout(self.dropout_rate, deterministic=not train)(y)
        return x + y


class ViT(nn.Module):
    """Patch-based image classifier.

    Args:
        num_classes: output classes.
        img_size: input images are img_size × img_size.
        patch_size: patch edge; the token grid is (img_size/patch_size)².
        dim, depth, num_heads: standard transformer widths.
        attn: "ssog" for the Gaussian field, "dot" for the baseline.
        num_atoms: Gaussian atoms per head (SSOG only).
    """

    num_classes: int = 100
    img_size: int = 32
    patch_size: int = 2
    dim: int = 256
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout_rate: float = 0.1
    attn: Literal["ssog", "dot"] = "ssog"
    num_atoms: int = 4

    @nn.compact
    def __call__(self, x, train: bool = False):
        # x: (B, H, W, C) images
        b, h, w, c = x.shape
        p = self.patch_size
        gh, gw = h // p, w // p

        # Non-overlapping patches, row-major raster order.
        x = x.reshape(b, gh, p, gw, p, c)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(b, gh * gw, p * p * c)
        x = nn.Dense(self.dim, name="patch_embed")(x)

        n = x.shape[1]
        pos = self.param(
            "pos_embed", nn.initializers.normal(stddev=0.02), (1, n, self.dim)
        )
        x = x + pos.astype(x.dtype)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        for i in range(self.depth):
            x = Block(
                dim=self.dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                attn=self.attn,
                grid_h=gh,
                grid_w=gw,
                num_atoms=self.num_atoms,
                name=f"block_{i}",
            )(x, train=train)

        x = nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32))
        x = jnp.mean(x, axis=1)  # global average pooling over tokens
        return nn.Dense(self.num_classes, name="head")(x)
