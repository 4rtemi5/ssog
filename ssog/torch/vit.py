# Copyright (c) 2026 Raphael Pisoni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A minimal vision transformer with either SSOG or dot-product attention.

Same architecture as ``ssog.jax.vit``: patch embedding, learned position
embedding, pre-norm blocks (attention + MLP), mean pooling, linear head.
Images are NCHW — the PyTorch convention — and dropout follows ``self.training``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from ssog.torch.attention import DotAttention, SSOGAttention


class Block(nn.Module):
    """Pre-norm transformer block: LN → attention → residual, LN → MLP → residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout_rate: float = 0.1,
        attn: Literal["ssog", "dot"] = "ssog",
        grid_h: int = 8,
        grid_w: int = 8,
        num_atoms: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attn == "ssog":
            self.attn: nn.Module = SSOGAttention(
                dim=dim,
                num_heads=num_heads,
                num_atoms=num_atoms,
                grid_h=grid_h,
                grid_w=grid_w,
            )
        else:
            self.attn = DotAttention(dim=dim, num_heads=num_heads)
        self.drop1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        return x + self.drop2(self.mlp(self.norm2(x)))


class ViT(nn.Module):
    """Patch-based image classifier.

    Args:
        num_classes: output classes.
        img_size: input images are img_size × img_size.
        patch_size: patch edge; the token grid is (img_size/patch_size)².
        dim, depth, num_heads: standard transformer widths.
        attn: ``"ssog"`` for the Gaussian field, ``"dot"`` for the baseline.
        num_atoms: Gaussian atoms per head (SSOG only).
    """

    def __init__(
        self,
        num_classes: int = 100,
        img_size: int = 32,
        patch_size: int = 2,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout_rate: float = 0.1,
        attn: Literal["ssog", "dot"] = "ssog",
        num_atoms: int = 4,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")
        gh = gw = img_size // patch_size
        self.patch_size = patch_size
        self.grid_h = gh
        self.grid_w = gw
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, dim)
        self.pos_embed = nn.Parameter(torch.empty(1, gh * gw, dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.drop = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                    attn=attn,
                    grid_h=gh,
                    grid_w=gw,
                    num_atoms=num_atoms,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        p, gh, gw = self.patch_size, self.grid_h, self.grid_w
        # Non-overlapping patches, row-major raster → (B, gh·gw, p·p·C).
        x = x.reshape(b, c, gh, p, gw, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, p * p * c)
        x = self.patch_embed(x) + self.pos_embed
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)
