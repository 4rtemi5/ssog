# Copyright (c) 2026 Raphael Pisoni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""JAX / Flax implementation of SSOG."""

from ssog._backend import require_jax

require_jax()

from ssog.jax.attention import DotAttention, SSOGAttention
from ssog.jax.vit import Block, ViT

__all__ = ["SSOGAttention", "DotAttention", "Block", "ViT"]
