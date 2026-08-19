# Copyright (c) 2026 Raphael Pisoni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PyTorch implementation of SSOG."""

from ssog._backend import require_torch

require_torch()

from ssog.torch.attention import DotAttention, SSOGAttention
from ssog.torch.vit import Block, ViT

__all__ = ["SSOGAttention", "DotAttention", "Block", "ViT"]
