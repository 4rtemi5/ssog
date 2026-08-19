# Copyright (c) 2026 Raphael Pisoni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SSOG: attention as a steerable Gaussian field over relative position.

JAX and PyTorch are optional backends — install only the one you need::

    uv add ssog --extra jax
    uv add ssog --extra torch

Then import from the matching subpackage::

    from ssog.jax import SSOGAttention, ViT
    from ssog.torch import SSOGAttention, ViT
"""

from __future__ import annotations

from ssog._backend import _BACKENDS, missing_root_export_message

__version__ = "0.1.0"

__all__ = ["__version__"]


def __dir__():
    return list(__all__)


def __getattr__(name: str):
    # Point people at the backend packages instead of silently pulling JAX.
    if name in _BACKENDS:
        raise ImportError(missing_root_export_message(name))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
