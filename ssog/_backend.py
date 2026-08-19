# Copyright (c) 2026 Raphael Pisoni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Install-time checks for the optional JAX / PyTorch backends.

Imported by ``ssog.jax`` and ``ssog.torch`` so a missing extra, a
``from ssog import ViT``, or a CPU wheel on a CUDA machine gets a
fix-it message instead of a raw ``No module named 'jax'``.
"""

from __future__ import annotations

import os
import sys
import warnings
from importlib.util import find_spec

_BACKENDS = ("SSOGAttention", "DotAttention", "Block", "ViT")


def _in_ssog_repo() -> bool:
    """True when cwd (or its parents) look like this source tree."""
    here = os.path.abspath(os.getcwd())
    for _ in range(4):
        if os.path.isfile(os.path.join(here, "pyproject.toml")):
            try:
                with open(os.path.join(here, "pyproject.toml"), encoding="utf-8") as f:
                    return 'name = "ssog"' in f.read()
            except OSError:
                return False
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return False


def _uv_install(extra: str) -> str:
    if _in_ssog_repo():
        return (
            f"    uv sync --extra {extra}\n"
            f"    uv add ssog --extra {extra}     # if this is a different project"
        )
    return (
        f"    uv add ssog --extra {extra}\n"
        f"    uv sync --extra {extra}          # if you cloned the SSOG repo"
    )


def _missing_backend_message(backend: str, extra: str, missing: list[str]) -> str:
    pkgs = ", ".join(missing)
    extra_hint = ""
    if extra == "jax":
        extra_hint = (
            "\nFor NVIDIA GPUs use the CUDA extra instead:\n"
            "    uv add ssog --extra jax-cuda\n"
        )
    elif extra == "torch":
        extra_hint = (
            "\nLinux PyPI torch is usually CUDA already. For a specific build:\n"
            "    uv add torch --index https://download.pytorch.org/whl/cu128\n"
            "    uv add ssog --extra torch\n"
        )
    return (
        f"ssog.{backend} requires {pkgs}, which is not installed.\n"
        f"\n"
        f"Install the extra:\n"
        f"{_uv_install(extra)}\n"
        f"    pip install 'ssog[{extra}]'\n"
        f"{extra_hint}"
        f"Then:  from ssog.{backend} import SSOGAttention, ViT"
    )


def missing_root_export_message(name: str) -> str:
    return (
        f"{name!r} is not exported from the top-level 'ssog' package — "
        f"JAX and PyTorch live in separate extras so one install never pulls "
        f"the other.\n"
        f"\n"
        f"JAX:\n"
        f"{_uv_install('jax')}\n"
        f"    from ssog.jax import {name}\n"
        f"\n"
        f"PyTorch:\n"
        f"{_uv_install('torch')}\n"
        f"    from ssog.torch import {name}"
    )


def _nvidia_gpu_present() -> bool:
    """Best-effort: user hid GPUs, or this machine has an NVIDIA device node."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    if sys.platform.startswith("linux"):
        return os.path.exists("/dev/nvidiactl") or os.path.exists("/dev/nvidia0")
    return False


def require_jax() -> None:
    missing: list[str] = []
    try:
        import jax  # noqa: F401
    except ImportError:
        missing.append("jax")
    try:
        import flax  # noqa: F401
    except ImportError:
        missing.append("flax")
    if missing:
        raise ImportError(
            _missing_backend_message("jax", "jax", missing)
        ) from None

    if _nvidia_gpu_present() and find_spec("jax_cuda12_plugin") is None:
        warnings.warn(
            "An NVIDIA GPU is visible but the JAX CUDA 12 plugin is not installed, "
            "so JAX will run on CPU. Fix with:\n"
            "    uv add ssog --extra jax-cuda\n"
            "    uv sync --extra jax-cuda          # this repo\n"
            "    uv run --extra examples-jax-cuda examples/train_cifar100.py",
            stacklevel=3,
        )


def require_torch() -> None:
    try:
        import torch
    except ImportError:
        raise ImportError(
            _missing_backend_message("torch", "torch", ["torch"])
        ) from None

    if _nvidia_gpu_present() and getattr(torch.version, "cuda", None) is None:
        warnings.warn(
            "An NVIDIA GPU is visible but this PyTorch build has no CUDA "
            "(CPU-only wheel). Training will be on CPU. Fix with a CUDA wheel "
            "from https://pytorch.org, e.g.:\n"
            "    uv add torch torchvision --index https://download.pytorch.org/whl/cu128\n"
            "    uv add ssog --extra torch",
            stacklevel=3,
        )
