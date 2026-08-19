# Copyright (c) 2026 Raphael Pisoni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PyTorch attention: SSOG and the matched dot-product baseline.

The math is the same as ``ssog.jax.attention``. This module is written so
``torch.compile`` can see a static graph: no Python loops over batch / heads /
atoms, no data-dependent shapes, and no tensor attributes written during
``forward``.

SSOG replaces content-scored attention with a learned geometric field. Each
head owns a handful of Gaussian atoms over *relative position* — five numbers
per atom: center offset (μy, μx), width (σy, σx) and a mixture weight λ.
Because a 2D Gaussian factorizes, applying the field is two 1D filter passes
per atom. The N×N attention matrix never exists.

With ``lookat=True`` (the default), each token predicts small bounded residuals
on μ, σ and λ through zero-initialized linear probes behind cold-started
gates. Content never scores content; it only deforms the field.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_EPS = 1e-4


def _softplus_std(raw: Tensor, floor: float = 0.0) -> Tensor:
    """Map an unconstrained parameter to a positive width / scale."""
    return F.softplus(raw) + _EPS + floor


def _log_kernel(d: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """log N(d; μ, σ²) with broadcasting, for every pair along one axis."""
    return (
        -0.5 * math.log(2.0 * math.pi)
        - torch.log(sigma)
        - (d - mu).square() / (2.0 * sigma.square())
    )


class DotAttention(nn.Module):
    """Scaled dot-product attention — the matched baseline.

    Uses ``F.scaled_dot_product_attention`` so the baseline gets the fused
    CUDA kernels / FlashAttention path under ``torch.compile``.
    """

    def __init__(self, dim: int, num_heads: int = 6):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, H, N, D) — the layout SDPA expects.
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, n, self.dim)
        return self.out(y)


class SSOGAttention(nn.Module):
    """Gaussian-mixture attention field over relative position.

    Tokens must be the row-major raster of a ``grid_h × grid_w`` image.

    Args:
        dim: token embedding width.
        num_heads: attention heads; each head owns ``num_atoms`` Gaussians.
        num_atoms: Gaussian atoms per head (4 is the sweet spot at small
            scale; even 1 works surprisingly well).
        grid_h, grid_w: token grid the field is defined over.
        lookat: enable content-conditioned steering residuals on μ, σ and λ.
            ``False`` is the purely fixed field.
        max_offset: bound on per-token μ travel, in grid cells (±4 default).
        cold_init: start the steering gates at ≈ 0 (frozen geometry) instead
            of open. Strongly recommended — the cold start is worth ~+1 pt.
        sigma_floor: minimum atom width in grid cells.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        num_atoms: int = 4,
        grid_h: int = 8,
        grid_w: int = 8,
        lookat: bool = True,
        max_offset: float = 4.0,
        cold_init: bool = True,
        sigma_floor: float = 0.25,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.lookat = lookat
        self.max_offset = max_offset
        self.sigma_floor = sigma_floor
        self.head_dim = dim // num_heads

        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        # Shared field: per-head Gaussian atoms.
        self.mu = nn.Parameter(torch.empty(num_heads, num_atoms, 2))
        nn.init.normal_(self.mu, mean=0.0, std=0.5)
        self.raw_sigma = nn.Parameter(torch.full((num_heads, num_atoms, 2), -0.5))
        self.log_lambda = nn.Parameter(torch.zeros(num_heads, num_atoms))
        # Softmax temperature, initialized slightly sharp: softplus(-1)+0.5 ≈ 0.81.
        self.raw_temperature = nn.Parameter(torch.tensor(-1.0))

        # Pairwise displacements along each grid axis — constant, not learned.
        ys = torch.arange(grid_h, dtype=torch.float32)
        xs = torch.arange(grid_w, dtype=torch.float32)
        self.register_buffer("dy", ys[:, None] - ys[None, :], persistent=False)
        self.register_buffer("dx", xs[:, None] - xs[None, :], persistent=False)

        if lookat:
            h, r = num_heads, num_atoms
            gate0 = -8.0 if cold_init else -2.0
            self.mu_delta = nn.Linear(dim, h * r * 2)
            self.sigma_delta = nn.Linear(dim, h * r * 2)
            self.lambda_gate = nn.Linear(dim, h * r)
            nn.init.zeros_(self.mu_delta.weight)
            nn.init.zeros_(self.mu_delta.bias)
            nn.init.zeros_(self.sigma_delta.weight)
            nn.init.zeros_(self.sigma_delta.bias)
            nn.init.zeros_(self.lambda_gate.weight)
            nn.init.zeros_(self.lambda_gate.bias)
            # softplus(-8)+ε ≈ 3e-4, so steering is off at init.
            self.raw_mu_delta_scale = nn.Parameter(torch.tensor(gate0))
            self.raw_sigma_delta_scale = nn.Parameter(torch.tensor(gate0))
            self.raw_lambda_gate_scale = nn.Parameter(torch.tensor(gate0))

    def _sigma(self) -> Tensor:
        return _softplus_std(self.raw_sigma, floor=self.sigma_floor)

    def _temperature(self) -> Tensor:
        return _softplus_std(self.raw_temperature) + 0.5 - _EPS

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        gh, gw, hd = self.grid_h, self.grid_w, self.head_dim
        v = self.v(x).view(b, gh, gw, self.num_heads, hd)
        sigma = self._sigma()
        temperature = self._temperature()
        if self.lookat:
            y = self._steered_apply(x, v, sigma, temperature)
        else:
            y = self._fixed_apply(v, sigma, temperature)
        return self.out(y.reshape(b, n, self.dim))

    def _fixed_apply(self, v: Tensor, sigma: Tensor, temperature: Tensor) -> Tensor:
        """Purely fixed field: the same attention for every image."""
        sy, sx = sigma[:, :, 0], sigma[:, :, 1]
        mu_y, mu_x = self.mu[:, :, 0], self.mu[:, :, 1]
        # (H, R, L, L) kernels over row / column displacements.
        log_ay = _log_kernel(self.dy[None, None], mu_y[:, :, None, None], sy[:, :, None, None])
        log_ax = _log_kernel(self.dx[None, None], mu_x[:, :, None, None], sx[:, :, None, None])
        ay = torch.softmax(log_ay / temperature, dim=-1)
        ax = torch.softmax(log_ax / temperature, dim=-1)
        lam = torch.softmax(self.log_lambda, dim=-1)
        y = torch.einsum("prij,bjwpd->biwpdr", ay, v)
        y = torch.einsum("prjk,bikpdr->bijpdr", ax, y)
        return torch.einsum("pr,bijpdr->bijpd", lam, y)

    def _steered_apply(
        self, x: Tensor, v: Tensor, sigma: Tensor, temperature: Tensor
    ) -> Tensor:
        """Fixed field + bounded per-query residuals on μ, σ and λ."""
        b = x.shape[0]
        gh, gw = self.grid_h, self.grid_w
        h, r = self.num_heads, self.num_atoms

        # μ: shift where each atom looks, bounded to ±max_offset grid cells.
        mu_scale = _softplus_std(self.raw_mu_delta_scale)
        dmu = self.mu_delta(x).view(b, gh, gw, h, r, 2)
        mu_y = self.mu[None, None, None, :, :, 0] + mu_scale * self.max_offset * torch.tanh(dmu[..., 0])
        mu_x = self.mu[None, None, None, :, :, 1] + mu_scale * self.max_offset * torch.tanh(dmu[..., 1])

        # σ: widen / tighten each atom, bounded log-space multiplier.
        sig_scale = _softplus_std(self.raw_sigma_delta_scale)
        dsig = self.sigma_delta(x).view(b, gh, gw, h, r, 2)
        sy = sigma[None, None, None, :, :, 0] * torch.exp(sig_scale * torch.tanh(dsig[..., 0]))
        sx = sigma[None, None, None, :, :, 1] * torch.exp(sig_scale * torch.tanh(dsig[..., 1]))

        # Per-query per-axis kernels: (B, gh, gw, H, R, L).
        log_ay = _log_kernel(self.dy[None, :, None, None, None, :], mu_y[..., None], sy[..., None])
        log_ax = _log_kernel(self.dx[None, None, :, None, None, :], mu_x[..., None], sx[..., None])
        ay = torch.softmax(log_ay / temperature, dim=-1)
        ax = torch.softmax(log_ax / temperature, dim=-1)

        # Two 1D filter passes per atom — the N×N matrix never exists.
        y = torch.einsum("biwprj,bjwpd->biwpdr", ay, v)
        y = torch.einsum("biwprk,bikpdr->biwpdr", ax, y)

        # λ: re-weight which atoms matter, per query.
        lam_scale = _softplus_std(self.raw_lambda_gate_scale)
        gate = self.lambda_gate(x).view(b, gh, gw, h, r)
        lam_q = torch.softmax(
            self.log_lambda[None, None, None] + lam_scale * torch.tanh(gate), dim=-1
        )
        return torch.einsum("biwpr,biwpdr->biwpd", lam_q, y)

    def axis_kernels(self) -> tuple[Tensor, Tensor]:
        """Fixed-field (ay, ax) kernels for plotting — not used in ``forward``.

        Shapes are ``(H, R, grid_h, grid_h)`` and ``(H, R, grid_w, grid_w)``.
        Call this in eager mode; it is not part of the compiled training graph.
        """
        sigma = self._sigma()
        temperature = self._temperature()
        sy, sx = sigma[:, :, 0], sigma[:, :, 1]
        mu_y, mu_x = self.mu[:, :, 0], self.mu[:, :, 1]
        ay = torch.softmax(
            _log_kernel(self.dy[None, None], mu_y[:, :, None, None], sy[:, :, None, None])
            / temperature,
            dim=-1,
        )
        ax = torch.softmax(
            _log_kernel(self.dx[None, None], mu_x[:, :, None, None], sx[:, :, None, None])
            / temperature,
            dim=-1,
        )
        return ay, ax
