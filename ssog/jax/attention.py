# Copyright (c) 2026 Raphael Pisoni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attention layers: SSOG and the dot-product baseline it replaces.

SSOG (Separable Sum of Gaussians) replaces content-scored attention with a
learned geometric field. Each head owns a handful of Gaussian atoms defined
over *relative position* — five numbers per atom: center offset (μy, μx),
width (σy, σx) and a mixture weight λ. The attention weight from token p to
token q is the value of that Gaussian mixture at the displacement p − q:

    A(p, q) = softmax_q( logsumexp_r( log λ_r + log N(p − q; μ_r, σ_r) ) )

Because a 2D Gaussian factorizes into two 1D Gaussians, applying the field to
the values is just two 1D filter passes per atom — the N×N attention matrix
never exists. Cost: O(N·√N·d) instead of O(N²·d).

With ``lookat=True`` (the default), each token additionally predicts small,
bounded residuals on the field parameters through zero-initialized linear
layers with cold-started gates — so the model starts life as a frozen
geometric animal and learns during training whether to let content steer its
geometry. Content never scores content; it only deforms the field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

_EPS = 1e-4


def _softplus_std(raw, floor=0.0):
    """Map an unconstrained raw parameter to a positive width/scale."""
    return jax.nn.softplus(raw) + _EPS + floor


class DotAttention(nn.Module):
    """Standard scaled dot-product attention — the matched baseline."""

    dim: int
    num_heads: int = 6

    @nn.compact
    def __call__(self, x):
        b, n, d = x.shape
        head_dim = self.dim // self.num_heads
        qkv = nn.Dense(3 * self.dim, use_bias=False, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(b, n, self.num_heads, head_dim)
        k = k.reshape(b, n, self.num_heads, head_dim)
        v = v.reshape(b, n, self.num_heads, head_dim)
        logits = jnp.einsum("bnhd,bmhd->bhnm", q, k) * head_dim**-0.5
        attn = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bhnm,bmhd->bnhd", attn, v).reshape(b, n, self.dim)
        return nn.Dense(self.dim, use_bias=False, name="out")(out)


class SSOGAttention(nn.Module):
    """Gaussian-mixture attention field over relative position.

    Args:
        dim: token embedding width.
        num_heads: attention heads; each head owns ``num_atoms`` Gaussians.
        num_atoms: Gaussian atoms per head (4 is the sweet spot at small
            scale; even 1 works surprisingly well).
        grid_h, grid_w: token grid the field is defined over (tokens are
            assumed to be the row-major raster of a grid_h × grid_w image).
        lookat: enable the content-conditioned steering residuals on μ, σ
            and λ. With ``lookat=False`` you get the purely fixed field.
        max_offset: bound on per-token μ travel, in grid cells (±4 default).
        cold_init: start the steering gates at ≈ 0 (frozen geometry) instead
            of open. Strongly recommended — the cold start is worth ~+1 pt.
        sigma_floor: minimum atom width in grid cells.
        capture_kernels: if True, sow the per-axis attention kernels into
            ``intermediates`` so you can plot the field afterwards.
    """

    dim: int
    num_heads: int = 6
    num_atoms: int = 4
    grid_h: int = 8
    grid_w: int = 8
    lookat: bool = True
    max_offset: float = 4.0
    cold_init: bool = True
    sigma_floor: float = 0.25
    capture_kernels: bool = False

    @nn.compact
    def __call__(self, x):
        b, n, d = x.shape
        gh, gw = self.grid_h, self.grid_w
        head_dim = self.dim // self.num_heads
        assert n == gh * gw, "tokens must tile the grid row-major"

        v = nn.Dense(self.dim, use_bias=False, name="v")(x)
        v = v.reshape(b, gh, gw, self.num_heads, head_dim)

        # Displacements between all pairs of grid rows / columns.
        ys = jnp.arange(gh, dtype=jnp.float32)
        xs = jnp.arange(gw, dtype=jnp.float32)
        dy = ys[:, None] - ys[None, :]  # (gh, gh)
        dx = xs[:, None] - xs[None, :]  # (gw, gw)

        # --- The field: learned per-head Gaussian atoms ---------------------
        mu = self.param(
            "mu", nn.initializers.normal(stddev=0.5),
            (self.num_heads, self.num_atoms, 2),
        )
        raw_sigma = self.param(
            "raw_sigma", nn.initializers.constant(-0.5),
            (self.num_heads, self.num_atoms, 2),
        )
        log_lambda = self.param(
            "log_lambda", nn.initializers.zeros, (self.num_heads, self.num_atoms)
        )
        sigma = _softplus_std(raw_sigma, floor=self.sigma_floor)
        lam = jax.nn.softmax(log_lambda, axis=-1)  # (H, R)

        # Softmax temperature, initialized slightly sharp.
        temperature = _softplus_std(
            self.param("raw_temperature", nn.initializers.constant(-1.0), ())
        ) + 0.5 - _EPS

        if self.lookat:
            y = self._steered_apply(x, v, dy, dx, mu, sigma, log_lambda, temperature)
        else:
            y = self._fixed_apply(v, dy, dx, mu, sigma, lam, temperature)

        y = y.reshape(b, n, self.dim)
        return nn.Dense(self.dim, use_bias=False, name="out")(y)

    # ------------------------------------------------------------------
    def _log_kernel(self, d, mu, sigma):
        """log N(d; μ, σ²) for every pair of positions along one axis."""
        return (
            -0.5 * jnp.log(2.0 * jnp.pi)
            - jnp.log(sigma)
            - jnp.square(d - mu) / (2.0 * jnp.square(sigma))
        )

    def _fixed_apply(self, v, dy, dx, mu, sigma, lam, temperature):
        """Purely fixed field: the same attention for every image."""
        sy, sx = sigma[:, :, 0], sigma[:, :, 1]
        mu_y, mu_x = mu[:, :, 0], mu[:, :, 1]
        log_ay = self._log_kernel(
            dy[None, None], mu_y[:, :, None, None], sy[:, :, None, None]
        )
        log_ax = self._log_kernel(
            dx[None, None], mu_x[:, :, None, None], sx[:, :, None, None]
        )
        ay = jax.nn.softmax(log_ay / temperature, axis=-1)  # (H, R, gh, gh)
        ax = jax.nn.softmax(log_ax / temperature, axis=-1)  # (H, R, gw, gw)
        if self.capture_kernels:
            self.sow("intermediates", "ay", ay)
            self.sow("intermediates", "ax", ax)
        y = jnp.einsum("prij,bjwpd->biwpdr", ay, v)
        y = jnp.einsum("prjk,bikpdr->bijpdr", ax, y)
        return jnp.einsum("pr,bijpdr->bijpd", lam, y)

    def _gate(self, x, out_features, name):
        """Zero-initialized per-token linear probe + cold-started gate scale.

        The Dense layer starts at exactly zero and the softplus gate scale
        starts at ≈ 0.0003, so at init the steering residuals vanish and the
        field is frozen. The model opens the taps itself during training.
        """
        delta = nn.Dense(
            out_features,
            use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name=name,
        )(x)
        raw_scale = self.param(
            f"raw_{name}_scale",
            nn.initializers.constant(-8.0 if self.cold_init else -2.0),
            (),
        )
        return delta, _softplus_std(raw_scale)

    def _steered_apply(self, x, v, dy, dx, mu, sigma, log_lambda, temperature):
        """Fixed field + bounded per-query residuals on μ, σ and λ."""
        b = x.shape[0]
        gh, gw = self.grid_h, self.grid_w
        H, R = self.num_heads, self.num_atoms

        # μ: shift where each atom looks, bounded to ±max_offset grid cells.
        dmu, mu_scale = self._gate(x, H * R * 2, "mu_delta")
        dmu = dmu.reshape(b, gh, gw, H, R, 2)
        mu_y = mu[None, None, None, :, :, 0] + mu_scale * self.max_offset * jnp.tanh(dmu[..., 0])
        mu_x = mu[None, None, None, :, :, 1] + mu_scale * self.max_offset * jnp.tanh(dmu[..., 1])

        # σ: widen/tighten each atom, bounded log-space multiplier.
        dsig, sig_scale = self._gate(x, H * R * 2, "sigma_delta")
        dsig = dsig.reshape(b, gh, gw, H, R, 2)
        sy = sigma[None, None, None, :, :, 0] * jnp.exp(sig_scale * jnp.tanh(dsig[..., 0]))
        sx = sigma[None, None, None, :, :, 1] * jnp.exp(sig_scale * jnp.tanh(dsig[..., 1]))

        # Per-query per-axis kernels: (B, gh, gw, H, R, L).
        log_ay = self._log_kernel(
            dy[None, :, None, None, None, :], mu_y[..., None], sy[..., None]
        )
        log_ax = self._log_kernel(
            dx[None, None, :, None, None, :], mu_x[..., None], sx[..., None]
        )
        ay = jax.nn.softmax(log_ay / temperature, axis=-1)
        ax = jax.nn.softmax(log_ax / temperature, axis=-1)
        if self.capture_kernels:
            self.sow("intermediates", "ay", ay)
            self.sow("intermediates", "ax", ax)

        # Two 1D filter passes per atom — the N×N matrix never exists.
        y = jnp.einsum("biwprj,bjwpd->biwpdr", ay, v)  # down the rows, per atom
        y = jnp.einsum("biwprk,bikpdr->biwpdr", ax, y)  # across the columns

        # λ: re-weight which atoms matter, per query.
        gate, lam_scale = self._gate(x, H * R, "lambda_gate")
        gate = gate.reshape(b, gh, gw, H, R)
        lam_q = jax.nn.softmax(
            log_lambda[None, None, None] + lam_scale * jnp.tanh(gate), axis=-1
        )
        return jnp.einsum("biwpr,biwpdr->biwpd", lam_q, y)  # mix atoms
