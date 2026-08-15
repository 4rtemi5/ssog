"""Train a small ViT on CIFAR-100 with SSOG or dot-product attention.

Quick compare (enough to see the small-data gap)::

    python examples/train_cifar100.py --attn ssog --epochs 10
    python examples/train_cifar100.py --attn dot  --epochs 10

Full recipe defaults to 100 epochs (closer to the blog numbers). Downloads
CIFAR-100 via HuggingFace ``datasets`` on first run.

Recipe (intentionally minimal, matched across ``--attn``)::

    * pixels in ``[0, 1]`` (÷255 only — no channel mean/std)
    * pad-4 random crop + horizontal flip with p=0.5
    * AdamW (wd=0.05) + linear warmup (5 epochs) + cosine decay
    * train metrics = mean over training batches (augment + dropout on)
    * last incomplete batch dropped (50_000 % batch_size leftovers unused)
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset

from ssog import ViT

# Classic CIFAR "RandomCrop(32, padding=4)": reflect-free zero pad, then crop.
CROP_PAD = 4


def load_cifar100():
    """Return ``((x_train, y_train), (x_test, y_test))`` as NumPy arrays.

    Images are float32 in ``[0, 1]``, NHWC. Labels are CIFAR-100 *fine*
    labels (100-way), matching ``ViT(num_classes=100)``.
    """

    def to_np(split):
        imgs = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in split["img"]])
        labels = np.asarray(split["fine_label"], dtype=np.int32)
        return imgs, labels

    ds = load_dataset("uoft-cs/cifar100")
    return to_np(ds["train"]), to_np(ds["test"])


def _augment_one(rng, img, pad: int = CROP_PAD):
    """Pad-then-crop + horizontal flip for a single HWC image."""
    h, w, c = img.shape
    padded = jnp.pad(img, ((pad, pad), (pad, pad), (0, 0)))
    rng_y, rng_x, rng_flip = jax.random.split(rng, 3)
    # Inclusive offsets into the padded image: {0, …, 2·pad}.
    oy = jax.random.randint(rng_y, (), 0, 2 * pad + 1)
    ox = jax.random.randint(rng_x, (), 0, 2 * pad + 1)
    out = jax.lax.dynamic_slice(padded, (oy, ox, 0), (h, w, c))
    flip = jax.random.bernoulli(rng_flip, p=0.5)
    # HWC: reverse the width axis for a horizontal flip.
    return jnp.where(flip, out[:, ::-1], out)


def augment(rng, imgs, pad: int = CROP_PAD):
    """Batched on-device augment: vmap of :func:`_augment_one`."""
    keys = jax.random.split(rng, imgs.shape[0])
    return jax.vmap(lambda k, im: _augment_one(k, im, pad))(keys, imgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attn", choices=["ssog", "dot"], default="ssog")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    (xtr, ytr), (xte, yte) = load_cifar100()
    # Whole CIFAR-100 fits on GPU (~600MB float32); gather + augment inside the step.
    xtr, ytr = jnp.asarray(xtr), jnp.asarray(ytr)
    xte, yte = jnp.asarray(xte), jnp.asarray(yte)

    model = ViT(
        num_classes=100, img_size=32, patch_size=2,
        dim=args.dim, depth=args.depth, num_heads=args.heads, attn=args.attn,
    )

    rng = jax.random.PRNGKey(args.seed)
    # Flax does not advance the Python key; split so train RNG ≠ init RNG.
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, jnp.zeros((1, 32, 32, 3)), train=False)
    params = variables["params"]
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"attn={args.attn}  params={n_params/1e6:.2f}M")

    n_train = int(xtr.shape[0])
    # Drop the last incomplete batch so every step has a fixed shape for JIT.
    steps_per_epoch = n_train // args.batch_size
    total_steps = max(1, args.epochs * steps_per_epoch)
    # 5-epoch warmup in the default recipe; clamp so short --epochs runs still peak.
    warmup_steps = min(5 * steps_per_epoch, total_steps - 1)
    # optax: decay_steps is the *total* schedule length (warmup + cosine).
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, warmup_steps=warmup_steps, decay_steps=total_steps,
    )
    # Minimal AdamW: weight decay applies to all params (no bias/LN exclusion).
    tx = optax.adamw(schedule, weight_decay=0.05)
    opt_state = tx.init(params)

    def loss_fn(params, x, y, rng):
        logits = model.apply({"params": params}, x, train=True, rngs={"dropout": rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss, logits

    @jax.jit
    def step(params, opt_state, idx, rng):
        rng, rng_aug, rng_drop = jax.random.split(rng, 3)
        x = augment(rng_aug, xtr[idx])
        y = ytr[idx]
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, y, rng_drop
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        acc = (logits.argmax(-1) == y).mean()
        return params, opt_state, loss, acc, rng

    @jax.jit
    def evaluate(params, x, y):
        logits = model.apply({"params": params}, x, train=False)
        return (logits.argmax(-1) == y).mean()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Deterministic per-epoch shuffle, independent of the step RNG stream.
        perm = jax.random.permutation(jax.random.PRNGKey(args.seed + epoch), n_train)
        ep_loss, ep_acc = jnp.array(0.0), jnp.array(0.0)
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            params, opt_state, loss, acc, rng = step(params, opt_state, idx, rng)
            # Accumulate on device; host sync once per epoch below.
            ep_loss += loss
            ep_acc += acc

        ep_loss = float(ep_loss) / steps_per_epoch
        ep_acc = float(ep_acc) / steps_per_epoch

        test_acc = np.mean(
            [float(evaluate(params, xte[i : i + 1000], yte[i : i + 1000]))
             for i in range(0, int(xte.shape[0]), 1000)]
        )
        print(
            f"epoch {epoch:3d} | train loss {ep_loss:.3f} batch acc {ep_acc:.3f} "
            f"| test acc {test_acc:.4f} | {time.time() - t0:.0f}s"
        )


if __name__ == "__main__":
    main()
