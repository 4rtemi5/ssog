# Copyright (c) 2026 Raphael Pisoni
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train a small ViT on CIFAR-100 with SSOG or dot-product attention (PyTorch).

Quick compare (enough to see the small-data gap)::

    uv run --extra examples-torch examples/train_cifar100_torch.py --attn ssog --epochs 10
    uv run --extra examples-torch examples/train_cifar100_torch.py --attn dot  --epochs 10

Full recipe defaults to 100 epochs (closer to the blog numbers). Downloads
CIFAR-100 via torchvision on first run.

Matched to the JAX example: pixels in ``[0, 1]``, pad-4 random crop +
horizontal flip p=0.5, AdamW (wd=0.05), 5-epoch warmup + cosine decay,
``drop_last`` so every step has a fixed batch size.

``torch.compile`` is on by default on CUDA (first epoch builds the graph).
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ssog.torch import ViT


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
    ap.add_argument("--data", default="./data", help="torchvision CIFAR-100 cache directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                    help="torch.compile the model (default: on when device is CUDA)")
    args = ap.parse_args()

    device = torch.device(args.device)
    do_compile = args.compile if args.compile is not None else device.type == "cuda"
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Same augment as the JAX script: RandomCrop(32, padding=4) + flip p=0.5.
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()
    train_ds = datasets.CIFAR100(args.data, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR100(args.data, train=False, download=True, transform=test_tf)

    pin = device.type == "cuda"
    workers = args.workers if pin else 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=pin, persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1000, shuffle=False,
        num_workers=workers, pin_memory=pin, persistent_workers=workers > 0,
    )

    model = ViT(
        num_classes=100, img_size=32, patch_size=2,
        dim=args.dim, depth=args.depth, num_heads=args.heads, attn=args.attn,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"attn={args.attn}  params={n_params / 1e6:.2f}M  device={device}  compile={do_compile}")
    if do_compile:
        model = torch.compile(model)

    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = min(5 * steps_per_epoch, total_steps - 1)

    # Minimal AdamW: weight decay on all params (no bias / LN exclusion).
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    warmup = LinearLR(
        opt, start_factor=1.0 / max(warmup_steps, 1), end_factor=1.0, total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(opt, T_max=max(total_steps - warmup_steps, 1), eta_min=0.0)
    sched = SequentialLR(opt, [warmup, cosine], milestones=[warmup_steps])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = torch.zeros((), device=device)
        ep_acc = torch.zeros((), device=device)
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.detach()
            ep_acc += (logits.argmax(dim=-1) == y).float().mean()

        ep_loss = float(ep_loss) / steps_per_epoch
        ep_acc = float(ep_acc) / steps_per_epoch

        model.eval()
        correct = torch.zeros((), device=device)
        total = 0
        with torch.inference_mode():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum()
                total += y.numel()
        test_acc = float(correct) / total
        print(
            f"epoch {epoch:3d} | train loss {ep_loss:.3f} batch acc {ep_acc:.3f} "
            f"| test acc {test_acc:.4f} | {time.time() - t0:.0f}s"
        )


if __name__ == "__main__":
    main()
