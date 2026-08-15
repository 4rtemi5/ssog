# SSOG: A Few Gaussians Is All You Need

> **Read the full story on the blog: [A Few Gaussians Is All You Need](https://www.pisoni.ai/posts/ssog/)**

![A Gaussian attention field steering itself onto a bird](figures/steer.gif)

**SSOG (Separable Sum of Gaussians)** swaps the transformer's content-scored
attention for a learned geometric field. Each attention head is a handful of
Gaussian atoms over *relative position*, plus tiny bounded nudges that let
content **steer** the field without ever scoring anything. No query-key dot
products, no N×N attention matrix. Just two 1D filter passes per atom.

Turns out it works suspiciously well.

## Results

From scratch, matched light recipe, one GPU:

| ImageNet-1k, 90 epochs | Val acc |
| :--- | :---: |
| Fixed field (zero content awareness) | 63.2 – 63.4% |
| Dot-product attention (baseline) | 64.34% |
| SSOG + steering (μδ) | 64.48% ± 0.13 |
| SSOG + full conditioning (μδ, σδ, λ) | **65.28% ± 0.02** |

On small data the geometric prior is a superpower: **70.4% vs 53.1%** for the
dot baseline on CIFAR-100 at matched recipe. And because the field is defined
over coordinates rather than token indices, you get zero-shot resolution
transfer for free: train at 224², score *higher* at 288² without a single
gradient step (d384 champion: **72.0% → 73.7%**).

## Install

```bash
pip install -e .                # core (jax + flax)
pip install -e ".[examples]"    # + optax, datasets for the training example
```

## Quickstart

```python
from ssog import ViT

model = ViT(num_classes=100, attn="ssog")   # the Gaussian field
model = ViT(num_classes=100, attn="dot")    # its dot-product twin
```

Train both on CIFAR-100 and compare for yourself:

```bash
python examples/train_cifar100.py --attn ssog --epochs 10
python examples/train_cifar100.py --attn dot  --epochs 10
```

## How it works

![Anatomy of a Gaussian attention head](figures/anatomy.png)

Each head owns a few Gaussian atoms. An atom is five numbers: **μ** (where to
look, as an offset from the query), **σ** (how wide to stare) and **λ** (how
much this atom counts). The attention weight from token *p* to token *q* is
the value of the Gaussian mixture at their displacement. And since a 2D
Gaussian factorizes, applying it is two 1D passes per atom:

```python
y = jnp.einsum("biwprj,bjwpd->biwpdr", ay, v)   # down the rows, per atom
y = jnp.einsum("biwprk,bikpdr->biwpdr", ax, y)  # across the columns
y = jnp.einsum("pr,biwpdr->biwpd", lam, y)      # mix atoms
```

With `lookat=True` (the default), zero-initialized per-token probes predict
bounded residuals on μ, σ and λ behind cold-started gates: the model begins
life as a frozen geometric animal and opens the content taps itself, layer by
layer, exactly where steering pays off.

## What you get for free

**Interpretability.** "What did attention learn?" stops being a
heatmap-and-a-shrug question. Every head is a few blobs you can plot and read
with a ruler. Early layers become convolutions in disguise, middle layers turn
into strip detectors, late layers go global:

![Learned attention geometry per layer and head](figures/learned_geometry.png)

**Resolution transfer.** The field is defined over coordinates, not token
indices, so you can train at 224² and simply evaluate at a higher resolution.
Only the small position embedding needs a bilinear resize; the Gaussians
re-evaluate on the bigger grid like nothing happened. On the d384 champion,
288² scores **73.7%** — *higher* than the 72.0% at train resolution — without
one gradient step of fine-tuning.

**Speed that scales.** The N×N matrix never exists: O(N·√N·d) instead of
O(N²·d), and the gap over dot attention widens with every token you add.

## Scope

This is the minimal, readable implementation of the mechanism and its matched
baseline. The part you need if you want to try it, extend it, or break it.

## More

The full story and all the figures:
**[A Few Gaussians Is All You Need, on pisoni.ai](https://www.pisoni.ai/posts/ssog/)**
