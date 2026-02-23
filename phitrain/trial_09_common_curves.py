"""
Common Mathematical Curves Used in ML

These curves show up everywhere in ML code — learning rate schedules, reward
shaping, noise generation, probability distributions. This script visualizes
each one so you can build intuition for what they look like and when to use them.

Generates 4 separate charts by logical grouping:
  1. Exponential family (decay, growth, shifted)
  2. Activation functions (sigmoid, tanh, softmax temperature)
  3. Random distributions (gaussian, exponential, uniform)
  4. Learning rate schedules (cosine, warmup+decay, step decay)

Run with: conda run -n pytorch_cpu_0606 python trial_09_common_curves.py
"""

import numpy as np
import matplotlib.pyplot as plt


n = 200
t = np.arange(n, dtype=float)
np.random.seed(42)


# ============================================================
# Chart 1: Exponential Family
# ============================================================
print("=" * 60)
print("Chart 1: Exponential Family")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Exponential Family", fontsize=14)

# Exponential decay: e^(-t/tau)
ax = axes[0]
for tau in [20, 60, 120]:
    ax.plot(t, np.exp(-t / tau), linewidth=2, label=f"tau={tau}")
ax.set_title("Exponential Decay: exp(-t/tau)")
ax.set_xlabel("t")
ax.set_ylabel("value")
ax.legend()
ax.set_ylim(-0.05, 1.05)
ax.text(0.02, 0.02, "Used in: LR decay, EMA weights,\ndiscount factors",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Saturating growth: 1 - e^(-t/tau)
ax = axes[1]
for tau in [20, 60, 120]:
    ax.plot(t, 1 - np.exp(-t / tau), linewidth=2, label=f"tau={tau}")
ax.set_title("Saturating Growth: 1 - exp(-t/tau)")
ax.set_xlabel("t")
ax.set_ylabel("value")
ax.legend()
ax.set_ylim(-0.05, 1.05)
ax.text(0.02, 0.98, "Used in: learning curves,\nwarmup, reward shaping",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Scaled + shifted: base + scale * (1 - e^(-t/tau))
ax = axes[2]
for base, scale, tau in [(0.3, 0.5, 60), (0.0, 1.0, 40), (0.5, 0.3, 100)]:
    y = base + scale * (1 - np.exp(-t / tau))
    ax.plot(t, y, linewidth=2, label=f"base={base}, scale={scale}, tau={tau}")
ax.set_title("Shifted Growth: base + scale*(1 - exp(-t/tau))")
ax.set_xlabel("t")
ax.set_ylabel("value")
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 1.15)
ax.text(0.02, 0.02, "Used in: synthetic reward curves\n(trial_08 uses this)",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

plt.tight_layout()
plt.savefig("plots/curves_1_exponential_family.png", dpi=120, bbox_inches="tight")
print("Saved: curves_1_exponential_family.png")


# ============================================================
# Chart 2: Activation Functions
# ============================================================
print("\n" + "=" * 60)
print("Chart 2: Activation Functions")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Activation Functions", fontsize=14)

# Sigmoid
x = np.linspace(-8, 8, 200)
ax = axes[0]
for k in [0.5, 1.0, 2.0]:
    ax.plot(x, 1 / (1 + np.exp(-k * x)), linewidth=2, label=f"k={k}")
ax.set_title("Sigmoid: 1 / (1 + exp(-kx))")
ax.set_xlabel("x")
ax.set_ylabel("value")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
ax.legend()
ax.text(0.02, 0.02, "Used in: classification,\ngating mechanisms",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Tanh
ax = axes[1]
for k in [0.5, 1.0, 2.0]:
    ax.plot(x, np.tanh(k * x), linewidth=2, label=f"k={k}")
ax.set_title("Tanh: tanh(kx)")
ax.set_xlabel("x")
ax.set_ylabel("value")
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
ax.legend()
ax.set_ylim(-1.2, 1.2)
ax.text(0.02, 0.02, "Used in: activations,\nnormalize to [-1, 1]",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Softmax temperature
ax = axes[2]
logits = np.array([-1.0, 0.5, 1.0, 2.0, 0.2])
temps = [0.3, 0.5, 1.0, 2.0, 5.0]
x_pos = np.arange(len(logits))
width = 0.15
for i, temp in enumerate(temps):
    scaled = logits / temp
    exp_scaled = np.exp(scaled - scaled.max())
    probs = exp_scaled / exp_scaled.sum()
    ax.bar(x_pos + i * width, probs, width, label=f"T={temp}")
ax.set_title("Softmax Temperature Effect")
ax.set_xlabel("Token index")
ax.set_ylabel("Probability")
ax.set_xticks(x_pos + width * 2)
ax.set_xticklabels([f"logit={l}" for l in logits], fontsize=7)
ax.legend(fontsize=8)
ax.text(0.02, 0.98, "Low T -> greedy\nHigh T -> uniform",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

plt.tight_layout()
plt.savefig("plots/curves_2_activations.png", dpi=120, bbox_inches="tight")
print("Saved: curves_2_activations.png")


# ============================================================
# Chart 3: Random Distributions
# ============================================================
print("\n" + "=" * 60)
print("Chart 3: Random Distributions")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Random Distributions", fontsize=14)

# Gaussian
ax = axes[0]
for mu, sigma in [(0, 1), (0, 0.5), (2, 1.5)]:
    samples = np.random.normal(mu, sigma, 5000)
    ax.hist(samples, bins=60, alpha=0.5, density=True, label=f"mu={mu}, sigma={sigma}")
ax.set_title("Gaussian: N(mu, sigma)")
ax.set_xlabel("value")
ax.set_ylabel("density")
ax.legend(fontsize=8)
ax.set_xlim(-5, 7)
ax.text(0.02, 0.98, "Used in: weight init, noise,\ndiffusion models",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Exponential
ax = axes[1]
for scale in [0.5, 1.0, 2.0]:
    samples = np.random.exponential(scale, 5000)
    ax.hist(samples, bins=60, alpha=0.5, density=True, range=(0, 8),
            label=f"mean={scale}")
ax.set_title("Exponential: Exp(mean)")
ax.set_xlabel("value")
ax.set_ylabel("density")
ax.legend()
ax.text(0.02, 0.98, "Used in: time gaps between\nevents (trial_08 uses this)",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Uniform
ax = axes[2]
for low, high in [(0, 1), (-2, 2), (0.5, 1.5)]:
    samples = np.random.uniform(low, high, 5000)
    ax.hist(samples, bins=60, alpha=0.5, density=True, label=f"[{low}, {high}]")
ax.set_title("Uniform: U(low, high)")
ax.set_xlabel("value")
ax.set_ylabel("density")
ax.legend()
ax.set_xlim(-3, 3)
ax.text(0.02, 0.98, "Used in: dropout, random\nsampling, augmentation",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

plt.tight_layout()
plt.savefig("plots/curves_3_distributions.png", dpi=120, bbox_inches="tight")
print("Saved: curves_3_distributions.png")


# ============================================================
# Chart 4: Learning Rate Schedules
# ============================================================
print("\n" + "=" * 60)
print("Chart 4: Learning Rate Schedules")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Learning Rate Schedules", fontsize=14)

# Cosine annealing
ax = axes[0]
for eta_min in [0.0, 0.1]:
    lr = eta_min + 0.5 * (1 - eta_min) * (1 + np.cos(np.pi * t / n))
    ax.plot(t, lr, linewidth=2, label=f"eta_min={eta_min}")
ax.set_title("Cosine Annealing LR")
ax.set_xlabel("step")
ax.set_ylabel("learning rate")
ax.legend()
ax.set_ylim(-0.05, 1.15)
ax.text(0.02, 0.5, "Used in: most modern\ntraining pipelines",
        transform=ax.transAxes, fontsize=8, va="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Linear warmup + decay
ax = axes[1]
warmup_steps = 30
for total in [200, 150]:
    lr = np.zeros(total)
    lr[:warmup_steps] = np.linspace(0, 1, warmup_steps)
    lr[warmup_steps:] = np.linspace(1, 0, total - warmup_steps)
    ax.plot(lr[:n], linewidth=2, label=f"total={total}, warmup={warmup_steps}")
ax.set_title("Linear Warmup + Decay")
ax.set_xlabel("step")
ax.set_ylabel("learning rate")
ax.axvline(warmup_steps, color="gray", linestyle=":", alpha=0.5, label="warmup end")
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 1.15)
ax.text(0.02, 0.5, "Used in: transformer training\n(warmup prevents instability)",
        transform=ax.transAxes, fontsize=8, va="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Step decay
ax = axes[2]
for gamma in [0.5, 0.1]:
    lr = np.ones(n)
    for milestone in [60, 120, 160]:
        lr[milestone:] *= gamma
    ax.plot(t, lr, linewidth=2, label=f"gamma={gamma}")
ax.set_title("Step Decay LR")
ax.set_xlabel("step")
ax.set_ylabel("learning rate")
ax.legend()
ax.set_yscale("log")
ax.text(0.02, 0.5, "Used in: classical CV\n(ResNet milestones)",
        transform=ax.transAxes, fontsize=8, va="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

plt.tight_layout()
plt.savefig("plots/curves_4_lr_schedules.png", dpi=120, bbox_inches="tight")
print("Saved: curves_4_lr_schedules.png")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Chart 1 — Exponential Family:
  exp(-t/tau)                     Decay toward 0 (LR decay, EMA, discount)
  1 - exp(-t/tau)                 Growth toward 1 (learning curves, warmup)
  base + scale*(1 - exp(-t/tau))  Shifted growth (synthetic reward curves)

Chart 2 — Activation Functions:
  1 / (1 + exp(-kx))             Sigmoid (classification, gating)
  tanh(kx)                       Tanh (activations, normalize to [-1,1])
  softmax(x/T)                   Temperature scaling (sampling, exploration)

Chart 3 — Random Distributions:
  N(mu, sigma)                   Gaussian (weight init, noise, diffusion)
  Exp(mean)                      Exponential (time gaps, wait times)
  U(low, high)                   Uniform (dropout, random sampling)

Chart 4 — Learning Rate Schedules:
  0.5*(1 + cos(pi*t/T))          Cosine annealing (modern default)
  linear warmup + linear decay   Transformer training (prevents instability)
  multiply by gamma at steps     Step decay (classical CV, ResNet)
""")
