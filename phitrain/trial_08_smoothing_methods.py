"""
Smoothing Methods: Time-Weighted EMA vs Running Average vs Gaussian vs EMA

When monitoring training metrics (loss, reward, etc.), raw values are noisy.
Smoothing helps reveal trends. Different smoothing methods have different
tradeoffs in terms of lag, noise reduction, and how they handle uneven spacing.

This script compares four methods that wandb and other tools offer:

1. Time-Weighted EMA — EMA that accounts for uneven time spacing between points.
   Used by wandb as the default smoothing. Adjusts the decay factor based on the
   actual time delta between consecutive points, so a long gap between points
   produces less smoothing than a short gap.

2. Running Average — Simple moving average over a fixed window of N points.
   Equal weight to all points in the window. Shows as a separate line in wandb.

3. Gaussian — Weighted average using a Gaussian kernel. Points near the center
   of the window get more weight, points at the edges get less. Smoother than
   running average but more expensive to compute.

4. Exponential Moving Average (EMA) — Classic EMA with fixed decay factor.
   Does NOT account for time spacing. Each new point is blended with the
   previous smoothed value: s_t = (1 - smoothing) * x_t + smoothing * s_{t-1}.
   Uses wandb's convention: higher smoothing = smoother line.

Key differences:
- EMA vs Time-Weighted EMA: If data points are evenly spaced, they're identical.
  With uneven spacing, time-weighted EMA adapts its decay.
- Running Average vs Gaussian: Running average has equal weights (box filter),
  Gaussian has bell-curve weights. Gaussian produces smoother output.
- EMA vs Running Average: EMA is causal (only uses past data), running average
  can be centered (uses past + future). EMA has infinite memory (all past points
  contribute), running average has finite window.

Run with: conda run -n pytorch_cpu_0606 python trial_08_smoothing_methods.py
"""

import numpy as np
import matplotlib.pyplot as plt


def time_weighted_ema(values, times, smoothing=0.6):
    """Time-weighted EMA that adjusts decay based on time spacing.

    Uses wandb's convention where `smoothing` is the weight on the previous
    smoothed value (higher = smoother). The effective weight adapts to the
    time delta between points. For evenly spaced data, this reduces to
    standard EMA.

    """

    smoothed = [values[0]]
    # Convert smoothing to a time constant (tau)
    # When dt=1 (unit spacing), the effective smoothing should equal `smoothing`
    tau = -1.0 / np.log(smoothing)

    for i in range(1, len(values)):
        dt = times[i] - times[i - 1]
        # Adapt weight based on time gap
        effective_smoothing = np.exp(-dt / tau)
        smoothed.append((1 - effective_smoothing) * values[i] + effective_smoothing * smoothed[-1])

    return np.array(smoothed)


def running_average(values, window=10):
    """Simple moving average over a fixed window."""

    smoothed = np.copy(values).astype(float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(values[start:i + 1])
    return smoothed


def gaussian_smooth(values, window=10, sigma=None):
    """Gaussian-weighted moving average."""

    if sigma is None:
        sigma = window / 3.0

    smoothed = np.copy(values).astype(float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        end = i + 1
        indices = np.arange(start, end)
        weights = np.exp(-0.5 * ((indices - i) / sigma) ** 2)
        weights /= weights.sum()
        smoothed[i] = np.sum(values[start:end] * weights)
    return smoothed


def ema(values, smoothing=0.6):
    """Standard EMA (ignores time spacing).

    Uses wandb's convention where `smoothing` is the weight on the previous
    smoothed value: s_t = (1 - smoothing) * x_t + smoothing * s_{t-1}.
    Higher smoothing = smoother line (more weight on history).

    """

    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append((1 - smoothing) * values[i] + smoothing * smoothed[-1])
    return np.array(smoothed)


# ============================================================
# Generate synthetic training data
# ============================================================
print("=" * 60)
print("Generating synthetic training reward curve")
print("=" * 60)

np.random.seed(42)
n_steps = 200

# Simulate a reward curve: starts low, improves, with noise
trend = 0.3 + 0.5 * (1 - np.exp(-np.arange(n_steps) / 60))
noise = np.random.normal(0, 0.12, n_steps)
values = trend + noise

# Evenly spaced times
times_even = np.arange(n_steps, dtype=float)

# Unevenly spaced times (some steps take longer than others)
time_gaps = np.random.exponential(1.0, n_steps)
time_gaps[80:100] *= 5  # simulate a slow period (e.g., harder instances)
times_uneven = np.cumsum(time_gaps)

smoothing = 0.7
window = 15

# ============================================================
# Plot 1: Compare all four methods (evenly spaced data)
# ============================================================
print("\n" + "=" * 60)
print("Plot 1: All four methods on evenly spaced data")
print("=" * 60)

s_tw_ema = time_weighted_ema(values, times_even, smoothing=smoothing)
s_running = running_average(values, window=window)
s_gaussian = gaussian_smooth(values, window=window)
s_ema = ema(values, smoothing=smoothing)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Smoothing Methods Compared (Evenly Spaced Data)", fontsize=14)

methods = [
    ("Time-Weighted EMA", s_tw_ema, "tab:red"),
    ("Running Average", s_running, "tab:blue"),
    ("Gaussian", s_gaussian, "tab:green"),
    ("EMA", s_ema, "tab:purple"),
]

for ax, (name, smoothed, color) in zip(axes.flat, methods):
    ax.plot(times_even, values, alpha=0.25, color="gray", linewidth=0.8, label="Raw")
    ax.plot(times_even, smoothed, color=color, linewidth=2, label=name)
    ax.set_title(name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig("plots/smoothing_even_spacing.png", dpi=120, bbox_inches="tight")
print("Saved: smoothing_even_spacing.png")

# ============================================================
# Plot 2: EMA vs Time-Weighted EMA on unevenly spaced data
# ============================================================
print("\n" + "=" * 60)
print("Plot 2: EMA vs Time-Weighted EMA on UNEVENLY spaced data")
print("This is the key difference — standard EMA ignores time gaps,")
print("time-weighted EMA adapts its decay to the actual time spacing.")
print("=" * 60)

s_tw_ema_uneven = time_weighted_ema(values, times_uneven, smoothing=smoothing)
s_ema_uneven = ema(values, smoothing=smoothing)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("EMA vs Time-Weighted EMA on Unevenly Spaced Data", fontsize=14)

# Top: plot against step index (hides the uneven spacing)
axes[0].plot(range(n_steps), values, alpha=0.25, color="gray", linewidth=0.8, label="Raw")
axes[0].plot(range(n_steps), s_ema_uneven, color="tab:purple", linewidth=2, label="EMA (ignores time)")
axes[0].plot(range(n_steps), s_tw_ema_uneven, color="tab:red", linewidth=2, label="Time-Weighted EMA")
axes[0].axvspan(80, 100, alpha=0.15, color="orange", label="Slow period (5x time gaps)")
axes[0].set_xlabel("Step Index")
axes[0].set_ylabel("Reward")
axes[0].set_title("Plotted by Step Index")
axes[0].legend(loc="lower right")
axes[0].set_ylim(0, 1.2)

# Bottom: plot against actual time (reveals the uneven spacing)
axes[1].plot(times_uneven, values, alpha=0.25, color="gray", linewidth=0.8, label="Raw")
axes[1].plot(times_uneven, s_ema_uneven, color="tab:purple", linewidth=2, label="EMA (ignores time)")
axes[1].plot(times_uneven, s_tw_ema_uneven, color="tab:red", linewidth=2, label="Time-Weighted EMA")
axes[1].axvspan(times_uneven[80], times_uneven[100], alpha=0.15, color="orange", label="Slow period")
axes[1].set_xlabel("Wall-Clock Time")
axes[1].set_ylabel("Reward")
axes[1].set_title("Plotted by Wall-Clock Time (reveals uneven spacing)")
axes[1].legend(loc="lower right")
axes[1].set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig("plots/smoothing_uneven_spacing.png", dpi=120, bbox_inches="tight")
print("Saved: smoothing_uneven_spacing.png")

# ============================================================
# Plot 3: Effect of smoothing parameter
# ============================================================
print("\n" + "=" * 60)
print("Plot 3: Effect of smoothing parameter (alpha / window size)")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect of Smoothing Strength", fontsize=14)

# EMA with different smoothing values (wandb convention: higher = smoother)
for sm, ls in [(0.1, "--"), (0.4, "-"), (0.7, "-"), (0.95, "--")]:
    s = ema(values, smoothing=sm)
    axes[0].plot(times_even, s, linewidth=1.5, linestyle=ls, label=f"smoothing={sm}")
axes[0].plot(times_even, values, alpha=0.2, color="gray", linewidth=0.8, label="Raw")
axes[0].set_title("EMA: varying smoothing (higher = smoother)")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Reward")
axes[0].legend(loc="lower right", fontsize=9)
axes[0].set_ylim(0, 1.2)

# Running average with different window sizes
for w, ls in [(5, "--"), (15, "-"), (30, "-"), (60, "--")]:
    s = running_average(values, window=w)
    axes[1].plot(times_even, s, linewidth=1.5, linestyle=ls, label=f"window={w}")
axes[1].plot(times_even, values, alpha=0.2, color="gray", linewidth=0.8, label="Raw")
axes[1].set_title("Running Average: varying window (larger = more smoothing)")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Reward")
axes[1].legend(loc="lower right", fontsize=9)
axes[1].set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig("plots/smoothing_parameters.png", dpi=120, bbox_inches="tight")
print("Saved: smoothing_parameters.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Method              | Handles Uneven Spacing | Lag    | Noise Reduction
--------------------|------------------------|--------|----------------
Time-Weighted EMA   | Yes (adapts decay)     | Low    | Good
Running Average     | No (counts points)     | Medium | Good
Gaussian            | No (counts points)     | Medium | Best
EMA                 | No (fixed alpha)       | Low    | Good

Key takeaway:
- For evenly spaced data, EMA and Time-Weighted EMA are identical.
- For unevenly spaced data (common in RL training where some rollouts
  take longer), Time-Weighted EMA gives more consistent smoothing.
- In wandb: Time-Weighted EMA modifies the line in-place (increase the
  smoothing slider to see the effect). Running Average adds a separate line.
""")
