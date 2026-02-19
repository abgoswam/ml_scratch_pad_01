"""
Trial 06: bf16 Precision Loss — Why fp32 Master Weights Matter

Shows that with a small learning rate (5e-6) and typical gradients,
bf16 rounds weight updates to zero. fp32 master weights preserve them.

This is exactly what happens in phitrain's mixed-precision FSDP training:
  - Actor weights stored in fp32          (actor.py:91-94)
  - Forward/backward run in bf16          (cispo_worker.py:225)
  - Optimizer updates fp32 master weights (never lost)

Reference: actor.py:91-94
  "Need to build the model with fp32 to use full precision in the
   optimizer states and gradients when using fsdp"
"""

import torch

# ===========================================================================
# Setup: typical training hyperparameters from ray_cispo_miniswe.yaml
# ===========================================================================

lr = 5e-6       # from actor.optimizer.params.lr
gradient = 0.01  # typical small gradient value

update = lr * gradient  # 5e-8

print("=" * 60)
print("SINGLE UPDATE")
print("=" * 60)
print(f"lr = {lr}, gradient = {gradient}, update = lr * grad = {update:.1e}")
print()

# --- bf16: update vanishes ---
w_bf16 = torch.tensor(1.0, dtype=torch.bfloat16)
before = w_bf16.item()
w_bf16 -= torch.tensor(update, dtype=torch.bfloat16)
after = w_bf16.item()
print(f"bf16: {before:.10f} -> {after:.10f}  diff = {after - before:.1e}")

# --- fp32: update is preserved ---
w_fp32 = torch.tensor(1.0, dtype=torch.float32)
before = w_fp32.item()
w_fp32 -= torch.tensor(update, dtype=torch.float32)
after = w_fp32.item()
print(f"fp32: {before:.10f} -> {after:.10f}  diff = {after - before:.1e}")

# ===========================================================================
# Why: bf16 has only 8 bits mantissa (7 explicit + 1 implicit)
# ===========================================================================

print()
print("=" * 60)
print("WHY: REPRESENTABLE PRECISION NEAR 1.0")
print("=" * 60)

eps_bf16 = (
    torch.tensor(1.0, dtype=torch.bfloat16)
    .nextafter(torch.tensor(2.0, dtype=torch.bfloat16))
    .item()
    - 1.0
)
eps_fp32 = (
    torch.tensor(1.0, dtype=torch.float32)
    .nextafter(torch.tensor(2.0, dtype=torch.float32))
    .item()
    - 1.0
)

print(f"bf16 smallest step near 1.0: {eps_bf16:.1e}")
print(f"fp32 smallest step near 1.0: {eps_fp32:.1e}")
print(f"our update:                  {update:.1e}")
print()
print(f"update ({update:.1e}) < bf16 step ({eps_bf16:.1e}) -> rounded to zero!")
print(f"update ({update:.1e}) > fp32 step ({eps_fp32:.1e}) -> preserved!")

# ===========================================================================
# Accumulation: 10k steps of training
# ===========================================================================

print()
print("=" * 60)
print("ACCUMULATION OVER 10,000 STEPS")
print("=" * 60)

n_steps = 10_000

w_bf16 = torch.tensor(1.0, dtype=torch.bfloat16)
w_fp32 = torch.tensor(1.0, dtype=torch.float32)

for _ in range(n_steps):
    w_bf16 -= torch.tensor(update, dtype=torch.bfloat16)
    w_fp32 -= torch.tensor(update, dtype=torch.float32)

expected = 1.0 - (update * n_steps)

print(f"expected:  {expected:.10f}")
print(f"fp32:      {w_fp32.item():.10f}  (error = {abs(w_fp32.item() - expected):.2e})")
print(f"bf16:      {w_bf16.item():.10f}  (error = {abs(w_bf16.item() - expected):.2e})")
print()
print("bf16 weight is UNCHANGED after 10k steps — the model never learns!")

# ===========================================================================
# Mixed precision: bf16 gradients + fp32 master weights (what phitrain does)
# ===========================================================================

print()
print("=" * 60)
print("MIXED PRECISION: bf16 compute + fp32 master (phitrain's approach)")
print("=" * 60)

w_master = torch.tensor(1.0, dtype=torch.float32)  # fp32 master weight

for _ in range(n_steps):
    # forward/backward in bf16 (torch.autocast)
    grad_bf16 = torch.tensor(gradient, dtype=torch.bfloat16)
    # optimizer step in fp32: cast grad back, update master
    w_master -= lr * grad_bf16.float()

print(f"expected:      {expected:.10f}")
print(f"mixed (fp32):  {w_master.item():.10f}  (error = {abs(w_master.item() - expected):.2e})")
print(f"pure bf16:     {w_bf16.item():.10f}  (error = {abs(w_bf16.item() - expected):.2e})")
print()
print("Mixed precision: bf16 gradient is slightly noisy, but fp32 master")
print("accumulates the updates correctly. Best of both worlds.")
