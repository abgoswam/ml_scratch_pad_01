"""
Gradient Norm: What it is and how it's calculated.

Grad norm = the total magnitude of all gradients across all parameters.
It tells you "how big of an update step the optimizer wants to take."

    grad_norm = sqrt( sum of (each gradient element)^2 )

Think of it like the Pythagorean theorem but in millions of dimensions.

"Norm" just means sqrt(sum of squares) — this is the L2 norm (the default).
Other norms exist (L1 = sum of abs values, L-inf = max abs value),
but in deep learning "grad norm" always means L2.
"""

import torch
import torch.nn as nn

# --- A tiny model: 2 inputs -> 3 hidden -> 1 output ---
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(2, 3),   # 6 weights + 3 biases = 9 params
    nn.ReLU(),
    nn.Linear(3, 1),   # 3 weights + 1 bias = 4 params
)
# Total: 13 parameters

# --- Forward pass with a simple MSE loss ---
x = torch.tensor([[1.0, 2.0]])
target = torch.tensor([[5.0]])

pred = model(x)
loss = nn.MSELoss()(pred, target)
loss.backward()

# --- Method 1: Compute grad norm manually ---
# Collect every individual gradient value, square it, sum, sqrt
total = 0.0
for name, p in model.named_parameters():
    grad = p.grad
    sum_of_squares = (grad ** 2).sum()  # sum of squares for this parameter
    total += sum_of_squares.item()
    print(f"  {name:10s}  shape={str(list(p.shape)):10s}  grad_norm={grad.norm():.4f}")

manual_grad_norm = total ** 0.5
print(f"\nManual grad norm:  {manual_grad_norm:.6f}")

# --- Method 2: Use PyTorch's built-in utility ---
# This is what training frameworks (including phitrain) use
pytorch_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
print(f"PyTorch grad norm: {pytorch_grad_norm:.6f}")

# --- They match ---
assert abs(manual_grad_norm - pytorch_grad_norm.item()) < 1e-5

# --- What does grad norm tell you? ---
print("\n--- Intuition ---")
print("Large grad norm  -> model is far from optimum, big updates needed")
print("Small grad norm  -> model is near optimum (convergence)")
print("                    OR the loss signal is weak (starvation)")
print()
print("In SFT:  grad_norm shrinks = convergence (good)")
print("In CISPO: grad_norm shrinks because EffBSZ -> 0 = starvation (bad)")

# --- Diagnosing training from loss + grad_norm together ---
print("\n--- Diagnosing training: what loss + grad_norm patterns mean ---")
print()
print("1. Loss decreasing, grad_norm decreasing:")
print("   -> Healthy convergence. Model is learning and approaching optimum.")
print()
print("2. Loss flat at 0, grad_norm ~ 0:")
print("   -> Done. Model predicts perfectly, no error, no gradient.")
print()
print("3. Loss flat at HIGH value, grad_norm steady & moderate:")
print("   -> Stuck. Gradients exist but cancel out or LR is wrong.")
print("   -> Model is moving in parameter space but going nowhere in loss space.")
print("   -> Like walking in circles.")
print()
print("4. Loss flat at HIGH value, grad_norm large & spiky:")
print("   -> Overshooting. Updates are too big, bouncing past the minimum.")
print("   -> Try reducing learning rate.")
print()
print("5. Loss flat at HIGH value, grad_norm shrinking toward 0:")
print("   -> Giving up. The training signal is vanishing.")
print("   -> Could be vanishing gradients, or (in RL/CISPO) starvation")
print("   ->   from EffBSZ -> 0. This is the scary case.")
print()
print("6. Loss nonzero, grad_norm = 0 exactly:")
print("   -> Frozen parameters. requires_grad=False on all params.")
print("   -> No gradients computed, so nothing to update.")
print("   -> Common when freezing a backbone and only training a head,")
print("   ->   or during inference with torch.no_grad().")
print()
print("Key insight: flat loss does NOT mean grad_norm = 0.")
print("  grad_norm = 0 means loss is flat w.r.t. every parameter (local minimum).")
print("  flat loss across steps just means updates aren't helping.")
print("  The gradients can be large but unproductive.")

# --- Demo: grad norm depends on how many samples contribute ---
print("\n--- Demo: effect of batch size on grad norm ---")
for n_samples in [1, 4, 16]:
    model.zero_grad()
    x = torch.randn(n_samples, 2)
    target = torch.ones(n_samples, 1) * 5.0
    pred = model(x)
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    print(f"  {n_samples:>2} samples -> grad_norm = {gn:.4f}")
