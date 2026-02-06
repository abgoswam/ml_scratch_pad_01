"""                                                                                                                                                                                                                                  
RoPE (Rotary Position Embeddings) — Explained From Scratch                                                                                                                                                                         
==========================================================                                                                                                                                                                           

This script walks through RoPE step by step using tiny dimensions                                                                                                                                                                    
so you can see every number. No ML framework knowledge needed.  

Run: python trial_01_basic.py
"""

import torch
import math

torch.manual_seed(42)

# ============================================================
# PART 1: Why do we need position embeddings?
# ============================================================
#
# Attention treats tokens as a SET — it doesn't know order.
# "The cat sat" and "sat cat the" look identical to attention.
#
# Position embeddings encode WHERE each token is.
# RoPE does this by ROTATING the Q and K vectors based on position.

print("=" * 60)
print("PART 1: Setup — A tiny model")
print("=" * 60)

head_dim = 4          # Each attention head has 4 dimensions
                    # (Real models use 128. We use 4 to see everything.)

# RoPE rotates dimension PAIRS: (dim0, dim1) and (dim2, dim3)
# So we need 2 frequencies — one per pair.
num_freq = head_dim // 2  # = 2

# inv_freq controls "how fast" each pair rotates with position.
# Low freq → slow rotation (captures long-range patterns)
# High freq → fast rotation (captures nearby patterns)
base = 10000.0
inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

print(f"head_dim = {head_dim}")
print(f"num_freq = {num_freq} (one per dimension pair)")
print(f"inv_freq = {inv_freq.tolist()}")
print(f"  → Pair (dim0,dim1) rotates at speed {inv_freq[0]:.4f} radians/position")
print(f"  → Pair (dim2,dim3) rotates at speed {inv_freq[1]:.6f} radians/position")
print()

# ============================================================
# PART 2: Computing frequencies for each position
# ============================================================

print("=" * 60)
print("PART 2: Compute rotation angle for each position")
print("=" * 60)

positions = torch.arange(5)  # 5 tokens: positions [0, 1, 2, 3, 4]

# The key operation: inv_freq * position_id
# This is the matmul that the FP64 patch targets.
#
# For each position, multiply by each inv_freq:
#   freq[pair][pos] = inv_freq[pair] * position_id

freqs = torch.outer(positions.float(), inv_freq)  # shape: (5, 2)

print(f"positions = {positions.tolist()}")
print(f"\nFrequencies (rotation angles in radians):")
print(f"{'pos':>3}  {'pair(0,1)':>10}  {'pair(2,3)':>10}")
for p in range(5):
    print(f"{p:>3}  {freqs[p, 0]:>10.4f}  {freqs[p, 1]:>10.6f}")

print(f"\nNotice: pair(0,1) rotates ~100x faster than pair(2,3)")
print(f"This is like a clock: fast hand = seconds, slow hand = hours")
print()

# ============================================================
# PART 3: From frequencies to cos/sin
# ============================================================

print("=" * 60)
print("PART 3: cos and sin of the rotation angles")
print("=" * 60)

cos_vals = freqs.cos()
sin_vals = freqs.sin()

print(f"\ncos values:")
print(f"{'pos':>3}  {'pair(0,1)':>10}  {'pair(2,3)':>10}")
for p in range(5):
    print(f"{p:>3}  {cos_vals[p, 0]:>10.4f}  {cos_vals[p, 1]:>10.4f}")

print(f"\nsin values:")
print(f"{'pos':>3}  {'pair(0,1)':>10}  {'pair(2,3)':>10}")
for p in range(5):
    print(f"{p:>3}  {sin_vals[p, 0]:>10.4f}  {sin_vals[p, 1]:>10.4f}")

print(f"\nPosition 0: cos=1, sin=0 → no rotation (identity)")
print(f"Higher positions → bigger rotation angles")
print()

# ============================================================
# PART 4: Applying the rotation to a Q vector
# ============================================================

print("=" * 60)
print("PART 4: Rotating a Q vector")
print("=" * 60)

# Say we have a Q vector for a token at position 2
q = torch.tensor([1.0, 0.0, 1.0, 0.0])  # [dim0, dim1, dim2, dim3]
pos = 2

print(f"Original Q vector:  {q.tolist()}")
print(f"Position: {pos}")
print(f"Rotation angles: pair(0,1)={freqs[pos, 0]:.4f} rad, pair(2,3)={freqs[pos, 1]:.6f} rad")

# RoPE rotation formula for each pair (d_i, d_{i+1}):
#   rotated_d_i     = d_i * cos(freq) - d_{i+1} * sin(freq)
#   rotated_d_{i+1} = d_i * sin(freq) + d_{i+1} * cos(freq)
#
# This is literally 2D rotation! Same as rotating a point (x, y) by angle θ.

c0, s0 = cos_vals[pos, 0].item(), sin_vals[pos, 0].item()
c1, s1 = cos_vals[pos, 1].item(), sin_vals[pos, 1].item()

q_rotated = torch.tensor([
    q[0] * c0 - q[1] * s0,   # pair (dim0, dim1)
    q[0] * s0 + q[1] * c0,
    q[2] * c1 - q[3] * s1,   # pair (dim2, dim3)
    q[2] * s1 + q[3] * c1,
])

print(f"Rotated Q vector: [{q_rotated[0]:.4f}, {q_rotated[1]:.4f}, {q_rotated[2]:.4f}, {q_rotated[3]:.6f}]")
print()
print("Key insight: Two tokens at DIFFERENT positions get DIFFERENT rotations.")
print("When attention computes Q·K, the dot product naturally decays")
print("for distant positions — this is how RoPE encodes distance.")
print()

# ============================================================
# PART 5: The bfloat16 precision bug
# ============================================================

print("=" * 60)
print("PART 5: The bfloat16 precision bug")
print("=" * 60)
print()
print("The bug: CUDA matmul gives DIFFERENT bfloat16 results for")
print("the SAME position when the tensor has a different total size.")
print()

# Use a more realistic head_dim to see the effect
head_dim_real = 128
inv_freq_real = 1.0 / (base ** (torch.arange(0, head_dim_real, 2).float() / head_dim_real))

# Simulate what happens during RL training:
# - Rollout (vLLM): generates 32 tokens at a time
# - Actor (training): processes 512 tokens (packed sequences)

small_positions = torch.arange(32)       # rollout: 32 tokens
large_positions = torch.arange(512)      # actor: 512 packed tokens

# Do the matmul in bfloat16 (what the model actually uses)
inv_freq_bf16 = inv_freq_real.bfloat16()

# Small tensor matmul
small_freqs = torch.outer(small_positions.bfloat16(), inv_freq_bf16)

# Large tensor matmul
large_freqs = torch.outer(large_positions.bfloat16(), inv_freq_bf16)

# Compare position 10 — should be IDENTICAL since inv_freq and position are the same
small_pos10 = small_freqs[10]   # position 10 from the 32-wide matmul
large_pos10 = large_freqs[10]   # position 10 from the 512-wide matmul

diff = (small_pos10 - large_pos10).abs()
max_diff = diff.max().item()
num_different = (diff > 0).sum().item()

print(f"Rollout computes position 10 in a {small_positions.shape[0]}-wide matmul")
print(f"Actor   computes position 10 in a {large_positions.shape[0]}-wide matmul")
print(f"\nResults for position 10 (first 5 frequency values):")
print(f"  Rollout (32-wide):  {small_pos10[:5].float().tolist()}")
print(f"  Actor  (512-wide):  {large_pos10[:5].float().tolist()}")
print(f"\nDifferences:")
print(f"  Max absolute diff:      {max_diff}")
print(f"  Frequencies that differ: {num_different} out of {inv_freq_real.shape[0]}")
print()

if max_diff == 0:
    print("NOTE: On CPU, results may be identical. The bug manifests on CUDA")
    print("where the matmul kernel uses different tiling for different tensor sizes.")
    print("On a GPU, you'd see non-zero differences here.")
else:
    print(f"Same position, same inv_freq, but {num_different} values differ!")
    print("This difference propagates through cos/sin → attention → logprobs.")
print()

# ============================================================
# PART 6: The float64 fix
# ============================================================

print("=" * 60)
print("PART 6: The float64 fix")
print("=" * 60)
print()

# Do the SAME matmul in float64, then cast back to bfloat16
inv_freq_f64 = inv_freq_real.double()

small_freqs_fixed = torch.outer(small_positions.double(), inv_freq_f64).bfloat16()
large_freqs_fixed = torch.outer(large_positions.double(), inv_freq_f64).bfloat16()

small_pos10_fixed = small_freqs_fixed[10]
large_pos10_fixed = large_freqs_fixed[10]

diff_fixed = (small_pos10_fixed - large_pos10_fixed).abs()
max_diff_fixed = diff_fixed.max().item()
num_different_fixed = (diff_fixed > 0).sum().item()

print(f"With float64 matmul → cast back to bfloat16:")
print(f"  Max absolute diff:      {max_diff_fixed}")
print(f"  Frequencies that differ: {num_different_fixed} out of {inv_freq_real.shape[0]}")
print()
print("float64 has enough precision that the rounding to bfloat16 is")
print("always the same, regardless of tensor size. Bug eliminated.")
print()

# ============================================================
# PART 7: Summary
# ============================================================

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
RoPE in 4 sentences:
1. Each token's Q and K vectors are ROTATED based on position
2. The rotation angle = inv_freq * position_id
3. This makes attention naturally aware of token distances
4. The rotation is a simple 2D rotation applied to dimension pairs

The precision bug:
- The matmul (inv_freq * position_ids) is done in bfloat16
- CUDA's matmul kernel gives slightly different results for
different tensor sizes, even for the same position
- This makes actor and rollout disagree on logprobs

The fix:
- Do that ONE matmul in float64 (plenty of precision)
- Cast results back to bfloat16
- Now both actor and rollout get identical RoPE embeddings
- Cost: negligible (RoPE is tiny compared to attention/MLP)
""")