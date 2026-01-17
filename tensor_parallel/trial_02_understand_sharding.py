"""
Tensor Parallelism: Understanding ColwiseParallel and RowwiseParallel

This example demonstrates the math behind tensor parallelism WITHOUT needing
multiple GPUs. We simulate what happens when weights are sharded.
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

# =============================================================================
# Simple 2-layer MLP: Input -> W1 -> ReLU -> W2 -> Output
# =============================================================================

# Dimensions
batch_size = 3  # Can be any value - independent of parallelism!
input_dim = 4
hidden_dim = 8  # Must be divisible by num_gpus
output_dim = 4
num_gpus = 4  # Simulating 4 GPUs

# Create input and weights
X = torch.randn(batch_size, input_dim)
W1 = torch.randn(input_dim, hidden_dim)  # Shape: (4, 8)
W2 = torch.randn(hidden_dim, output_dim)  # Shape: (8, 4)

print("=" * 60)
print("ORIGINAL (No Parallelism)")
print("=" * 60)
print(f"Input X shape: {X.shape}")
print(f"W1 shape: {W1.shape}")
print(f"W2 shape: {W2.shape}")

# Standard forward pass (no parallelism)
H = X @ W1          # (2, 4) @ (4, 8) = (2, 8)
H_relu = torch.relu(H)
Y_original = H_relu @ W2  # (2, 8) @ (8, 4) = (2, 4)

print(f"\nHidden H shape: {H.shape}")
print(f"Output Y shape: {Y_original.shape}")
print(f"\nOutput Y:\n{Y_original}")

# =============================================================================
# COLWISE PARALLEL on W1
# Split W1 columns across GPUs. Each GPU computes partial hidden states.
# =============================================================================

print("\n" + "=" * 60)
print("COLWISE PARALLEL on W1")
print("=" * 60)

# Split W1 by columns
chunk_size = hidden_dim // num_gpus  # 8 / 4 = 2 columns per GPU

W1_gpu0 = W1[:, 0*chunk_size:1*chunk_size]  # (4, 2) - columns 0-1
W1_gpu1 = W1[:, 1*chunk_size:2*chunk_size]  # (4, 2) - columns 2-3
W1_gpu2 = W1[:, 2*chunk_size:3*chunk_size]  # (4, 2) - columns 4-5
W1_gpu3 = W1[:, 3*chunk_size:4*chunk_size]  # (4, 2) - columns 6-7

print(f"W1_gpu0 shape: {W1_gpu0.shape} (columns 0-1)")
print(f"W1_gpu1 shape: {W1_gpu1.shape} (columns 2-3)")
print(f"W1_gpu2 shape: {W1_gpu2.shape} (columns 4-5)")
print(f"W1_gpu3 shape: {W1_gpu3.shape} (columns 6-7)")

# Each GPU computes its portion (input X is REPLICATED on all GPUs)
H_gpu0 = X @ W1_gpu0  # (3, 4) @ (4, 2) = (3, 2)
H_gpu1 = X @ W1_gpu1  # (3, 4) @ (4, 2) = (3, 2)
H_gpu2 = X @ W1_gpu2  # (3, 4) @ (4, 2) = (3, 2)
H_gpu3 = X @ W1_gpu3  # (3, 4) @ (4, 2) = (3, 2)

print(f"\nH_gpu0 shape: {H_gpu0.shape}")
print(f"H_gpu1 shape: {H_gpu1.shape}")
print(f"H_gpu2 shape: {H_gpu2.shape}")
print(f"H_gpu3 shape: {H_gpu3.shape}")

# The hidden activations are now SHARDED across GPUs
# Each GPU has 2 hidden dims
H_reconstructed = torch.cat([H_gpu0, H_gpu1, H_gpu2, H_gpu3], dim=1)  # (2, 8)

print(f"\nVerify ColwiseParallel: H matches? {torch.allclose(H, H_reconstructed)}")

# =============================================================================
# ROWWISE PARALLEL on W2
# Split W2 rows across GPUs. Takes sharded input, produces partial outputs.
# =============================================================================

print("\n" + "=" * 60)
print("ROWWISE PARALLEL on W2")
print("=" * 60)

# Apply ReLU to sharded hidden states (each GPU applies ReLU locally)
H_relu_gpu0 = torch.relu(H_gpu0)
H_relu_gpu1 = torch.relu(H_gpu1)
H_relu_gpu2 = torch.relu(H_gpu2)
H_relu_gpu3 = torch.relu(H_gpu3)

# Split W2 by rows (to match the sharded hidden states)
W2_gpu0 = W2[0*chunk_size:1*chunk_size, :]  # (2, 4) - rows 0-1
W2_gpu1 = W2[1*chunk_size:2*chunk_size, :]  # (2, 4) - rows 2-3
W2_gpu2 = W2[2*chunk_size:3*chunk_size, :]  # (2, 4) - rows 4-5
W2_gpu3 = W2[3*chunk_size:4*chunk_size, :]  # (2, 4) - rows 6-7

print(f"W2_gpu0 shape: {W2_gpu0.shape} (rows 0-1)")
print(f"W2_gpu1 shape: {W2_gpu1.shape} (rows 2-3)")
print(f"W2_gpu2 shape: {W2_gpu2.shape} (rows 4-5)")
print(f"W2_gpu3 shape: {W2_gpu3.shape} (rows 6-7)")

# Each GPU computes PARTIAL output
Y_partial_gpu0 = H_relu_gpu0 @ W2_gpu0  # (3, 2) @ (2, 4) = (3, 4)
Y_partial_gpu1 = H_relu_gpu1 @ W2_gpu1  # (3, 2) @ (2, 4) = (3, 4)
Y_partial_gpu2 = H_relu_gpu2 @ W2_gpu2  # (3, 2) @ (2, 4) = (3, 4)
Y_partial_gpu3 = H_relu_gpu3 @ W2_gpu3  # (3, 2) @ (2, 4) = (3, 4)

print(f"\nY_partial_gpu0 shape: {Y_partial_gpu0.shape}")
print(f"Y_partial_gpu1 shape: {Y_partial_gpu1.shape}")
print(f"Y_partial_gpu2 shape: {Y_partial_gpu2.shape}")
print(f"Y_partial_gpu3 shape: {Y_partial_gpu3.shape}")

# ALL-REDUCE: Sum partial outputs from all GPUs
# This is the KEY communication step in RowwiseParallel!
Y_parallel = Y_partial_gpu0 + Y_partial_gpu1 + Y_partial_gpu2 + Y_partial_gpu3  # (2, 4)

print(f"\nAfter ALL-REDUCE, Y_parallel shape: {Y_parallel.shape}")

# =============================================================================
# VERIFY: Parallel output matches original!
# =============================================================================

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
print(f"\nOriginal output Y:\n{Y_original}")
print(f"\nParallel output Y:\n{Y_parallel}")
print(f"\n✅ Outputs match? {torch.allclose(Y_original, Y_parallel)}")

# =============================================================================
# WHY DOES THIS WORK? (The Math)
# =============================================================================

print("\n" + "=" * 60)
print("WHY IT WORKS (Matrix Math) - 4 GPU Example")
print("=" * 60)
print("""
Original:    Y = X @ W1 @ W2

Split W1 by columns: W1 = [W1_0 | W1_1 | W1_2 | W1_3]
Split W2 by rows:    W2 = [W2_0]
                          [W2_1]
                          [W2_2]
                          [W2_3]

Then:
    H = X @ W1 = X @ [W1_0 | W1_1 | W1_2 | W1_3] 
              = [X @ W1_0 | X @ W1_1 | X @ W1_2 | X @ W1_3] 
              = [H_0 | H_1 | H_2 | H_3]
    
    Y = H @ W2 = [H_0 | H_1 | H_2 | H_3] @ [W2_0]
                                           [W2_1]
                                           [W2_2]
                                           [W2_3]
              = H_0 @ W2_0 + H_1 @ W2_1 + H_2 @ W2_2 + H_3 @ W2_3
              (This is the ALL-REDUCE!)

Each GPU computes H_i @ W2_i, then we sum (all-reduce) to get Y.
No single GPU ever needs the full weight matrices!

With 4 GPUs, each GPU only holds 1/4 of each weight matrix.
""")
