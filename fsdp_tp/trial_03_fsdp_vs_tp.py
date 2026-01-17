"""
FSDP vs Tensor Parallelism: Key Differences

This example shows how FSDP differs from TP in terms of:
1. What input each GPU gets
2. How weights are handled during computation
3. Communication patterns
"""

import torch

torch.manual_seed(42)

# =============================================================================
# Setup
# =============================================================================
batch_size = 8  # Total batch size
input_dim = 4
hidden_dim = 8
output_dim = 4
num_gpus = 4

# Full weights (same as before)
W1 = torch.randn(input_dim, hidden_dim)
W2 = torch.randn(hidden_dim, output_dim)

# Full batch of data
X_full = torch.randn(batch_size, input_dim)

print("=" * 70)
print("SETUP")
print("=" * 70)
print(f"Total batch size: {batch_size}")
print(f"X_full shape: {X_full.shape}")
print(f"W1 shape: {W1.shape}")
print(f"W2 shape: {W2.shape}")

# Ground truth output
Y_original = torch.relu(X_full @ W1) @ W2
print(f"Original output shape: {Y_original.shape}")

# =============================================================================
# TENSOR PARALLELISM (TP)
# =============================================================================
print("\n" + "=" * 70)
print("TENSOR PARALLELISM (TP)")
print("=" * 70)
print("""
Key characteristics:
- SAME input X on ALL GPUs (replicated)
- Weights SHARDED permanently across GPUs
- Each GPU computes PARTIAL result
- Communication: All-reduce to combine partial results
- Use case: Model too large for single GPU
""")

# TP: Same input on all GPUs
X_tp_gpu0 = X_full.clone()  # Same X
X_tp_gpu1 = X_full.clone()  # Same X
X_tp_gpu2 = X_full.clone()  # Same X
X_tp_gpu3 = X_full.clone()  # Same X

print(f"GPU 0 input shape: {X_tp_gpu0.shape} (SAME data)")
print(f"GPU 1 input shape: {X_tp_gpu1.shape} (SAME data)")
print(f"GPU 2 input shape: {X_tp_gpu2.shape} (SAME data)")
print(f"GPU 3 input shape: {X_tp_gpu3.shape} (SAME data)")

# TP: Sharded weights (column-wise for W1, row-wise for W2)
chunk = hidden_dim // num_gpus
W1_shards = [W1[:, i*chunk:(i+1)*chunk] for i in range(num_gpus)]
W2_shards = [W2[i*chunk:(i+1)*chunk, :] for i in range(num_gpus)]

print(f"\nW1 shard shape per GPU: {W1_shards[0].shape} (1/4 of columns)")
print(f"W2 shard shape per GPU: {W2_shards[0].shape} (1/4 of rows)")

# TP: Each GPU computes partial hidden, then partial output
H_partials = [X_full @ W1_shards[i] for i in range(num_gpus)]
H_relu_partials = [torch.relu(h) for h in H_partials]
Y_partials = [H_relu_partials[i] @ W2_shards[i] for i in range(num_gpus)]

# TP: All-reduce (sum) partial outputs
Y_tp = sum(Y_partials)

print(f"\nPartial output shape per GPU: {Y_partials[0].shape}")
print(f"After ALL-REDUCE: {Y_tp.shape}")
print(f"✅ TP matches original: {torch.allclose(Y_original, Y_tp)}")

# =============================================================================
# FSDP (Fully Sharded Data Parallel)
# =============================================================================
print("\n" + "=" * 70)
print("FSDP (Fully Sharded Data Parallel)")
print("=" * 70)
print("""
Key characteristics:
- DIFFERENT input (micro-batch) on each GPU
- Weights SHARDED for storage, but ALL-GATHERED before compute
- Each GPU computes FULL result on its micro-batch
- Communication: All-gather weights, reduce-scatter gradients
- Use case: Large model + large data, memory efficient training
""")

# FSDP: Different micro-batches on each GPU
micro_batch = batch_size // num_gpus  # 8 / 4 = 2 samples per GPU
X_fsdp_gpu0 = X_full[0*micro_batch:1*micro_batch]  # Samples 0-1
X_fsdp_gpu1 = X_full[1*micro_batch:2*micro_batch]  # Samples 2-3
X_fsdp_gpu2 = X_full[2*micro_batch:3*micro_batch]  # Samples 4-5
X_fsdp_gpu3 = X_full[3*micro_batch:4*micro_batch]  # Samples 6-7

print(f"GPU 0 input shape: {X_fsdp_gpu0.shape} (samples 0-1, DIFFERENT data)")
print(f"GPU 1 input shape: {X_fsdp_gpu1.shape} (samples 2-3, DIFFERENT data)")
print(f"GPU 2 input shape: {X_fsdp_gpu2.shape} (samples 4-5, DIFFERENT data)")
print(f"GPU 3 input shape: {X_fsdp_gpu3.shape} (samples 6-7, DIFFERENT data)")

# FSDP: Weights stored in shards (to save memory)
# Using a simple sharding scheme - flatten and split
W1_flat = W1.flatten()
W2_flat = W2.flatten()

W1_fsdp_shards = torch.chunk(W1_flat, num_gpus)
W2_fsdp_shards = torch.chunk(W2_flat, num_gpus)

print(f"\nW1 total params: {W1.numel()}, shard size: {W1_fsdp_shards[0].numel()}")
print(f"W2 total params: {W2.numel()}, shard size: {W2_fsdp_shards[0].numel()}")

# FSDP: Before compute, ALL-GATHER weights to reconstruct full weights
# This happens on each GPU (simulating what GPU 0 would do)
print("\n--- GPU 0's perspective ---")
print("Step 1: All-gather W1 shards from all GPUs...")
W1_gathered = torch.cat(W1_fsdp_shards).reshape(W1.shape)
print(f"  Reconstructed W1 shape: {W1_gathered.shape}")

print("Step 2: Compute forward pass with FULL weights on MY micro-batch...")
H_gpu0 = X_fsdp_gpu0 @ W1_gathered  # Full matmul, but only on 2 samples
H_relu_gpu0 = torch.relu(H_gpu0)

print("Step 3: Discard W1, all-gather W2 shards...")
W2_gathered = torch.cat(W2_fsdp_shards).reshape(W2.shape)
print(f"  Reconstructed W2 shape: {W2_gathered.shape}")

print("Step 4: Compute second layer...")
Y_fsdp_gpu0 = H_relu_gpu0 @ W2_gathered

print(f"\nGPU 0 output shape: {Y_fsdp_gpu0.shape} (only for its micro-batch)")

# Simulate all GPUs doing the same
X_fsdp_gpus = [X_fsdp_gpu0, X_fsdp_gpu1, X_fsdp_gpu2, X_fsdp_gpu3]
Y_fsdp_all = []
for i, X_micro in enumerate(X_fsdp_gpus):
    # Each GPU all-gathers weights, computes on its micro-batch
    W1_full = torch.cat(W1_fsdp_shards).reshape(W1.shape)
    W2_full = torch.cat(W2_fsdp_shards).reshape(W2.shape)
    Y_micro = torch.relu(X_micro @ W1_full) @ W2_full
    Y_fsdp_all.append(Y_micro)

# Combine outputs (in practice, each GPU keeps its own output for backward)
Y_fsdp_combined = torch.cat(Y_fsdp_all, dim=0)

print(f"\nCombined FSDP output shape: {Y_fsdp_combined.shape}")
print(f"✅ FSDP matches original: {torch.allclose(Y_original, Y_fsdp_combined)}")

# =============================================================================
# FSDP BACKWARD PASS: Where Reduce-Scatter Happens!
# =============================================================================
print("\n" + "=" * 70)
print("FSDP BACKWARD PASS (Reduce-Scatter)")
print("=" * 70)
print("""
During backward pass:
1. All-gather W2 (if discarded after forward)
2. Compute local gradients for W2 using local activations
3. REDUCE-SCATTER gradients: each GPU gets 1/N of the summed gradient
4. All-gather W1, compute local gradients
5. REDUCE-SCATTER gradients for W1
""")

# Let's simulate backward pass for W2 on GPU 0
# Assume simple MSE loss: L = sum((Y - target)^2)

# Each GPU has its own target (for its micro-batch)
targets = [torch.randn(micro_batch, output_dim) for _ in range(num_gpus)]

print("--- Simulating Backward Pass ---")

# Step 1: Each GPU computes LOCAL gradient for W2
# dL/dW2 = H_relu.T @ dL/dY, where dL/dY = 2*(Y - target)
local_grads_W2 = []
for i in range(num_gpus):
    # Recompute forward (in practice, activations are saved)
    X_micro = X_fsdp_gpus[i]
    W1_full = torch.cat(W1_fsdp_shards).reshape(W1.shape)
    W2_full = torch.cat(W2_fsdp_shards).reshape(W2.shape)
    
    H_micro = X_micro @ W1_full
    H_relu_micro = torch.relu(H_micro)
    Y_micro = H_relu_micro @ W2_full
    
    # Gradient of loss w.r.t. output
    dL_dY = 2 * (Y_micro - targets[i])
    
    # Gradient of W2: H_relu.T @ dL_dY
    local_grad_W2 = H_relu_micro.T @ dL_dY  # Shape: (hidden_dim, output_dim) = (8, 4)
    local_grads_W2.append(local_grad_W2)
    
print(f"GPU 0 local gradient for W2 shape: {local_grads_W2[0].shape}")
print(f"GPU 1 local gradient for W2 shape: {local_grads_W2[1].shape}")

# Step 2: REDUCE-SCATTER (showing GPU 0's perspective accurately)
print("\n--- REDUCE-SCATTER Operation (GPU 0's Perspective) ---")

# Each GPU's gradient is (8, 4) = 32 elements
# With 4 GPUs, each GPU is "responsible" for 8 elements (32 / 4 = 8)
grad_size = local_grads_W2[0].numel()  # 32
shard_size = grad_size // num_gpus      # 8

print(f"Total gradient size: {grad_size} elements")
print(f"Each GPU responsible for: {shard_size} elements")
print(f"GPU 0 is responsible for elements [0:8]")

# Flatten each GPU's local gradient
local_grads_flat = [g.flatten() for g in local_grads_W2]

print(f"\nGPU 0's local grad (flat): shape={local_grads_flat[0].shape}")
print(f"GPU 1's local grad (flat): shape={local_grads_flat[1].shape}")
print(f"GPU 2's local grad (flat): shape={local_grads_flat[2].shape}")
print(f"GPU 3's local grad (flat): shape={local_grads_flat[3].shape}")

# GPU 0 only receives elements [0:8] from each GPU
# This is the "scatter" part - each GPU sends its chunk to the responsible GPU
print("\n--- GPU 0 receives ONLY elements [0:8] from each GPU ---")
chunk_from_gpu0 = local_grads_flat[0][0:shard_size]  # GPU 0's elements [0:8]
chunk_from_gpu1 = local_grads_flat[1][0:shard_size]  # GPU 1's elements [0:8]
chunk_from_gpu2 = local_grads_flat[2][0:shard_size]  # GPU 2's elements [0:8]
chunk_from_gpu3 = local_grads_flat[3][0:shard_size]  # GPU 3's elements [0:8]

print(f"  Received from GPU 0: {chunk_from_gpu0.shape}")
print(f"  Received from GPU 1: {chunk_from_gpu1.shape}")
print(f"  Received from GPU 2: {chunk_from_gpu2.shape}")
print(f"  Received from GPU 3: {chunk_from_gpu3.shape}")

# GPU 0 sums ONLY these chunks - this is the "reduce" part
# GPU 0 NEVER sees elements [8:32] from any GPU!
print("\n--- GPU 0 sums ONLY its responsible chunk ---")
gpu0_grad_shard = chunk_from_gpu0 + chunk_from_gpu1 + chunk_from_gpu2 + chunk_from_gpu3

print(f"  GPU 0's final gradient shard: {gpu0_grad_shard.shape}")
print(f"  GPU 0 only ever allocated memory for {shard_size} elements, not {grad_size}!")

# Verify this matches what we'd get from the "wrong" approach
total_grad_W2 = sum(local_grads_W2)
total_grad_W2_flat = total_grad_W2.flatten()
expected_shard = total_grad_W2_flat[0:shard_size]
print(f"\n✅ GPU 0's shard matches expected: {torch.allclose(gpu0_grad_shard, expected_shard)}")

# Now show GPU 1's perspective
print("\n--- REDUCE-SCATTER Operation (GPU 1's Perspective) ---")
print(f"GPU 1 is responsible for elements [8:16]")

# GPU 1 only receives elements [8:16] from each GPU
chunk_from_gpu0_for_gpu1 = local_grads_flat[0][shard_size:2*shard_size]  # GPU 0's elements [8:16]
chunk_from_gpu1_for_gpu1 = local_grads_flat[1][shard_size:2*shard_size]  # GPU 1's elements [8:16]
chunk_from_gpu2_for_gpu1 = local_grads_flat[2][shard_size:2*shard_size]  # GPU 2's elements [8:16]
chunk_from_gpu3_for_gpu1 = local_grads_flat[3][shard_size:2*shard_size]  # GPU 3's elements [8:16]

print(f"  Received from GPU 0: elements [8:16] → {chunk_from_gpu0_for_gpu1.shape}")
print(f"  Received from GPU 1: elements [8:16] → {chunk_from_gpu1_for_gpu1.shape}")
print(f"  Received from GPU 2: elements [8:16] → {chunk_from_gpu2_for_gpu1.shape}")
print(f"  Received from GPU 3: elements [8:16] → {chunk_from_gpu3_for_gpu1.shape}")

# GPU 1 sums ONLY these chunks
gpu1_grad_shard = chunk_from_gpu0_for_gpu1 + chunk_from_gpu1_for_gpu1 + chunk_from_gpu2_for_gpu1 + chunk_from_gpu3_for_gpu1

print(f"\n  GPU 1's final gradient shard: {gpu1_grad_shard.shape}")

# Verify
expected_shard_gpu1 = total_grad_W2_flat[shard_size:2*shard_size]
print(f"✅ GPU 1's shard matches expected: {torch.allclose(gpu1_grad_shard, expected_shard_gpu1)}")

print("""
COMMUNICATION PATTERN in reduce-scatter:
┌─────────────────────────────────────────────────────────────────────────┐
│                        GPU 0    GPU 1    GPU 2    GPU 3                │
│                       [0:8]    [8:16]   [16:24]  [24:32]               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ GPU 0 sends:        keep   → GPU 1  → GPU 2  → GPU 3             │  │
│  │ GPU 1 sends:       → GPU 0   keep   → GPU 2  → GPU 3             │  │
│  │ GPU 2 sends:       → GPU 0 → GPU 1    keep   → GPU 3             │  │
│  │ GPU 3 sends:       → GPU 0 → GPU 1  → GPU 2    keep              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Each GPU receives 3 chunks + keeps 1 chunk = 4 chunks of size 8       │
│  Each GPU sums its 4 chunks → final shard of size 8                    │
└─────────────────────────────────────────────────────────────────────────┘
""")

print("""
KEY INSIGHT: 
  - GPU 0 only ever sees 8 elements from each GPU (not 32!)
  - GPU 0 only ever sums 4 tensors of size 8 (not size 32!)
  - The full 32-element sum NEVER exists on any single GPU!
  
Meanwhile (in parallel):
  - GPU 1 receives elements [8:16] from all GPUs, sums them
  - GPU 2 receives elements [16:24] from all GPUs, sums them
  - GPU 3 receives elements [24:32] from all GPUs, sums them
""")

print("""
This is the key insight of FSDP!

STORAGE:
  - Each GPU only STORES 1/N of weights
  - Each GPU only STORES 1/N of gradients  <-- from reduce-scatter
  - Each GPU only STORES 1/N of optimizer states

COMPUTE:
  - All-gather weights → full weights temporarily
  - Compute forward/backward with full weights
  - Reduce-scatter gradients → only keep 1/N
  - Update only YOUR shard of weights with YOUR shard of gradients
""")

# =============================================================================
# SIDE-BY-SIDE COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: TP vs FSDP")
print("=" * 70)

comparison = """
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│     Aspect          │   Tensor Parallelism     │         FSDP             │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Input per GPU       │ SAME (replicated)        │ DIFFERENT (micro-batch)  │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Weights per GPU     │ Shard (permanent)        │ Shard (gathered before   │
│                     │                          │ compute, then discarded) │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Computation         │ Partial (each GPU does   │ Full (each GPU does      │
│                     │ part of the math)        │ complete forward pass)   │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Communication       │ All-reduce outputs       │ All-gather weights,      │
│                     │                          │ reduce-scatter gradients │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Memory savings      │ Weights split, but       │ Weights, gradients, and  │
│                     │ activations replicated   │ optimizer states split   │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Primary use case    │ Single very large layer  │ Training large models    │
│                     │ (e.g., huge FFN)         │ with limited GPU memory  │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Scales best with    │ Model size               │ Both model & data size   │
└─────────────────────┴──────────────────────────┴──────────────────────────┘

Memory at any moment:
- TP GPU:   1/N weights + FULL activations for entire batch
- FSDP GPU: 1/N weights (1/N or full during compute) + 1/N activations
"""

print(comparison)

# =============================================================================
# KEY INSIGHT
# =============================================================================
print("\n" + "=" * 70)
print("KEY INSIGHT: Why Your Question Makes Sense")
print("=" * 70)
print("""
Both TP and FSDP shard weights, but for DIFFERENT reasons:

TP shards weights to PARALLELIZE COMPUTATION:
  - The math itself is split across GPUs
  - Each GPU does 1/N of the matmul
  - Requires same input to produce mathematically correct result

FSDP shards weights to SAVE MEMORY:
  - The math is NOT split (each GPU does full matmul)
  - Weights are temporarily reconstructed for compute
  - Different inputs allow data parallelism (faster training)

Think of it this way:
  - TP:   "Let's divide this ONE calculation across GPUs"
  - FSDP: "Let's divide the MEMORY across GPUs, but each GPU 
           does its own complete calculation on different data"
""")
