import torch



# ...existing code...
torch.manual_seed(123)

attn_weights = torch.rand(4, 6)

dropout = torch.nn.Dropout(0.5) # dropout rate of 50%

# Method 1: Compare original vs dropped out tensor
original = attn_weights.clone()
dropped_out = dropout(attn_weights)
print(f"Original:\n{original}")
print(f"After dropout:\n{dropped_out}")

# Method 2: Create binary mask showing which elements were dropped
dropout_mask = (dropped_out != 0).float()  # 1 where kept, 0 where dropped
print(f"Dropout mask (1=kept, 0=dropped):\n{dropout_mask}")

# Method 3: Show the scaling factor effect
# During training, kept values are scaled by 1/(1-p) where p is dropout rate
scaling_factor = 1.0 / (1.0 - 0.5)  # 1/0.5 = 2.0
expected_scaled = original * dropout_mask * scaling_factor
print(f"Expected scaled values:\n{expected_scaled}")
print(f"Actual dropout output:\n{dropped_out}")

# Method 4: Visualize which positions were zeroed out
zero_positions = (dropped_out == 0).float()
print(f"Positions that were zeroed (1=zeroed, 0=kept):\n{zero_positions}")

# Method 5: For better visualization, you can also print as integers
print(f"Dropout mask as int (1=kept, 0=dropped):\n{dropout_mask.int()}")

# Add this to see the expectation preservation:

print("\n=== Expectation Preservation Demo ===")
torch.manual_seed(42)
original_mean = attn_weights.mean()
print(f"Original mean: {original_mean:.4f}")

# Run dropout many times and average to see expectation
num_trials = 1000
sum_after_dropout = torch.zeros_like(attn_weights)

for _ in range(num_trials):
    torch.manual_seed(torch.randint(0, 10000, (1,)).item())  # Different seed each time
    dropped = dropout(attn_weights.clone())
    sum_after_dropout += dropped

average_after_dropout = sum_after_dropout / num_trials
print(f"Average after {num_trials} dropout trials: {average_after_dropout.mean():.4f}")
print(f"Difference from original: {abs(original_mean - average_after_dropout.mean()):.6f}")