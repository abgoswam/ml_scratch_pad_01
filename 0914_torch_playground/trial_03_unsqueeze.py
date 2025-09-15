import torch

print("=== torch.unsqueeze() Examples ===\n")

# Example 1: Basic unsqueeze - adding dimensions
print("Example 1: Basic unsqueeze operations")
x = torch.tensor([1, 2, 3, 4])
print(f"Original tensor: {x}")
print(f"Original shape: {x.shape}")  # torch.Size([4])

# Add dimension at position 0 (becomes first dimension)
y0 = torch.unsqueeze(x, dim=0)
print(f"unsqueeze(dim=0): {y0}")
print(f"Shape after unsqueeze(dim=0): {y0.shape}")  # torch.Size([1, 4])

# Add dimension at position 1 (becomes second dimension)
y1 = torch.unsqueeze(x, dim=1)
print(f"unsqueeze(dim=1): {y1}")
print(f"Shape after unsqueeze(dim=1): {y1.shape}")  # torch.Size([4, 1])

print()

# Example 2: Negative indexing
print("Example 2: Negative indexing")
x = torch.tensor([1, 2, 3])
print(f"Original: {x}, shape: {x.shape}")

# dim=-1 means add dimension at the end
y_neg1 = torch.unsqueeze(x, dim=-1)
print(f"unsqueeze(dim=-1): {y_neg1}")
print(f"Shape: {y_neg1.shape}")  # torch.Size([3, 1])

# dim=-2 means add dimension second from the end
y_neg2 = torch.unsqueeze(x, dim=-2)
print(f"unsqueeze(dim=-2): {y_neg2}")
print(f"Shape: {y_neg2.shape}")  # torch.Size([1, 3])

print()

# Example 3: 2D tensor unsqueeze
print("Example 3: 2D tensor unsqueeze")
x = torch.tensor([[1, 2], [3, 4]])
print(f"Original 2D tensor: {x}")
print(f"Original shape: {x.shape}")  # torch.Size([2, 2])

# Add dimension at different positions
y0 = torch.unsqueeze(x, dim=0)  # Shape: [1, 2, 2]
y1 = torch.unsqueeze(x, dim=1)  # Shape: [2, 1, 2]
y2 = torch.unsqueeze(x, dim=2)  # Shape: [2, 2, 1]

print(f"unsqueeze(dim=0) shape: {y0.shape}")
print(f"unsqueeze(dim=1) shape: {y1.shape}")
print(f"unsqueeze(dim=2) shape: {y2.shape}")

print()

# Example 4: Method vs function
print("Example 4: Tensor method vs torch function")
x = torch.tensor([1, 2, 3])

# Using torch function
y_func = torch.unsqueeze(x, dim=0)
print(f"torch.unsqueeze(x, dim=0): {y_func}, shape: {y_func.shape}")

# Using tensor method
y_method = x.unsqueeze(dim=0)
print(f"x.unsqueeze(dim=0): {y_method}, shape: {y_method.shape}")

# They're equivalent
print(f"Are they equal? {torch.equal(y_func, y_method)}")

print()

# Example 5: Common use case - preparing for batch operations
print("Example 5: Preparing single sample for batch operations")
# Simulate a single image (height=28, width=28)
single_image = torch.randn(28, 28)
print(f"Single image shape: {single_image.shape}")

# Add batch dimension to make it compatible with batch operations
batched_image = single_image.unsqueeze(0)  # Add batch dimension
print(f"After adding batch dim: {batched_image.shape}")  # [1, 28, 28]

# Add channel dimension too (common for CNN inputs)
cnn_ready = single_image.unsqueeze(0).unsqueeze(0)  # batch and channel dims
print(f"CNN-ready shape: {cnn_ready.shape}")  # [1, 1, 28, 28]

print()

# Example 6: Multiple unsqueeze operations
print("Example 6: Chaining unsqueeze operations")
x = torch.tensor([1, 2, 3])
print(f"Original: {x}, shape: {x.shape}")

# Chain multiple unsqueeze operations
result = x.unsqueeze(0).unsqueeze(2).unsqueeze(1)
print(f"After chaining unsqueeze(0).unsqueeze(2).unsqueeze(1):")
print(f"Shape: {result.shape}")
print(f"Result: {result}")

print()

# Example 7: Broadcasting preparation
print("Example 7: Preparing tensors for broadcasting")
a = torch.tensor([1, 2, 3])          # Shape: [3]
b = torch.tensor([10, 20])           # Shape: [2]

print(f"a: {a}, shape: {a.shape}")
print(f"b: {b}, shape: {b.shape}")

# Prepare for broadcasting: a should be [3, 1], b should be [1, 2]
a_reshaped = a.unsqueeze(1)          # Shape: [3, 1]
b_reshaped = b.unsqueeze(0)          # Shape: [1, 2]

print(f"a_reshaped: {a_reshaped}, shape: {a_reshaped.shape}")
print(f"b_reshaped: {b_reshaped}, shape: {b_reshaped.shape}")

# Now they can be broadcasted together
result = a_reshaped + b_reshaped     # Shape: [3, 2]
print(f"Broadcasting result: {result}")
print(f"Result shape: {result.shape}")

print()

# Example 8: Real ML scenario - attention weights
print("Example 8: ML scenario - preparing attention weights")
# Simulate attention scores for one head
attention_scores = torch.tensor([[0.2, 0.3, 0.5],
                                [0.1, 0.4, 0.5],
                                [0.3, 0.3, 0.4]])
print(f"Attention scores shape: {attention_scores.shape}")  # [3, 3]

# Add batch and head dimensions for multi-head attention
# Final shape should be [batch_size, num_heads, seq_len, seq_len]
batch_ready = attention_scores.unsqueeze(0).unsqueeze(0)
print(f"Ready for multi-head attention: {batch_ready.shape}")  # [1, 1, 3, 3]

print()

# Example 9: Comparison with squeeze (opposite operation, see trial_01_squeeze.py)
print("Example 9: Unsqueeze vs Squeeze (opposite operations)")
original = torch.tensor([1, 2, 3])
print(f"Original: {original}, shape: {original.shape}")

# Unsqueeze to add dimension
unsqueezed = original.unsqueeze(0)
print(f"After unsqueeze(0): {unsqueezed}, shape: {unsqueezed.shape}")

# Squeeze to remove dimension (back to original)
squeezed_back = unsqueezed.squeeze(0)
print(f"After squeeze(0): {squeezed_back}, shape: {squeezed_back.shape}")

print(f"Back to original? {torch.equal(original, squeezed_back)}")