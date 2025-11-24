import torch
import torch.nn as nn

print("=== Demonstrating Symmetry Problem ===\n")

# Simple case: 1 input -> 2 outputs
x = torch.tensor([[1.0]], requires_grad=True)
print(f"Input: {x}")

# Create a linear layer and manually set weights to SAME constant
layer = nn.Linear(1, 2, bias=False)  # No bias to keep it simple

# Initialize weights to the SAME value (this is the problem!)
with torch.no_grad():
    layer.weight.fill_(0.5)  # Both weights = 0.5

print(f"Initial weights (SAME): {layer.weight}")
print(f"Weight shape: {layer.weight.shape}")  # [2, 1] - 2 outputs, 1 input
print(f"\nGradients wrt weights: {layer.weight.grad}")
print("="*30)

# Forward pass
y = layer(x)
print(f"Output: {y}")  # Both outputs will be the same!


# Simple loss
loss = y.sum()
print(f"Loss: {loss}")

# Backward pass
loss.backward()

print(f"\nGradients wrt weights: {layer.weight.grad}")
print(f"Are gradients the same? {layer.weight.grad[0] == layer.weight.grad[1]}")
