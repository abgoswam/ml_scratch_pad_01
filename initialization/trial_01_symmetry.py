import torch
import torch.nn as nn

# single input with 3 features
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
print(f"{x}: {x.dtype}")

# Linear layer: 3 inputs → 2 outputs
layer = nn.Linear(3, 2, bias=True)

print("=== Random Initialization ===")
print("Weights:\n", layer.weight)
print("Bias:\n", layer.bias)

# Forward pass
y = layer(x)
print("Output:", y)
print("Grad fn of y:", y.grad_fn)   # shows computational graph

# Backward pass
loss = y.sum()
loss.backward()

print(f"loss: {loss}")

print("\nGradients wrt input x:\n", x.grad)
print("Gradients wrt weights:\n", layer.weight.grad)

print("done")
