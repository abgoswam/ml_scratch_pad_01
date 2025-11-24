import torch
import torch.nn as nn

# Reset gradients
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

layer_const = nn.Linear(3, 2, bias=True)

# Force ALL weights to 1.0 and biases to 0
nn.init.constant_(layer_const.weight, 1.0)
nn.init.constant_(layer_const.bias, 0.0)

print("\n=== Constant Initialization ===")
print("Weights:\n", layer_const.weight)
print("Bias:\n", layer_const.bias)

# Forward pass
y_const = layer_const(x)
print("Output:", y_const)
print("Grad fn of y_const:", y_const.grad_fn)

# Backward
loss = y_const.sum()
loss.backward()

print("\nGradients wrt input x:\n", x.grad)
print("Gradients wrt weights:\n", layer_const.weight.grad)
