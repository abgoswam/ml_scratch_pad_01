import torch
import torch.nn as nn

print("=== Why Zero Gradients? ===\n")

# Simple setup
x = torch.tensor([[1.0]], requires_grad=True)
layer = nn.Linear(1, 1, bias=False)
with torch.no_grad():
    layer.weight.fill_(0.5)

print(f"Initial weight: {layer.weight}")

# First forward/backward pass
print("\n--- First Pass ---")
y1 = layer(x)
loss1 = y1.sum()
print(f"Loss 1: {loss1}")
loss1.backward()
print(f"Gradient after 1st backward: {layer.weight.grad}")

# Second forward/backward pass WITHOUT zeroing gradients
print("\n--- Second Pass (NO zeroing) ---")
y2 = layer(x)
loss2 = y2.sum()
print(f"Loss 2: {loss2}")
loss2.backward()  # This ADDS to existing gradient!
print(f"Gradient after 2nd backward: {layer.weight.grad}")
print(f"Notice: gradient = {layer.weight.grad.item():.1f} = 1.0 + 1.0 (accumulated!)")

# Reset and try with zeroing
print("\n" + "="*50)
print("=== With Proper Gradient Zeroing ===")

layer2 = nn.Linear(1, 1, bias=False)
with torch.no_grad():
    layer2.weight.fill_(0.5)

# First pass
y = layer2(x)
loss = y.sum()
loss.backward()
print(f"Gradient after 1st backward: {layer2.weight.grad}")

# Zero gradients before second pass
layer2.zero_grad()  # This is the key!
print(f"Gradient after zeroing: {layer2.weight.grad}")

# Second pass
y = layer2(x)
loss = y.sum()
loss.backward()
print(f"Gradient after 2nd backward: {layer2.weight.grad}")
print("Now gradient is correct: 1.0 (not accumulated)")