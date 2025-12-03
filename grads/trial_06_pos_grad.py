import torch
import matplotlib.pyplot as plt

# Simple 1D example to illustrate the concept
x = torch.linspace(-3, 3, 100, requires_grad=True)
y = x**2 + 0.1*x  # Simple quadratic with minimum around x=0

# Compute gradients at different points
gradients = []
positions = []

for i in range(0, 100, 10):
    if x.grad is not None:
        x.grad.zero_()
    
    loss = y[i]
    loss.backward(retain_graph=True)
    
    positions.append(x[i].item())
    gradients.append(abs(x.grad[i].item()))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x.detach(), y.detach(), 'b-', linewidth=2)
plt.xlabel('Parameter Value')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(positions, gradients, 'ro-', linewidth=2)
plt.xlabel('Parameter Value')
plt.ylabel('Absolute Gradient')
plt.title('Gradient Magnitude')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()