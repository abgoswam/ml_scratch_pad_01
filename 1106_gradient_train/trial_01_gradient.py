import torch
import torch.nn as nn
import torch.optim as optim

# Simple model: 1 Linear layer
model = nn.Linear(3, 1)   # input_dim=3, output_dim=1

# Dummy dataset: y = x1 + x2 + x3
X = torch.randn(10, 3)
y = X.sum(dim=1, keepdim=True)

optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

def print_grad_flow_and_norm(model):
    total_norm = 0
    print("\n---- Gradient Flow ----")
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"[NO GRAD] {name}")
        else:
            grad_mean = param.grad.mean().item()
            grad_norm = param.grad.norm(2).item()
            print(f"[GRAD] {name}: mean={grad_mean:.4e}, norm={grad_norm:.4e}")
            total_norm += grad_norm ** 2

    print("Total Gradient L2 Norm =", (total_norm ** 0.5))

for step in range(5):
    optimizer.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, y)
    loss.backward()

    print(f"\nStep {step} | Loss = {loss.item():.4f}")
    print_grad_flow_and_norm(model)

    optimizer.step()
