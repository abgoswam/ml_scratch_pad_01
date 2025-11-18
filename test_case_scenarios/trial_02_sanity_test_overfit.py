import torch
import torch.nn as nn
import torch.optim as optim

# Define a slightly larger model
model = nn.Sequential(
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Create a tiny dataset of exactly 5 samples
X = torch.tensor([
    [1., 2., 3.],
    [0., 1., 1.],
    [4., -1., 2.],
    [2., 2., 2.],
    [-1., -2., -3.]
])
y = torch.tensor([[6.], [2.], [5.], [6.], [-6.]])  # sum of components

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("Starting 5-sample overfit test...")

for step in range(501):
    optimizer.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step:3d} | Loss = {loss.item():.6f}")

# After training, predictions should match labels closely:
print("\nPredictions vs Target:")
print(torch.cat([preds, y], dim=1))  # [pred, target]
