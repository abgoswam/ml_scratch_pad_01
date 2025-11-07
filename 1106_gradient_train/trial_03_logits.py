import torch
import torch.nn.functional as F

# Suppose the model outputs logits for 3 classes:
logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.1, 0.1, 0.3]
    ])  # shape: [batch=1, classes=3]
print(f"logits shape: {logits.shape}")

# Convert logits to probabilities
probs = F.softmax(logits, dim=1)
print("Probabilities:", probs)
print(f"probs shape: {logits.shape}")

# Suppose the true class is class index 0
target = torch.tensor([0, 0])

# Compute cross entropy loss directly from logits (SOFTMAX NOT NEEDED HERE)
loss = F.cross_entropy(logits, target)
print("Cross Entropy Loss:", loss.item())

print("done")
