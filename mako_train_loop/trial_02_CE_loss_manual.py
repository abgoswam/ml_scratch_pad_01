import torch
import torch.nn.functional as F

# Create a small example
batch_size = 2
seq_len = 3
vocab_size = 5

# Set seed for reproducibility
torch.manual_seed(42)

# Create sample logits and targets
logits = torch.randn(batch_size * seq_len, vocab_size)  # [6, 5]
target = torch.randint(0, vocab_size, (batch_size * seq_len,))  # [6]

print("Sample logits shape:", logits.shape)
print("Sample targets:", target)
print("Sample logits:\n", logits)
print()

# Method 1: Built-in cross_entropy
loss_builtin = F.cross_entropy(logits, target)
print(f"Built-in cross_entropy loss: {loss_builtin.item():.6f}")

# Method 2: Manual implementation using softmax + gather
probs = F.softmax(logits, dim=1)
correct_class_probs = torch.gather(probs, 1, target.unsqueeze(1)).squeeze(1)
negative_log_probs = -torch.log(correct_class_probs)
loss_manual_1 = torch.mean(negative_log_probs)
print(f"Manual implementation 1:   {loss_manual_1.item():.6f}")

# Method 3: Manual implementation using log_softmax + indexing
log_probs = F.log_softmax(logits, dim=1)
selected_log_probs = log_probs[torch.arange(len(target)), target]
loss_manual_2 = -torch.mean(selected_log_probs)
print(f"Manual implementation 2:   {loss_manual_2.item():.6f}")

# Check if they're equal (within floating point precision)
print(f"\nAre they equal?")
print(f"Builtin ≈ Manual 1: {torch.allclose(loss_builtin, loss_manual_1)}")
print(f"Builtin ≈ Manual 2: {torch.allclose(loss_builtin, loss_manual_2)}")
print(f"Manual 1 ≈ Manual 2: {torch.allclose(loss_manual_1, loss_manual_2)}")

# Let's also verify the gradients are the same
logits_1 = logits.clone().requires_grad_(True)
logits_2 = logits.clone().requires_grad_(True)

loss_1 = F.cross_entropy(logits_1, target)
loss_2 = -torch.mean(F.log_softmax(logits_2, dim=1)[torch.arange(len(target)), target])

loss_1.backward()
loss_2.backward()

print(f"\nGradients are equal: {torch.allclose(logits_1.grad, logits_2.grad)}")