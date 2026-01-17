import torch
import torch.nn as nn

# Parameters
batch_size = 2
n_tokens = 4
emb_size = 8
n_heads = 2
head_dim = emb_size // n_heads  # head_dim = 4

# Input tensor
x = torch.randn(batch_size, n_tokens, emb_size)  # Shape: (2, 4, 8)
print(f"Original x shape: {x.shape}")

# Your proposed approach:
q_layer = nn.Linear(head_dim, head_dim)  # Linear layer expecting input size 4

# First, reshape x
x_reshaped = x.view(batch_size, n_tokens, n_heads, head_dim)  
print(f"Reshaped x shape: {x_reshaped.shape}")  # Shape: (2, 4, 2, 4)

# Now try to apply the linear layer
try:
    q = q_layer(x_reshaped)  # This will cause an error!
    print(f"q.shape:{q.shape}")
except Exception as e:
    print(f"Error: {e}")