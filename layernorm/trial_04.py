import torch
import torch.nn as nn

# Manual LayerNorm to show what happens
def manual_layer_norm(x, weight, bias, eps=1e-5):
    # x shape: (batch, seq_len, emb_dim)
    mean = x.mean(dim=-1, keepdim=True)  # Mean over last dimension
    var = x.var(dim=-1, keepdim=True, unbiased=False)  # Variance over last dimension
    
    # Normalize
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # Scale and shift (if elementwise_affine=True)
    return weight * x_norm + bias

# Compare with PyTorch's LayerNorm
x = torch.randn(2, 4, 8)
layer_norm = nn.LayerNorm(8)

# PyTorch version
output_pytorch = layer_norm(x)

# Manual version
print(f"layer_norm.weight.shape: {layer_norm.weight.shape}")
print(f"layer_norm.bias.shape: {layer_norm.bias.shape}")
output_manual = manual_layer_norm(x, layer_norm.weight, layer_norm.bias)

print(f"Outputs are close: {torch.allclose(output_pytorch, output_manual)}")