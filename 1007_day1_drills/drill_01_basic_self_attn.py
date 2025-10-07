import torch
import torch.nn.functional as F

def self_attention(Q, K, V, mask=None):
    """
    Basic scaled dot-product attention.
    Shapes:
    - Q, K, V: [B, H, T, D]
    - mask: [B, 1, T, T] or None
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5  # [B, H, T, T]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)  # [B, H, T, D]
    return output, attn_weights

# ---- Example Run ----
B, H, T, D = 2, 4, 5, 64  # batch, heads, tokens, dim
Q = torch.randn(B, H, T, D)
K = torch.randn(B, H, T, D)
V = torch.randn(B, H, T, D)

out, attn = self_attention(Q, K, V)
print(out.shape)  # [2, 4, 5, 64]
