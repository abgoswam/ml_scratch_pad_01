import torch

def greedy_decode(logits):
    """logits: [B, Vocab]"""
    return torch.argmax(logits, dim=-1)

def beam_decode(logits, beam_width=3):
    """Return top-k tokens per batch"""
    topk = torch.topk(logits, beam_width, dim=-1)
    return topk.indices, topk.values  # [B, k], [B, k]

# ---- Example Run ----
B, V = 2, 10
logits = torch.randn(B, V)

print("Greedy:", greedy_decode(logits))
print("Beam:", beam_decode(logits, beam_width=3))
