import torch
import torch.nn as nn
import torch.nn.functional as F

# tiny fake vocab
vocab_size = 8

# tiny fake Transformer-like model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, vocab_size)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        return self.linear2(h)  # logits [batch, seq, vocab]

teacher = TinyModel()
student = TinyModel()

batch = 1
seq_len = 5
embed_dim = 16

x = torch.randn(batch, seq_len, embed_dim)
