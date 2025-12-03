import torch
import torch.nn as nn
from typing import List

class Ensemble(nn.Module):
    def __init__(self, gnns: List, freeze_nn_parameters = False):
        self.n = len(gnns)
        self.alpha = nn.Parameter(torch.randn(self.n))
        self.gnns = gnns
        self.free_nn_parameters = freeze_nn_parameters

    def forward(self, x):
        if self.free_nn_parameters:
            for i in range(self.n):
                for param in self.gnns[i].parameters():
                    param.requires_grad = False

        # Solution 1
        y = torch.zeros(B, E2)
        for i in range(self.n):
            xi_prime = self.gnns[i](x)
            y += self.alpha[i] * xi_prime
        return y

        # Solution 2
        y = torch.cat([
            self.gnns[i](x)[..., None] * self.alpha[i]
            for i in range(self.n)
        ], axis=-1)
        return y.sum(axis=-1)

# follow up 1: 
    # compare sol_1 vs sol_2 in terms of latency vs peak memory
    # requires_grad = True
# questions about : 
    # FSDP, TP, PP
    # Mixed precision training