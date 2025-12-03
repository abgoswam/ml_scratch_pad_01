import torch
import torch.nn as nn
from typing import List
# You are given a list of n untrained neural networks: [nn_1, nn_2, ..., nn_n].

# Each network takes an input tensor x of size (B, E_1) and produces an output tensor of the shape (B, E_2).
# Create nn.Module that takes a list of these `n` modules and performs a weighted sum of their outputs. 
# The weights should be trainable parameters of your module.


class Ensenble(nn.Module):
    def __init__(self, gnns: List, freeze_nn_parameters: bool = False):  #TODO
        self.n = len(gnns)
        self.alpha = nn.Parameter(torch.randn(self.n))
        self.gnns = gnns
        self.freeze_nn_parameters = freeze_nn_parameters
        # nn.Parameter([1]) # 1.
        # nn.Parameter(torch.randn(10)) # (10,) -> random normally distributed  
        
    def forward(self, x):
        # g1 : could be nn.Lineaar(E1, E2)
        # x : (B, E1)
        # x' : (B, E2)
        
        if self.freeze_nn_parameters:
            for i in range(self.n):
                for param in self.gnns[i].parameters():
                    param.requires_grad = False
            
        # Solution 1
        y = torch.zeros(B, E2) # TODO
        for i in range(self.n):
            xi_prime = self.gnns[i](x)  # (B, E2)
            y += self.alpha[i] * xi_prime
        
        # Solution 2
        # y = torch.cat([
        #     self.gnns[i](x)[..., None] * self.alpha[i]
        #     for i in range(self.n)
        # ], axis=-1)
        # return y.sum(axis=-1)
        
        
        # Latency
        # Peak memory
        
        return y
        
    