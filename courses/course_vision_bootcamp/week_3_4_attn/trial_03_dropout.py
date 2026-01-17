import torch
import torch.nn as nn

torch.manual_seed(0)

# Create a dropout layer with p=0.5
drop = nn.Dropout(p=0.5)

# Input activations
x = torch.ones(10)

# Training mode: dropout active
drop.train()
train_out = drop(x)

# Eval mode: dropout disabled
drop.eval()
eval_out = drop(x)

print(train_out, eval_out)
