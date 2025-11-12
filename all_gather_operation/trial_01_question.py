import torch
import torch.nn as nn
import torch.optim as optim

def make_model():
    return nn.Linear(4,2, bias=False)

def average_gradients(models: list[torch.nn.Module]) -> None:
    """
    in-place average the gradients across all model replicas.
    After this call, every model in models should have identical, averaged gradients in its .grad fields.
    """
    # TODO: add code

models = [make_model(), make_model()]
opt = optim.Adam(models[0].parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# dummy input batches for each replica
x1, y1 = torch.randn(5,4), torch.randn(5,2)
x2, y2 = torch.randn(5,4), torch.randn(5,2) 

for m, (x, y) in zip(models, [(x1, y1), (x2, y2)]):
    out = m(x)
    loss = loss_fn(out, y)
    loss.backward()

average_gradients(models)