import torch
import torch.nn as nn
import torch.optim as optim

def make_model():
    return nn.Linear(4,2, bias=True)

def average_gradients(models: list[torch.nn.Module]) -> None:
    """
    in-place average the gradients across all model replicas.
    After this call, every model in models should have identical, averaged gradients in its .grad fields.
    """
    # TODO: add code

    num_params = len(list(models[0].parameters()))
    print(f"num_params:{num_params}")

    for i in range(num_params):

        state = []
        for j in range(len(models)):
            ith_p = list(models[j].parameters())[i]

            # print(f"[{i}:{j}] ith_p:\n{ith_p}")
            print(f"[{i}:{j}] ith_p.grad:\n{ith_p.grad}")
            state.append(ith_p.grad)

        print(f"\n****state***:\n{state}")
        _sum = state[0]
        for k in range(1, len(models)):
            _sum += state[k]
        _avg = _sum / len(models)
        print(f"\n****avg***:\n{_avg}")

        for j in range(len(models)):
            ith_p = list(models[j].parameters())[i]
            ith_p.grad = _avg

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

print("A:\n")
print([p.grad for p in models[0].parameters()])
print("B:\n")
print([p.grad for p in models[1].parameters()])