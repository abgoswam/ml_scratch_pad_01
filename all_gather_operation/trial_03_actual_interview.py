import torch
import torch.nn as nn
import torch.optim as optim

# tiny toy model
def make_model():
    return nn.Linear(4, 2, bias=False)

def average_gradients(models: list[torch.nn.Module]) -> None:
    """
    In-place, average the gradients across all model replicas.
    After this call, every model in `models` should have identical, 
    averaged gradients in its .grad fields.
    """
    #TODO: add code
    # 1. x_i [p.grad for p in models[i].parameters()]
    # 2. average of X_i  over all models
    # 3. assign average to each model[i].parameters() grad

    num_parameters = len(list(models[0].parameters()))
    print(f"num_parameters:{num_parameters}")
    
    for i in range(num_parameters):
        state = []
        for j in range(len(models)):
            ith_p = list(models[j].parameters())[i]
            state.append(ith_p.grad)

        print(f"state:{state}")
        _sum = state[0]
        for k in range(1, len(models)):
            _sum += state[k] 
        _avg = _sum / len(models)
        print(f"_avg:{_avg}")

        for j in range(len(models)):
            ith_p = list(models[j].parameters())[i]
            ith_p.grad = _avg

models = [make_model(), make_model()]
opt = optim.Adam(models[0].parameters(), lr=0.1)  # models[0] ?
loss_fn = nn.MSELoss()

# dummy input batches for each replica
x1, y1 = torch.randn(5, 4), torch.randn(5, 2)
x2, y2 = torch.randn(5, 4), torch.randn(5, 2)

# model.eval ?
# 0 out gradients

# if running more than one iteration, need to zero out grads but okay in this setup
# print(models[0].zero_grad)
print([p.grad for p in models[0].parameters()])

for m, (x, y) in zip(models, [(x1, y1), (x2, y2)]):
    out = m(x)               #  models[0]  (x1,y1)   # models[1]  (x2,y2) 
    loss = loss_fn(out, y)
    loss.backward()          # models[0].grad  # models[1].grad
    
print([p.grad for p in models[0].parameters()])

print("-----------")
average_gradients(models)
print("+++++++++++")

print([p.grad for p in models[0].parameters()])
print("***")
print([p.grad for p in models[1].parameters()])
opt.step()
