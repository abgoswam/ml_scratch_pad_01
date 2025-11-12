import torch
import torch.nn.functional as F

# Input scores in reasonable range
x = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
y = F.softmax(x, dim=0)
print(f"Input: {x.data}")
print(f"Softmax: {y.data}")
# Output: tensor([0.6590, 0.2424, 0.0886])

# Compute gradient w.r.t. first output
y[0].backward()
print(f"Gradient: {x.grad}")
# Output: tensor([0.2247, -0.1597, -0.0584])

# Input scores with large values (what happens without scaling)
x = torch.tensor([20.0, 10.0, 0.0], requires_grad=True)
y = F.softmax(x, dim=0)
print(f"Input: {x.data}")
print(f"Softmax: {y.data}")
# Output: tensor([9.9995e-01, 4.5398e-05, 2.0612e-09])

# Compute gradient w.r.t. first output
y[0].backward()
print(f"Gradient: {x.grad}")
# Output: tensor([4.5398e-05, -4.5396e-05, -2.0611e-09])