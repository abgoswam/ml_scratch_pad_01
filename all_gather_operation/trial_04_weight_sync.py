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
    num_params = len(list(models[0].parameters()))
    print(f"num_params:{num_params}")

    for i in range(num_params):
        state = []
        for j in range(len(models)):
            ith_p = list(models[j].parameters())[i]
            print(f"[{i}:{j}] ith_p.grad:\n{ith_p.grad}")
            state.append(ith_p.grad)

        print(f"\n****state***:\n{state}")
        _sum = state[0]
        for k in range(1, len(models)):
            _sum += state[k]
        _avg = _sum / len(models)
        print(f"\n****avg***:\n{_avg}")

        # Apply averaged gradients to all models
        for j in range(len(models)):
            ith_p = list(models[j].parameters())[i]
            ith_p.grad = _avg.clone()  # Use clone() to avoid reference issues

def sync_model_weights(models):
    """
    Synchronize weights from models[0] to all other models.
    This simulates broadcasting updated weights in distributed training.
    """
    print("=== SYNCHRONIZING WEIGHTS ===")
    with torch.no_grad():
        for i in range(1, len(models)):
            for p0, pi in zip(models[0].parameters(), models[i].parameters()):
                pi.data.copy_(p0.data)
    print("All models now have identical weights")

# Create models with same initial weights (realistic distributed training)
torch.manual_seed(42)
model0 = make_model()

torch.manual_seed(42)  # Reset seed to get identical weights
model1 = make_model()

models = [model0, model1]

# Verify same initial weights
print("=== VERIFYING INITIAL WEIGHTS ===")
for i, (p0, p1) in enumerate(zip(models[0].parameters(), models[1].parameters())):
    print(f"Parameter {i}:")
    print(f"  Are equal: {torch.equal(p0.data, p1.data)}")
print()

opt = optim.Adam(models[0].parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# Set seed for reproducible data batches (optional, for debugging)
torch.manual_seed(123)
x1, y1 = torch.randn(5,4), torch.randn(5,2)
x2, y2 = torch.randn(5,4), torch.randn(5,2) 

print("=== FORWARD PASS AND GRADIENT COMPUTATION ===")
for i, (m, (x, y)) in enumerate(zip(models, [(x1, y1), (x2, y2)])):
    print(f"Processing batch {i+1} on model {i}")
    out = m(x)
    loss = loss_fn(out, y)
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()

print("\n=== GRADIENT AVERAGING ===")
average_gradients(models)

print("\n=== VERIFYING AVERAGED GRADIENTS ===")
print("Model 0 gradients:")
for i, p in enumerate(models[0].parameters()):
    print(f"  Param {i}: {p.grad}")
    
print("\nModel 1 gradients:")
for i, p in enumerate(models[1].parameters()):
    print(f"  Param {i}: {p.grad}")

print("\nGradients are identical:", 
      all(torch.equal(p0.grad, p1.grad) 
          for p0, p1 in zip(models[0].parameters(), models[1].parameters())))

print("\n=== OPTIMIZER STEP ===")
print("Updating only models[0] parameters...")
opt.step()

print("\n=== AFTER OPTIMIZER STEP ===")
print("Weights are still identical:", 
      all(torch.equal(p0.data, p1.data) 
          for p0, p1 in zip(models[0].parameters(), models[1].parameters())))

# This is what would happen in real distributed training
sync_model_weights(models)

print("\nAfter weight sync - weights are identical:", 
      all(torch.equal(p0.data, p1.data) 
          for p0, p1 in zip(models[0].parameters(), models[1].parameters())))