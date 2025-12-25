# %%
import torch
import torch.nn as nn

# %%
class ManualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# %%
seq_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# %%
manual_model = ManualModel()

# %%
x = torch.randn(3, 10)
manual_out = manual_model(x)
seq_out = seq_model(x)

print(f"Manual output shape: {manual_out.shape}")
print(f"Sequential output shape: {seq_out.shape}")

# %%
x = torch.randn(2, 10)

output1 = seq_model(x)  # This calls __call__ which calls forward
output2 = seq_model.forward(x)  # Direct call to forward

print("\nAre they the same?", torch.allclose(output1, output2))



