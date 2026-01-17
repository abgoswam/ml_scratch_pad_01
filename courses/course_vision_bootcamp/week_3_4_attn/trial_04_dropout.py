import torch
import torch.nn as nn
import time


# Define a simple network with dropout
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Create the model
torch.manual_seed(0)
model = SimpleNet()

# Sample input
x = torch.ones((1, 5))

# Forward passes in training mode
torch.manual_seed(int(time.time()))
model.train()
train_outputs = [model(x).item() for _ in range(5)]

# Forward passes in eval mode
model.eval()
eval_outputs = [model(x).item() for _ in range(5)]

print(train_outputs, eval_outputs)
