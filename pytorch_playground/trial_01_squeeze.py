import torch

# Example 1: Remove all dimensions of size 1
x = torch.zeros(2, 1, 3, 1, 4)
print(x.shape)  # torch.Size([2, 1, 3, 1, 4])
y = torch.squeeze(x)
print(y.shape)  # torch.Size([2, 3, 4])

# Example 2: Remove specific dimension
x = torch.zeros(2, 1, 3, 1, 4)
y = torch.squeeze(x, dim=1)  # Remove dimension 1 (size 1)
print(y.shape)  # torch.Size([2, 3, 1, 4])

# Example 3: Using tensor method
x = torch.zeros(2, 1, 3)
y = x.squeeze()  # Same as torch.squeeze(x)
print(y.shape)  # torch.Size([2, 3])

# Example 4: Won't squeeze if dimension size > 1
x = torch.zeros(2, 3, 4)
y = torch.squeeze(x)
print(y.shape)  # torch.Size([2, 3, 4]) - no change