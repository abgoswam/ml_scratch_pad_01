import torch
import torch.nn as nn


input_image = torch.randn(1, 1, 3, 3)  # Format: [B, C, H, W]
print(input_image.shape)

conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0)

output = conv1(input_image)

print("Input shape: ", input_image.shape)
print("Output shape after Conv1: ", output.shape)

print(conv1)