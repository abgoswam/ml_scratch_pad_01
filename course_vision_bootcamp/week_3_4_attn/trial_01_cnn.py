import torch
import torch.nn as nn

# 1. Simulate a dummy RGB image of size (batch_size=1, channels=3, height=224, width=224)
input_image = torch.randn(1, 3, 224, 224)  # Format: [B, C, H, W]
print(input_image.shape)

# 2. Define Conv1: 3 input channels (RGB), 64 output channels, 3x3 kernel
conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

# 3. Pass the input image through Conv1
output = conv1(input_image)

# 4. Print shapes
print("Input shape: ", input_image.shape)
print("Output shape after Conv1: ", output.shape)

print(conv1)