import torch
import torch.nn as nn


input_image = torch.randn(1, 1, 7, 7)  # [1, 1, 5, 5]
print("Input shape: ", input_image.shape)

conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0)
output1 = conv1(input_image)  # [1, 1, 5, 5] → [1, 1, 3, 3]
print("Output shape after Conv1: ", output1.shape)

conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0)  
output2 = conv2(output1)      # [1, 1, 3, 3] → [1, 1, 1, 1]
print("Output shape after Conv2: ", output2.shape)

conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0)
output3 = conv3(output2)      # This would FAIL!
