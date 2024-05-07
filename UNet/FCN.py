
from Conv2 import Conv2
import torch.nn as nn
import torch

    
class FCN(nn.Module):
    def __init__(self):
        # Super constructor
        super().__init__()
        self.conv = nn.Sequential(
            Conv2(in_channels=3, out_channels=16),
            Conv2(in_channels=16, out_channels=32),
            Conv2(in_channels=32, out_channels=64),
            Conv2(in_channels=64, out_channels=128),
            Conv2(in_channels=128, out_channels=256),            
            Conv2(in_channels=256, out_channels=128),
            Conv2(in_channels=128, out_channels=64),
            Conv2(in_channels=64, out_channels=32),
            Conv2(in_channels=32, out_channels=16),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
        )
        

    def forward(self, images):
        return torch.sigmoid(self.conv(images))