import torch.nn as nn 
from Conv2 import Conv2 
import torch

class UNet_add(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2(in_channels=3, out_channels=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2(in_channels=16, out_channels=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Conv2(in_channels=32, out_channels=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = Conv2(in_channels=64, out_channels=128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = Conv2(in_channels=128, out_channels=256)

        # Decoder 
        self.transpose_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.conv6 = Conv2(in_channels=128, out_channels=128)
        self.transpose_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.conv7 = Conv2(in_channels=64, out_channels=64)
        self.transpose_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

        self.conv8 = Conv2(in_channels=32, out_channels=32)
        self.transpose_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)

        self.conv9 = Conv2(in_channels=16, out_channels=16)

        self.conv10 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        

    def forward(self, images):
        # Encoder

        # print(f'images.shape = {images.shape}')
        c1 = self.conv1(images)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        # Decoder 
        u6 = self.transpose_conv1(c5)
        # u6 = torch.cat((u6, c4), dim=1) # first dim is batch dimension, second (1) is the channel
        u6 += c4

        c6 = self.conv6(u6)
        u7 = self.transpose_conv2(c6)
        u7 += c3
        # u7 = torch.cat((u7, c3), dim=1)

        c7 = self.conv7(u7)
        u8 = self.transpose_conv3(c7)
        u8 += c2 
        # u8 = torch.cat((u8, c2), dim=1)

        c8 = self.conv8(u8)
        u9 = self.transpose_conv4(c8)
        u9 += c1 
        # u9 = torch.cat((u9, c1), dim=1)

        c9 = self.conv9(u9)
        
        return torch.sigmoid(self.conv10(c9))