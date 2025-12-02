import torch
import torch.nn as nn
from ..layers.conv_layer import ConvLayer
from ..layers.se_block import SEBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16, downsample=None):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.se = SEBlock(out_channels, reduction)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
