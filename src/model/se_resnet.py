import torch
import torch.nn as nn
from ..layers.conv_layer import ConvLayer
from ..layers.flatten_layer import FlattenLayer
from ..layers.fc_layer import FCLayer
from ..layers.pool_layers.avgpool_layer import AvgPoolLayer
from ..blocks.residual_block import ResidualBlock

class SEResNet(nn.Module):
    def __init__(self, layers, num_classes=1000, reduction=16):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            ConvLayer(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers
        self.layer1 = self._make_layer(64, layers[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(256, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(512, layers[3], stride=2, reduction=reduction)

        self.avgpool = AvgPoolLayer((1,1))
        self.flatten = FlattenLayer()
        self.fc = FCLayer(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride, reduction):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, reduction, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, reduction=reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
