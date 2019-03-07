import torch.nn as nn
import torch.nn.functional as F


def initialize(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight.data, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.Linear):
        nn.init.constant_(module.bias.data, 0)


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        self.apply(initialize)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)


class DarknetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBN(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(
            mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_shortcut = shortcut

        self.apply(initialize)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.use_shortcut:
            y += x
        return y


class DarknetStage(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.add_module(
            'conv1',
            ConvBN(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        for index in range(num_blocks):
            self.add_module(
                f'resblock{index}',
                DarknetBottleneck(out_channels, out_channels, shortcut=True))
