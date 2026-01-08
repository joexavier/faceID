"""
MobileFaceNet Architecture

PyTorch implementation of MobileFaceNet for face recognition.
This architecture matches the weights from: https://github.com/foamliu/MobileFaceNet

Architecture:
- Input: 112x112x3
- Backbone: MobileNetV2-style with inverted residuals
- Output: 128-D embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Initial depthwise separable convolution block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.prelu(x, torch.tensor(0.25, device=x.device))
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.prelu(x, torch.tensor(0.25, device=x.device))
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV2-style)"""

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expand_ratio)

        self.conv = nn.Sequential(
            # Expansion (pointwise)
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
            ),
            # Depthwise
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            ),
            # Projection (pointwise)
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class GlobalDepthwiseConv(nn.Module):
    """Global depthwise convolution (7x7)"""

    def __init__(self, in_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, stride=1, padding=0,
            groups=in_channels, bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet architecture for face recognition.

    Input: 112x112x3 RGB image
    Output: 128-D embedding
    """

    def __init__(self, embedding_size=128):
        super().__init__()

        # Initial convolution: 112x112x3 -> 56x56x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        # Depthwise separable: 56x56x64 -> 56x56x64
        self.dw_conv = DepthwiseSeparableConv(64, 64)

        # Inverted residual blocks (features)
        # Configuration: (expand_ratio, out_channels, num_blocks, stride)
        settings = [
            # Stage 1: 56x56x64 -> 28x28x64
            (2, 64, 5, 2),
            # Stage 2: 28x28x64 -> 14x14x128
            (4, 128, 1, 2),
            # Stage 3: 14x14x128 -> 14x14x128
            (2, 128, 6, 1),
            # Stage 4: 14x14x128 -> 7x7x128
            (4, 128, 1, 2),
            # Stage 5: 7x7x128 -> 7x7x128
            (2, 128, 2, 1),
        ]

        features = []
        in_channels = 64

        for expand_ratio, out_channels, num_blocks, stride in settings:
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                features.append(InvertedResidual(in_channels, out_channels, s, expand_ratio))
                in_channels = out_channels

        self.features = nn.Sequential(*features)

        # Expansion: 7x7x128 -> 7x7x512
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
        )

        # Global depthwise: 7x7x512 -> 1x1x512
        self.gdconv = GlobalDepthwiseConv(512)

        # Linear: 512 -> 128
        self.conv3 = nn.Conv2d(512, embedding_size, kernel_size=1, stride=1, padding=0, bias=True)

        # Final batch normalization
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = F.prelu(x, torch.tensor(0.25, device=x.device))

        # Depthwise separable
        x = self.dw_conv(x)

        # Inverted residual blocks
        x = self.features(x)

        # Expansion
        x = self.conv2(x)
        x = F.prelu(x, torch.tensor(0.25, device=x.device))

        # Global depthwise
        x = self.gdconv(x)

        # Linear
        x = self.conv3(x)

        # Flatten and batch norm
        x = x.view(x.size(0), -1)
        x = self.bn(x)

        return x
