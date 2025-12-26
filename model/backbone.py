"""
ResNet-50 Backbone for feature extraction.
Extracts intermediate features at different scales: C2, C3, C4, C5.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    ResNet-50 backbone that extracts features at multiple scales.

    Returns:
        C2: 256 channels, 1/4 resolution
        C3: 512 channels, 1/8 resolution
        C4: 1024 channels, 1/16 resolution
        C5: 2048 channels, 1/32 resolution
    """

    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)

        # Extract layers for feature extraction
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet blocks
        self.layer1 = resnet.layer1  # C2: 256 channels
        self.layer2 = resnet.layer2  # C3: 512 channels
        self.layer3 = resnet.layer3  # C4: 1024 channels
        self.layer4 = resnet.layer4  # C5: 2048 channels

    def forward(self, x):
        """
        Forward pass through ResNet-50.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            c2: Features from layer1, shape (B, 256, H/4, W/4)
            c3: Features from layer2, shape (B, 512, H/8, W/8)
            c4: Features from layer3, shape (B, 1024, H/16, W/16)
            c5: Features from layer4, shape (B, 2048, H/32, W/32)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)  # 1/4 resolution
        c3 = self.layer2(c2)  # 1/8 resolution
        c4 = self.layer3(c3)  # 1/16 resolution
        c5 = self.layer4(c4)  # 1/32 resolution

        return c2, c3, c4, c5
