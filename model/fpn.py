"""
Feature Pyramid Network (FPN) implementation.
Creates a top-down pathway with lateral connections to generate
multi-scale features with uniform channel dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network.

    Takes backbone features [C2, C3, C4, C5] and generates
    pyramid features [P2, P3, P4, P5] with uniform 256 channels.
    """

    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        """
        Args:
            in_channels_list: List of input channel dimensions [C2, C3, C4, C5]
            out_channels: Output channel dimension (default: 256)
        """
        super(FPN, self).__init__()

        self.out_channels = out_channels

        # Lateral connections (1x1 convs to reduce channels)
        self.lateral_conv_c2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_conv_c3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_conv_c4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lateral_conv_c5 = nn.Conv2d(in_channels_list[3], out_channels, 1)

        # Top-down pathway (3x3 convs for feature refinement)
        self.fpn_conv_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_conv_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_conv_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_conv_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize FPN weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, c2, c3, c4, c5):
        """
        Forward pass through FPN.

        Args:
            c2: Features from backbone layer1, shape (B, 256, H/4, W/4)
            c3: Features from backbone layer2, shape (B, 512, H/8, W/8)
            c4: Features from backbone layer3, shape (B, 1024, H/16, W/16)
            c5: Features from backbone layer4, shape (B, 2048, H/32, W/32)

        Returns:
            p2: Pyramid feature at 1/4 resolution, shape (B, 256, H/4, W/4)
            p3: Pyramid feature at 1/8 resolution, shape (B, 256, H/8, W/8)
            p4: Pyramid feature at 1/16 resolution, shape (B, 256, H/16, W/16)
            p5: Pyramid feature at 1/32 resolution, shape (B, 256, H/32, W/32)
        """
        # Lateral connections
        l5 = self.lateral_conv_c5(c5)
        l4 = self.lateral_conv_c4(c4)
        l3 = self.lateral_conv_c3(c3)
        l2 = self.lateral_conv_c2(c2)

        # Top-down pathway
        p5 = self.fpn_conv_p5(l5)

        # Upsample and add
        p4 = l4 + F.interpolate(p5, size=l4.shape[-2:], mode='nearest')
        p4 = self.fpn_conv_p4(p4)

        p3 = l3 + F.interpolate(p4, size=l3.shape[-2:], mode='nearest')
        p3 = self.fpn_conv_p3(p3)

        p2 = l2 + F.interpolate(p3, size=l2.shape[-2:], mode='nearest')
        p2 = self.fpn_conv_p2(p2)

        return p2, p3, p4, p5
