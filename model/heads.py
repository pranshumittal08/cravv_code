"""
Task-specific heads for segmentation, detection, and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Segmentation head that upsamples P2 features to 256x256.

    Output: 4-class segmentation mask (background, pan, stirrer, inlet_pipes)
    """

    def __init__(self, in_channels=256, num_classes=1, input_size=(64, 64), output_size=(256, 256)):
        """
        Args:
            in_channels: Input feature channels (from FPN, default: 256)
            num_classes: Number of segmentation classes (default: 1)
            input_size: Input feature map size (default: 64x64 for 256/4)
            output_size: Output mask size (default: 256x256)
        """
        super(SegmentationHead, self).__init__()

        # Transposed convolutions for upsampling
        # P2 is at 1/4 resolution, so we need to upsample 4x
        self.deconv1 = nn.ConvTranspose2d(
            in_channels, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Final classification layer
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, p2):
        """
        Forward pass through segmentation head.

        Args:
            p2: FPN feature at 1/4 resolution, shape (B, 256, H/4, W/4)

        Returns:
            seg_logits: Segmentation logits, shape (B, num_classes, 256, 256)
        """
        x = F.relu(self.bn1(self.deconv1(p2)))
        x = F.relu(self.bn2(self.deconv2(x)))
        seg_logits = self.classifier(x)

        return seg_logits


class ClassificationHead(nn.Module):
    """
    Classification head for cooking state prediction.

    Uses C5 features (before FPN) with Global Average Pooling + FC layers.
    Output: Class logits (classification)
    """

    def __init__(self, in_channels=2048, num_classes=10, hidden_dim=512):
        """
        Args:
            in_channels: Input feature channels from C5 (default: 2048)
            num_classes: Number of classification classes (default: 10)
            hidden_dim: Hidden layer dimension (default: 512)
        """
        super(ClassificationHead, self).__init__()

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)  # Class logits

    def forward(self, c5):
        """
        Forward pass through classification head.

        Args:
            c5: Backbone features from layer4, shape (B, 2048, H/32, W/32)

        Returns:
            class_logits: Class logits, shape (B, num_classes)
        """
        x = self.gap(c5)  # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 2048)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        class_logits = self.fc3(x)  # (B, num_classes)

        return class_logits


class DetectionHead(nn.Module):
    """
    Detection head for object detection (anchor-free FCOS-style).

    Uses multi-scale features P3, P4, P5 for detection.
    Output: Bounding boxes + class labels (pan, stirrer, inlet_pipes)
    """

    def __init__(self, in_channels=256, num_classes=3):
        """
        Args:
            in_channels: Input feature channels from FPN (default: 256)
            num_classes: Number of detection classes (default: 3)
        """
        super(DetectionHead, self).__init__()

        self.num_classes = num_classes

        # Shared feature extraction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Classification branch (object vs background + class)
        self.cls_head = nn.Conv2d(
            in_channels, num_classes + 1, 1)  # +1 for background

        # Regression branch (4 values: center_x, center_y, width, height)
        self.reg_head = nn.Conv2d(in_channels, 4, 1)

        # Center-ness branch (for FCOS-style)
        self.centerness_head = nn.Conv2d(in_channels, 1, 1)

    def forward(self, p3, p4, p5):
        """
        Forward pass through detection head.

        Args:
            p3: FPN feature at 1/8 resolution, shape (B, 256, H/8, W/8)
            p4: FPN feature at 1/16 resolution, shape (B, 256, H/16, W/16)
            p5: FPN feature at 1/32 resolution, shape (B, 256, H/32, W/32)

        Returns:
            det_outputs: Dictionary with:
                - 'cls_logits': List of classification logits for each scale
                - 'reg_preds': List of regression predictions for each scale
                - 'centerness': List of centerness predictions for each scale
        """
        outputs = {
            'cls_logits': [],
            'reg_preds': [],
            'centerness': []
        }

        for p in [p3, p4, p5]:
            features = self.shared_conv(p)

            cls_logits = self.cls_head(features)
            reg_preds = self.reg_head(features)
            centerness = self.centerness_head(features)

            outputs['cls_logits'].append(cls_logits)
            outputs['reg_preds'].append(reg_preds)
            outputs['centerness'].append(centerness)

        return outputs
