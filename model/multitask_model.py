"""
Multi-task model combining backbone, FPN, and three task heads.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from .backbone import ResNetBackbone
from .fpn import FPN
from .heads import SegmentationHead, DetectionHead, ClassificationHead


class MultiTaskModel(nn.Module):
    """
    Multi-task learning model with shared ResNet-50 backbone and FPN.

    Three task heads:
    1. Segmentation: Pan, stirrer, inlet pipes segmentation
    2. Detection: Object detection (pan, stirrer, inlet pipes)
    3. Classification: Cooking state classification
    """

    def __init__(
        self,
        num_seg_classes: int = 4,
        num_det_classes: int = 3,
        num_cls_classes: int = 10,
        pretrained_backbone: bool = True,
        fpn_channels: int = 256,
    ) -> None:
        """
        Args:
            num_seg_classes: Number of segmentation classes (default: 4)
            num_det_classes: Number of detection classes (default: 3)
            num_cls_classes: Number of classification classes (default: 10)
            pretrained_backbone: Whether to use pretrained ResNet-50 (default: True)
            fpn_channels: Number of channels in FPN/BiFPN (default: 256)
        """
        super(MultiTaskModel, self).__init__()

        # Backbone
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)

        # FPN
        self.fpn = FPN(in_channels_list=[
                       256, 512, 1024, 2048], out_channels=256)

        # Task heads
        self.seg_head = SegmentationHead(
            in_channels=fpn_channels, num_classes=num_seg_classes)
        self.det_head = DetectionHead(
            in_channels=fpn_channels, num_classes=num_det_classes)
        self.cls_head = ClassificationHead(
            in_channels=2048, num_classes=num_cls_classes)

    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-task model.

        Args:
            x: Input image tensor, shape (B, 3, 256, 256)
            task: Optional task identifier ('seg', 'det', 'cls', or None for all)

        Returns:
            Dictionary with requested task outputs

        Raises:
            ValueError: If task is not one of 'seg', 'det', 'cls', or None
        """
        if task is not None and task not in ('seg', 'det', 'cls'):
            raise ValueError(
                f"task must be 'seg', 'det', 'cls', or None, got '{task}'")
        # Extract backbone features
        c2, c3, c4, c5 = self.backbone(x)

        # Generate FPN features
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        outputs = {}

        # Compute task-specific outputs
        if task is None or task == 'seg':
            outputs['seg'] = self.seg_head(p2)

        if task is None or task == 'det':
            outputs['det'] = self.det_head(p3, p4, p5)

        if task is None or task == 'cls':
            outputs['cls'] = self.cls_head(c5)

        return outputs

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for phase 1 training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters for phase 2 training."""
        for param in self.backbone.parameters():
            param.requires_grad = True
