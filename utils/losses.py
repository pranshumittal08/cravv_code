"""
Loss functions for multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fcos_losses import assign_targets_to_locations, compute_focal_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining segmentation, detection, and classification losses.

    L_total = weight_seg * L_seg + weight_det * L_det + weight_cls * L_cls
    """

    def __init__(self, weight_seg=1.0, weight_det=1.0, weight_cls=0.5):
        """
        Args:
            weight_seg: Weight for segmentation loss (default: 1.0)
            weight_det: Weight for detection loss (default: 1.0)
            weight_cls: Weight for classification loss (default: 0.5)
        """
        super(MultiTaskLoss, self).__init__()
        self.weight_seg = weight_seg
        self.weight_det = weight_det
        self.weight_cls = weight_cls

        # Loss functions
        # Use BCE per class for segmentation (binary cross-entropy for each class)
        self.seg_loss_fn = nn.BCEWithLogitsLoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def compute_seg_loss(self, seg_logits, seg_mask, num_classes=4):
        """
        Compute segmentation loss using Binary Cross-Entropy.

        Args:
            seg_logits: Segmentation logits, shape (B, num_classes, H, W)
            seg_mask: Ground truth mask with class indices, shape (B, H, W)
            num_classes: Number of segmentation classes (default: 4)

        Returns:
            seg_loss: Segmentation loss
        """
        B, C, H, W = seg_logits.shape

        # Binary segmentation: single channel output
        # seg_logits: (B, 1, H, W), seg_mask: (B, H, W)
        seg_mask_float = seg_mask.float().unsqueeze(1)  # (B, 1, H, W)
        seg_loss = self.seg_loss_fn(seg_logits, seg_mask_float)

        return seg_loss

    def compute_det_loss(self, det_outputs, boxes=None, labels=None, targets=None, image_size=256):
        """
        Compute FCOS-style detection loss with proper ground truth matching.

        Args:
            det_outputs: Dictionary with 'cls_logits', 'reg_preds', 'centerness'
            boxes: Ground truth boxes (list of tensors, one per image) in normalized [cx, cy, w, h] format
            labels: Ground truth labels (list of tensors, one per image)
            targets: Dictionary with 'boxes' and 'labels' keys (optional)
            image_size: Input image size (default: 256)

        Returns:
            det_loss: Detection loss
        """
        # Support both old API (boxes, labels) and new API (targets dict)
        if targets is not None:
            boxes = targets.get('boxes', boxes)
            labels = targets.get('labels', labels)

        # FCOS uses Focal Loss for classification
        # Strides for each scale: P3=8, P4=16, P5=32
        strides = [8, 16, 32]

        total_cls_loss = 0
        total_reg_loss = 0
        total_centerness_loss = 0

        # Process each scale
        for scale_idx, (cls_logits, reg_preds, centerness_preds) in enumerate(zip(
            det_outputs['cls_logits'],
            det_outputs['reg_preds'],
            det_outputs['centerness']
        )):
            B, num_classes_plus_bg, H, W = cls_logits.shape
            num_classes = num_classes_plus_bg - 1  # Exclude background
            stride = strides[scale_idx]
            device = cls_logits.device

            # Reshape predictions: (B, C, H, W) -> (B, H, W, C)
            cls_logits = cls_logits.permute(
                0, 2, 3, 1)  # (B, H, W, num_classes+1)
            reg_preds = reg_preds.permute(0, 2, 3, 1)  # (B, H, W, 4)
            # centerness_preds shape: (B, 1, H, W) -> (B, H, W)
            if centerness_preds.dim() == 4:
                centerness_preds = centerness_preds.squeeze(1)  # (B, H, W)

            scale_cls_loss = 0
            scale_reg_loss = 0
            scale_centerness_loss = 0

            # Process each image in batch
            for b in range(B):
                # Get GT boxes and labels for this image
                if isinstance(boxes, list):
                    gt_boxes = boxes[b]  # (N, 4)
                    gt_labels = labels[b]  # (N,)
                else:
                    gt_boxes = boxes[b] if boxes.dim() > 2 else boxes
                    gt_labels = labels[b] if labels.dim() > 1 else labels

                # Move to device if needed
                if not isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = torch.tensor(
                        gt_boxes, dtype=torch.float32, device=device)
                if not isinstance(gt_labels, torch.Tensor):
                    gt_labels = torch.tensor(
                        gt_labels, dtype=torch.long, device=device)

                # Assign targets to locations
                cls_targets, reg_targets, centerness_targets, positive_mask = \
                    assign_targets_to_locations(
                        (H, W), gt_boxes, gt_labels, stride, image_size
                    )

                # Move targets to same device as predictions
                cls_targets = cls_targets.to(device)
                reg_targets = reg_targets.to(device)
                centerness_targets = centerness_targets.to(device)
                positive_mask = positive_mask.to(device)

                # Classification loss: Use Focal Loss helper function
                # Use all object classes (exclude background class 0)
                cls_logits_obj = cls_logits[b, :, :, 1:]  # (H, W, num_classes)
                # (H*W, num_classes)
                cls_logits_flat = cls_logits_obj.reshape(-1, num_classes)
                cls_targets_flat = cls_targets.reshape(-1)  # (H*W,)

                # Use the focal loss helper function
                cls_loss = compute_focal_loss(
                    cls_logits_flat, cls_targets_flat, alpha=0.25, gamma=2.0, reduction='mean'
                )
                scale_cls_loss += cls_loss

                # Flatten positive_mask for indexing flattened tensors
                positive_mask_flat = positive_mask.reshape(-1)  # (H*W,)

                # Regression loss: Smooth L1 only on positive samples
                if positive_mask_flat.sum() > 0:
                    # Get positive predictions and targets
                    reg_preds_flat = reg_preds[b].reshape(-1, 4)  # (H*W, 4)
                    reg_targets_flat = reg_targets.reshape(-1, 4)  # (H*W, 4)

                    # (N_pos, 4)
                    pos_reg_preds = reg_preds_flat[positive_mask_flat]
                    # (N_pos, 4)
                    pos_reg_targets = reg_targets_flat[positive_mask_flat]

                    # Use Smooth L1 loss on (l, t, r, b) distances directly (standard FCOS)
                    reg_loss = F.smooth_l1_loss(
                        pos_reg_preds, pos_reg_targets, reduction='mean')
                    scale_reg_loss += reg_loss

                # Centerness loss: BCE only on positive samples
                if positive_mask_flat.sum() > 0:
                    # (H*W,)
                    centerness_preds_flat = centerness_preds[b].reshape(-1)
                    # (H*W,)
                    centerness_targets_flat = centerness_targets.reshape(-1)

                    pos_centerness_preds = centerness_preds_flat[positive_mask_flat]
                    pos_centerness_targets = centerness_targets_flat[positive_mask_flat]

                    centerness_loss = F.binary_cross_entropy_with_logits(
                        pos_centerness_preds, pos_centerness_targets, reduction='mean')
                    scale_centerness_loss += centerness_loss

            # Average over batch
            total_cls_loss += scale_cls_loss / B
            total_reg_loss += scale_reg_loss / B
            total_centerness_loss += scale_centerness_loss / B

        # Average over scales and combine losses (FCOS style)
        num_scales = len(det_outputs['cls_logits'])
        det_loss = (total_cls_loss / num_scales +
                    total_reg_loss / num_scales +
                    total_centerness_loss / num_scales)

        return det_loss

    def compute_cls_loss(self, cls_pred, cls_target, use_focal=True, gamma=2.0, alpha=0.25):
        """
        Compute classification loss using Focal Loss (sigmoid-based, multi-label style).
        Good for imbalanced classes.

        Args:
            cls_pred: Predicted class logits, shape (B, num_classes)
            cls_target: Ground truth class indices, shape (B,)
            use_focal: Whether to use focal loss (default: True)
            gamma: Focal loss gamma parameter (default: 2.0)
            alpha: Balancing factor (default: 0.25)

        Returns:
            cls_loss: Classification loss
        """
        if use_focal:
            return compute_focal_loss(
                cls_pred, cls_target, alpha=alpha, gamma=gamma
            )
        else:
            return self.cls_loss_fn(cls_pred, cls_target)

    def forward(self, outputs, targets):
        """
        Compute total multi-task loss.

        Args:
            outputs: Dictionary with 'seg', 'det', 'cls' keys
            targets: Dictionary with ground truth data

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        loss_dict = {}
        total_loss = 0

        # Segmentation loss
        if 'seg' in outputs and 'seg' in targets:
            seg_loss = self.compute_seg_loss(outputs['seg'], targets['seg'])
            loss_dict['seg'] = seg_loss
            total_loss += self.weight_seg * seg_loss

        # Detection loss
        if 'det' in outputs and 'det' in targets:
            det_loss = self.compute_det_loss(
                outputs['det'],
                targets['det']['boxes'],
                targets['det']['labels']
            )
            loss_dict['det'] = det_loss
            total_loss += self.weight_det * det_loss

        # Classification loss
        if 'cls' in outputs and 'cls' in targets:
            cls_loss = self.compute_cls_loss(outputs['cls'], targets['cls'])
            loss_dict['cls'] = cls_loss
            total_loss += self.weight_cls * cls_loss

        loss_dict['total'] = total_loss
        return total_loss, loss_dict
