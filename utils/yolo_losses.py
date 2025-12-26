"""
YOLOv8-style loss functions for detection.
Implements CIoU loss and Distribution Focal Loss (DFL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_iou(box1, box2, xywh=True, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of boxes.

    Args:
        box1: Predicted boxes, shape (N, 4) in format [x, y, w, h] or [x1, y1, x2, y2]
        box2: Ground truth boxes, shape (N, 4) in same format
        xywh: If True, boxes are in [x, y, w, h] format (center), else [x1, y1, x2, y2]
        CIoU: If True, compute Complete IoU
        eps: Small value to avoid division by zero

    Returns:
        iou: IoU values, shape (N,)
    """
    # Convert to [x1, y1, x2, y2] format
    if xywh:
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if CIoU:
        # Complete IoU calculation
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) -
                                           torch.atan(w1 / (h1 + eps)), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def distribution_focal_loss(pred, target, gamma=1.5):
    """
    Distribution Focal Loss (DFL) as used in YOLOv8.

    Args:
        pred: Predicted distribution, shape (N, 4) or (N,)
        target: Target values, shape (N,)
        gamma: Focusing parameter (default: 1.5)

    Returns:
        loss: DFL loss value
    """
    # Simplified DFL - in full implementation, this would use a distribution
    # For now, we'll use a focal loss variant
    pred = pred.sigmoid()
    target = target.unsqueeze(-1) if target.dim() == 1 else target

    # Binary cross entropy with focal weighting
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    loss = (focal_weight * bce).mean()

    return loss


def compute_ciou_loss(pred_boxes, target_boxes):
    """
    Compute Complete IoU (CIoU) loss for bounding box regression.

    Args:
        pred_boxes: Predicted boxes, shape (N, 4) in [x, y, w, h] format
        target_boxes: Ground truth boxes, shape (N, 4) in [x, y, w, h] format

    Returns:
        ciou_loss: CIoU loss (1 - CIoU)
    """
    ciou = bbox_iou(pred_boxes, target_boxes, xywh=True, CIoU=True)
    return (1.0 - ciou).mean()


def compute_dfl_loss(pred_dist, target_values):
    """
    Compute Distribution Focal Loss for box regression.

    Args:
        pred_dist: Predicted distribution values, shape (N, 4)
        target_values: Target box values, shape (N, 4)

    Returns:
        dfl_loss: DFL loss
    """
    # DFL is applied to each coordinate (x, y, w, h)
    loss = 0.0
    for i in range(4):
        loss += distribution_focal_loss(pred_dist[:, i], target_values[:, i])
    return loss / 4.0
