"""
FCOS-style loss computation utilities.
Handles proper matching of predictions to ground truth boxes.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_centerness_target(
    l: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
    Compute centerness target for FCOS.

    Args:
        l, t, r, b: Distances to left, top, right, bottom boundaries

    Returns:
        centerness: Centerness value in [0, 1]
    """
    # Centerness = sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    left_right = torch.min(l, r) / (torch.max(l, r) + eps)
    top_bottom = torch.min(t, b) / (torch.max(t, b) + eps)
    centerness = torch.sqrt(left_right * top_bottom + eps)
    return centerness


def assign_targets_to_locations(
    feature_map_size: Tuple[int, int],
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    stride: int,
    image_size: int = 256,
    negative_threshold: float = float('inf')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign ground truth boxes to feature map locations (FCOS-style).

    Args:
        feature_map_size: (H, W) of feature map
        gt_boxes: Ground truth boxes in normalized format [center_x, center_y, width, height], shape (N, 4)
        gt_labels: Ground truth labels, shape (N,)
        stride: Feature map stride (8, 16, or 32)
        image_size: Input image size (default: 256)
        negative_threshold: Maximum distance to be negative (default: inf)

    Returns:
        cls_targets: Classification targets, shape (H, W) - class index or -1 for negative
        reg_targets: Regression targets, shape (H, W, 4) - (l, t, r, b) distances
        centerness_targets: Centerness targets, shape (H, W)
        positive_mask: Boolean mask for positive locations, shape (H, W)
    """
    H, W = feature_map_size
    device = gt_boxes.device if isinstance(gt_boxes, torch.Tensor) else 'cpu'

    # Initialize targets
    cls_targets = torch.full((H, W), -1, dtype=torch.long, device=device)
    reg_targets = torch.zeros(H, W, 4, dtype=torch.float32, device=device)
    centerness_targets = torch.zeros(H, W, dtype=torch.float32, device=device)
    positive_mask = torch.zeros(H, W, dtype=torch.bool, device=device)

    if len(gt_boxes) == 0:
        return cls_targets, reg_targets, centerness_targets, positive_mask

    # Convert normalized boxes to pixel coordinates
    gt_boxes_pixel = gt_boxes.clone()
    gt_boxes_pixel[:, 0] *= image_size  # center_x
    gt_boxes_pixel[:, 1] *= image_size  # center_y
    gt_boxes_pixel[:, 2] *= image_size  # width
    gt_boxes_pixel[:, 3] *= image_size  # height

    # Convert to (x1, y1, x2, y2) format
    x1 = gt_boxes_pixel[:, 0] - gt_boxes_pixel[:, 2] / 2
    y1 = gt_boxes_pixel[:, 1] - gt_boxes_pixel[:, 3] / 2
    x2 = gt_boxes_pixel[:, 0] + gt_boxes_pixel[:, 2] / 2
    y2 = gt_boxes_pixel[:, 1] + gt_boxes_pixel[:, 3] / 2

    # Convert to numpy or Python lists for easier scalar operations
    # This avoids tensor/scalar mixing issues
    if isinstance(x1, torch.Tensor):
        x1 = x1.cpu().numpy()
        y1 = y1.cpu().numpy()
        x2 = x2.cpu().numpy()
        y2 = y2.cpu().numpy()

    # For each location in feature map
    for y in range(H):
        for x in range(W):
            # Location in image coordinates
            loc_x = (x + 0.5) * stride
            loc_y = (y + 0.5) * stride

            # Check which boxes this location falls into
            best_iou = 0.0
            best_box_idx = -1
            best_l, best_t, best_r, best_b = 0, 0, 0, 0

            for box_idx in range(len(gt_boxes)):
                # Check if location is inside box
                if (loc_x >= x1[box_idx] and loc_x <= x2[box_idx] and
                        loc_y >= y1[box_idx] and loc_y <= y2[box_idx]):

                    # Compute distances to boundaries (in pixel coordinates)
                    l = loc_x - x1[box_idx]
                    t = loc_y - y1[box_idx]
                    r = x2[box_idx] - loc_x
                    b = y2[box_idx] - loc_y

                    # Normalize by stride (convert to feature map space)
                    l = l / stride
                    t = t / stride
                    r = r / stride
                    b = b / stride

                    # Check if within positive range (max distance threshold)
                    # In FCOS, locations too far from center are filtered out
                    max_dist = max(l, t, r, b)  # Python max for scalars
                    if max_dist <= negative_threshold:
                        # For overlapping boxes, assign to the box with smallest area
                        # (standard FCOS behavior for nested/overlapping boxes)
                        box_area = (x2[box_idx] - x1[box_idx]) * \
                            (y2[box_idx] - y1[box_idx])

                        # Use area as tie-breaker (smaller area = better match)
                        # Invert area so smaller boxes have higher score
                        score = 1.0 / (box_area + 1e-6)

                        if score > best_iou:
                            best_iou = score
                            best_box_idx = box_idx
                            best_l, best_t, best_r, best_b = l, t, r, b

            # Assign target if found
            if best_box_idx >= 0:
                cls_targets[y, x] = gt_labels[best_box_idx]
                reg_targets[y, x] = torch.tensor(
                    [best_l, best_t, best_r, best_b], device=device)

                # Compute centerness (convert scalars to tensors)
                # best_l, best_t, best_r, best_b are Python floats
                l_tensor = torch.tensor(
                    float(best_l), device=device, dtype=torch.float32)
                t_tensor = torch.tensor(
                    float(best_t), device=device, dtype=torch.float32)
                r_tensor = torch.tensor(
                    float(best_r), device=device, dtype=torch.float32)
                b_tensor = torch.tensor(
                    float(best_b), device=device, dtype=torch.float32)

                centerness_targets[y, x] = compute_centerness_target(
                    l_tensor, t_tensor, r_tensor, b_tensor)
                positive_mask[y, x] = True

    return cls_targets, reg_targets, centerness_targets, positive_mask


def compute_focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 0.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Focal Loss for classification (FCOS-style).
    Computes loss on all samples (positive and negative).

    Args:
        pred_logits: Predicted logits, shape (N, num_classes)
        target: Target class indices, shape (N,) with -1 for negative samples
        alpha: Balancing factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean' or 'sum'

    Returns:
        loss: Focal loss averaged over all samples
    """
    # Create one-hot targets
    num_classes = pred_logits.shape[-1]
    device = pred_logits.device

    # Create one-hot encoding (negatives are all zeros)
    target_onehot = torch.zeros(
        pred_logits.shape[0], num_classes, device=device)
    valid_mask = target >= 0
    if valid_mask.sum() > 0:
        target_onehot[valid_mask, target[valid_mask]] = 1.0
    # Negative samples (target == -1) remain all zeros

    # Compute probabilities
    pred_probs = torch.sigmoid(pred_logits)

    # Compute BCE
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target_onehot, reduction='none')

    # Compute p_t
    p_t = pred_probs * target_onehot + (1 - pred_probs) * (1 - target_onehot)

    # Compute focal weight
    focal_weight = alpha * (1 - p_t) ** gamma

    # Apply focal weight
    focal_loss = focal_weight * bce

    # Sum over classes and average over all samples (including negatives)
    focal_loss = focal_loss.sum(dim=1)  # (N,)

    if reduction == 'mean':
        return focal_loss.mean()
    else:  # 'sum'
        return focal_loss.sum()
