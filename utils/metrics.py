"""
Evaluation metrics for each task with per-class breakdowns.
"""

import torch
import numpy as np
from collections import defaultdict


def compute_seg_metrics(pred_mask, gt_mask, num_classes=4, class_names=None):
    """
    Compute segmentation metrics (mIoU) with per-class breakdown.

    Args:
        pred_mask: Predicted mask, shape (B, H, W) or (B, num_classes, H, W)
        gt_mask: Ground truth mask, shape (B, H, W)
        num_classes: Number of classes
        class_names: Optional list of class names

    Returns:
        dict with 'miou' and 'per_class_iou'
    """
    # Handle binary segmentation (single channel output)
    if pred_mask.dim() == 4:
        if pred_mask.shape[1] == 1:
            # Binary segmentation: threshold sigmoid output
            pred_mask = (torch.sigmoid(pred_mask) > 0.5).squeeze(1).long()
        else:
            # Multi-class: use argmax
            pred_mask = torch.argmax(pred_mask, dim=1)

    pred_mask = pred_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()

    # For binary segmentation, we have 2 classes (background=0, foreground=1)
    actual_classes = 2 if num_classes == 1 else num_classes

    ious = []
    per_class_iou = {}

    for cls in range(actual_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0

        ious.append(iou)

        if class_names and cls < len(class_names):
            per_class_iou[class_names[cls]] = iou
        else:
            cls_name = 'background' if cls == 0 else f'foreground' if actual_classes == 2 else f'class_{cls}'
            per_class_iou[cls_name] = iou

    miou = np.mean(ious)

    return {
        'miou': miou,
        'per_class_iou': per_class_iou
    }


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in [cx, cy, w, h] format.
    """
    # Convert to [x1, y1, x2, y2]
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area > 0:
        return inter_area / union_area
    return 0.0


def compute_det_metrics(pred_boxes, pred_labels, pred_scores,
                        gt_boxes, gt_labels,
                        num_classes=3, iou_threshold=0.5, class_names=None):
    """
    Compute detection metrics (mAP@0.5) with per-class breakdown.

    Args:
        pred_boxes: List of predicted boxes per image [(N, 4), ...]
        pred_labels: List of predicted labels per image [(N,), ...]
        pred_scores: List of prediction scores per image [(N,), ...]
        gt_boxes: List of ground truth boxes per image [(M, 4), ...]
        gt_labels: List of ground truth labels per image [(M,), ...]
        num_classes: Number of detection classes
        iou_threshold: IoU threshold for matching (default: 0.5)
        class_names: Optional list of class names

    Returns:
        dict with 'map50' and 'per_class_ap'
    """
    # Per-class tracking
    class_tp = defaultdict(list)  # True positives per class
    class_fp = defaultdict(list)  # False positives per class
    class_scores = defaultdict(list)  # Scores per class
    class_n_gt = defaultdict(int)  # Number of ground truth per class

    # Process each image
    for img_idx in range(len(pred_boxes)):
        p_boxes = pred_boxes[img_idx]
        p_labels = pred_labels[img_idx]
        p_scores = pred_scores[img_idx]
        g_boxes = gt_boxes[img_idx]
        g_labels = gt_labels[img_idx]

        # Convert to numpy if needed
        if isinstance(p_boxes, torch.Tensor):
            p_boxes = p_boxes.cpu().numpy()
        if isinstance(p_labels, torch.Tensor):
            p_labels = p_labels.cpu().numpy()
        if isinstance(p_scores, torch.Tensor):
            p_scores = p_scores.cpu().numpy()
        if isinstance(g_boxes, torch.Tensor):
            g_boxes = g_boxes.cpu().numpy()
        if isinstance(g_labels, torch.Tensor):
            g_labels = g_labels.cpu().numpy()

        # Count ground truth per class
        for label in g_labels:
            class_n_gt[int(label)] += 1

        # Track which GT boxes have been matched
        gt_matched = np.zeros(len(g_boxes), dtype=bool)

        # Sort predictions by score (descending)
        if len(p_scores) > 0:
            sort_idx = np.argsort(-p_scores)
            p_boxes = p_boxes[sort_idx] if len(p_boxes.shape) > 1 else p_boxes
            p_labels = p_labels[sort_idx]
            p_scores = p_scores[sort_idx]

        # Match predictions to ground truth
        for pred_idx in range(len(p_boxes)):
            pred_box = p_boxes[pred_idx]
            pred_label = int(p_labels[pred_idx])
            pred_score = p_scores[pred_idx]

            class_scores[pred_label].append(pred_score)

            # Find best matching GT box
            best_iou = 0
            best_gt_idx = -1

            for gt_idx in range(len(g_boxes)):
                if gt_matched[gt_idx]:
                    continue
                if int(g_labels[gt_idx]) != pred_label:
                    continue

                iou = compute_iou(pred_box, g_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if match
            if best_iou >= iou_threshold:
                class_tp[pred_label].append(1)
                class_fp[pred_label].append(0)
                gt_matched[best_gt_idx] = True
            else:
                class_tp[pred_label].append(0)
                class_fp[pred_label].append(1)

    # Compute AP per class
    per_class_ap = {}
    aps = []

    for cls in range(num_classes):
        n_gt = class_n_gt[cls]

        if n_gt == 0:
            # No ground truth for this class
            cls_name = class_names[cls] if class_names and cls < len(
                class_names) else f'class_{cls}'
            per_class_ap[cls_name] = 0.0
            continue

        tp = np.array(class_tp[cls])
        fp = np.array(class_fp[cls])
        scores = np.array(class_scores[cls])

        if len(scores) == 0:
            cls_name = class_names[cls] if class_names and cls < len(
                class_names) else f'class_{cls}'
            per_class_ap[cls_name] = 0.0
            continue

        # Sort by score
        sort_idx = np.argsort(-scores)
        tp = tp[sort_idx]
        fp = fp[sort_idx]

        # Compute cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (n_gt + 1e-6)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = precision[recall >= t]
            if len(prec_at_recall) > 0:
                ap += np.max(prec_at_recall)
        ap /= 11.0

        cls_name = class_names[cls] if class_names and cls < len(
            class_names) else f'class_{cls}'
        per_class_ap[cls_name] = ap
        aps.append(ap)

    map50 = np.mean(aps) if aps else 0.0

    return {
        'map50': map50,
        'per_class_ap': per_class_ap
    }


def compute_cls_metrics(pred_logits, gt_labels, class_names=None):
    """
    Compute classification metrics (accuracy) with per-class breakdown.

    Args:
        pred_logits: Predicted class logits, shape (B, num_classes)
        gt_labels: Ground truth class indices, shape (B,)
        class_names: Optional dict mapping idx to class name

    Returns:
        dict with 'accuracy' and 'per_class_accuracy'
    """
    if isinstance(pred_logits, torch.Tensor):
        pred_classes = torch.argmax(pred_logits, dim=1).cpu().numpy()
    else:
        pred_classes = np.argmax(pred_logits, axis=1)
    num_classes = pred_logits.shape[1]

    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.cpu().numpy()

    # Overall accuracy
    correct = (pred_classes == gt_labels).sum()
    total = len(gt_labels)
    accuracy = correct / total if total > 0 else 0.0

    # Per-class accuracy
    per_class_accuracy = {}
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for pred, gt in zip(pred_classes, gt_labels):
        per_class_total[int(gt)] += 1
        if pred == gt:
            per_class_correct[int(gt)] += 1

    for cls in range(num_classes):
        if per_class_total[cls] > 0:
            cls_acc = per_class_correct[cls] / per_class_total[cls]
        else:
            cls_acc = 0.0

        if class_names and cls in class_names:
            per_class_accuracy[class_names[cls]] = cls_acc
        else:
            per_class_accuracy[f'class_{cls}'] = cls_acc

    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy
    }
