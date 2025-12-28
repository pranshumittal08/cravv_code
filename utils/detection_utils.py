"""
Detection utility functions for inference.
Includes NMS and decoding of FCOS-style detection outputs.
"""

from typing import List, Tuple, Union

import torch
import numpy as np
from torchvision.ops import nms


def apply_nms(
    boxes: Union[List[List[float]], np.ndarray],
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
    iou_threshold: float = 0.5
) -> Tuple[List[List[float]], List[int], List[float]]:
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.

    Args:
        boxes: List or array of [cx, cy, w, h] boxes
        labels: List or array of class labels
        scores: List or array of confidence scores
        iou_threshold: IoU threshold for NMS

    Returns:
        Filtered boxes, labels, scores as lists
    """
    if len(boxes) == 0:
        return [], [], []

    # Convert to tensors
    boxes_arr = np.asarray(boxes)
    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    boxes_tensor = torch.tensor(boxes_arr, dtype=torch.float32)
    scores_tensor = torch.tensor(scores_arr, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_arr, dtype=torch.long)

    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2] for NMS
    boxes_xyxy = torch.zeros_like(boxes_tensor)
    boxes_xyxy[:, 0] = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2  # y2

    keep_indices = []
    unique_labels = labels_tensor.unique()

    for label in unique_labels:
        label_mask = labels_tensor == label
        label_indices = torch.where(label_mask)[0]

        if len(label_indices) > 0:
            label_boxes = boxes_xyxy[label_mask]
            label_scores = scores_tensor[label_mask]

            # Apply NMS for this class
            keep = nms(label_boxes, label_scores, iou_threshold)
            keep_indices.extend(label_indices[keep].tolist())

    # Sort by score (descending) for consistent ordering
    keep_indices = sorted(
        keep_indices, key=lambda i: scores_arr[i], reverse=True)

    # Filter results
    filtered_boxes = [boxes_arr[i].tolist() for i in keep_indices]
    filtered_labels = [int(labels_arr[i]) for i in keep_indices]
    filtered_scores = [float(scores_arr[i]) for i in keep_indices]

    return filtered_boxes, filtered_labels, filtered_scores


def decode_detections_batch(
    det_outputs: dict,
    image_size: int = 256,
    score_threshold: float = 0.3,
    strides: Tuple[int, ...] = (8, 16, 32),
    normalize_coords: bool = True,
    nms_threshold: float = 0.5,
    apply_nms_filter: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Decode FCOS-style detection outputs to boxes, labels, and scores for a batch.

    Args:
        det_outputs: Detection outputs from model with batch dimension
            - 'cls_logits': List of tensors (B, num_classes+1, H, W) per scale
            - 'reg_preds': List of tensors (B, 4, H, W) per scale
            - 'centerness': List of tensors (B, 1, H, W) per scale
        image_size: Input image size
        score_threshold: Minimum score threshold for filtering
        strides: Feature map strides for each FPN level
        normalize_coords: If True, return normalized coords [0, 1]; else pixel coords
        nms_threshold: IoU threshold for NMS (only used if apply_nms_filter=True)
        apply_nms_filter: If True, apply NMS per image

    Returns:
        batch_boxes: List of numpy arrays of boxes [cx, cy, w, h] per image
        batch_labels: List of numpy arrays of labels per image
        batch_scores: List of numpy arrays of scores per image
    """
    batch_size = det_outputs['cls_logits'][0].shape[0]
    batch_boxes = []
    batch_labels = []
    batch_scores = []

    for batch_idx in range(batch_size):
        img_boxes = []
        img_labels = []
        img_scores = []

        for scale_idx, stride in enumerate(strides):
            # (num_classes+1, H, W)
            cls_logits = det_outputs['cls_logits'][scale_idx][batch_idx]
            # (4, H, W)
            reg_preds = det_outputs['reg_preds'][scale_idx][batch_idx]
            # (1, H, W)
            centerness = det_outputs['centerness'][scale_idx][batch_idx]

            # Get class probabilities (exclude background class 0)
            cls_probs = torch.sigmoid(cls_logits[1:])  # (num_classes, H, W)
            centerness_prob = torch.sigmoid(centerness.squeeze(0))  # (H, W)

            # Combine with centerness
            combined_scores = cls_probs * centerness_prob.unsqueeze(0)

            # Get max score and class per location
            max_scores, max_classes = combined_scores.max(dim=0)  # (H, W)

            # Filter by threshold
            mask = max_scores > score_threshold

            if mask.sum() > 0:
                # Get locations
                y_coords, x_coords = torch.where(mask)

                for y, x in zip(y_coords, x_coords):
                    # Get box (l, t, r, b) -> convert to (cx, cy, w, h)
                    l, t, r, b = reg_preds[:, y, x].cpu().detach().numpy()

                    # Convert from distances to box coordinates
                    loc_x = (x.item() + 0.5) * stride
                    loc_y = (y.item() + 0.5) * stride

                    x1 = loc_x - l * stride
                    y1 = loc_y - t * stride
                    x2 = loc_x + r * stride
                    y2 = loc_y + b * stride

                    # Clamp to image bounds
                    x1 = np.clip(x1, 0, image_size)
                    y1 = np.clip(y1, 0, image_size)
                    x2 = np.clip(x2, 0, image_size)
                    y2 = np.clip(y2, 0, image_size)

                    # Convert to (cx, cy, w, h)
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    # Skip invalid boxes
                    if w <= 0 or h <= 0:
                        continue

                    # Normalize if requested
                    if normalize_coords:
                        cx = cx / image_size
                        cy = cy / image_size
                        w = w / image_size
                        h = h / image_size

                    img_boxes.append([cx, cy, w, h])
                    img_labels.append(max_classes[y, x].item())
                    img_scores.append(max_scores[y, x].item())

        # Apply NMS per image if requested
        if apply_nms_filter and len(img_boxes) > 0:
            img_boxes, img_labels, img_scores = apply_nms(
                img_boxes, img_labels, img_scores, nms_threshold
            )

        # Convert to numpy arrays
        batch_boxes.append(np.array(img_boxes)
                           if img_boxes else np.zeros((0, 4)))
        batch_labels.append(np.array(img_labels)
                            if img_labels else np.zeros(0))
        batch_scores.append(np.array(img_scores)
                            if img_scores else np.zeros(0))

    return batch_boxes, batch_labels, batch_scores


def decode_detection_output(
    det_output: dict,
    image_size: int = 256,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.5,
    strides: Tuple[int, ...] = (8, 16, 32),
    normalize_coords: bool = False,
    apply_nms_filter: bool = True
) -> Tuple[List[List[float]], List[int], List[float]]:
    """
    Decode FCOS-style detection output for single-image inference.

    This is a convenience wrapper around decode_detections_batch for
    single-image inference with NMS. For batch processing, use
    decode_detections_batch directly.

    Args:
        det_output: Dictionary with 'cls_logits', 'reg_preds', 'centerness'
        image_size: Input image size
        score_threshold: Score threshold for filtering
        nms_threshold: IoU threshold for NMS
        strides: Feature map strides for each FPN level
        normalize_coords: If True, return normalized coords [0, 1]; else pixel coords
        apply_nms_filter: If True, apply NMS to filter overlapping boxes

    Returns:
        boxes: List of bounding boxes [cx, cy, w, h]
        labels: List of class labels
        scores: List of confidence scores
    """
    # Use batch decoder (handles single image as batch of 1)
    batch_boxes, batch_labels, batch_scores = decode_detections_batch(
        det_output,
        image_size=image_size,
        score_threshold=score_threshold,
        strides=strides,
        normalize_coords=normalize_coords,
        nms_threshold=nms_threshold,
        apply_nms_filter=apply_nms_filter
    )

    # Extract first (and only) image results, convert to lists
    boxes = batch_boxes[0].tolist() if len(batch_boxes[0]) > 0 else []
    labels = batch_labels[0].astype(
        int).tolist() if len(batch_labels[0]) > 0 else []
    scores = batch_scores[0].tolist() if len(batch_scores[0]) > 0 else []

    return boxes, labels, scores
