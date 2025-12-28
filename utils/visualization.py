"""
Visualization utilities for inference results.
"""

from typing import Optional, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color palette for detection boxes (by class)
DETECTION_COLORS: List[str] = ['lime', 'cyan', 'yellow',
                               'magenta', 'orange', 'red', 'blue', 'pink']


def visualize_predictions(
    image: np.ndarray,
    seg_pred: Optional[torch.Tensor] = None,
    det_pred: Optional[List[List[float]]] = None,
    cls_pred: Optional[str] = None,
    cls_gt: Optional[str] = None,
    det_labels: Optional[List[int]] = None,
    det_scores: Optional[List[float]] = None,
    det_class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create a composite visualization showing all three task predictions.

    Args:
        image: Input image tensor (3, H, W) or numpy array (H, W, 3)
        seg_pred: Segmentation mask (1, H, W) or (H, W)
        det_pred: Detection boxes list of [cx, cy, w, h] in pixel coordinates
        cls_pred: Classification prediction (class name string or tensor)
        cls_gt: Ground truth class (optional, for comparison)
        det_labels: Detection class label indices (list)
        det_scores: Detection confidence scores (list)
        det_class_names: Class name list for detection labels
        save_path: Path to save the figure (displays if None)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    image_np = np.asarray(image, dtype=np.float32)
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    # Plot 1: Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot 2: Segmentation
    axes[1].imshow(image_np)
    if seg_pred is not None:
        axes[1].imshow(seg_pred, alpha=0.6, cmap='jet')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')

    # Plot 3: Detection + Classification
    axes[2].imshow(image_np)
    _draw_detection_boxes(axes[2], det_pred, det_labels, det_scores,
                          det_class_names, image_np.shape[:2])

    # Build title with classification info
    title = 'Detection + Classification'
    if cls_pred is not None:
        pred_str = cls_pred.item() if isinstance(
            cls_pred, torch.Tensor) and cls_pred.numel() == 1 else str(cls_pred)
        title += f'\nPred: {pred_str}'
    if cls_gt is not None:
        gt_str = cls_gt.item() if isinstance(
            cls_gt, torch.Tensor) and cls_gt.numel() == 1 else str(cls_gt)
        title += f' (GT: {gt_str})'
    axes[2].set_title(title)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _draw_detection_boxes(
    ax: plt.Axes,
    boxes: Optional[List[List[float]]],
    labels: Optional[List[int]],
    scores: Optional[List[float]],
    class_names: Optional[List[str]],
    img_shape: Tuple[int, int]
) -> None:
    """Draw detection bounding boxes with labels on the given axis."""
    if boxes is None or len(boxes) == 0:
        return

    img_h, img_w = img_shape

    for i, box in enumerate(boxes):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        cx, cy, w, h = box

        # Convert normalized coords to pixels if needed
        if cx <= 1.0 and cy <= 1.0 and w <= 1.0 and h <= 1.0:
            cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h

        # Center to corner format
        x1, y1 = cx - w / 2, cy - h / 2

        # Get label info
        label_idx = labels[i] if labels and i < len(labels) else None
        score = scores[i] if scores and i < len(scores) else None
        color = DETECTION_COLORS[label_idx % len(
            DETECTION_COLORS)] if label_idx is not None else 'lime'

        # Draw box
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Draw label
        if label_idx is not None or score is not None:
            if class_names and label_idx is not None and label_idx < len(class_names):
                label_text = class_names[label_idx]
            elif label_idx is not None:
                label_text = f'cls_{label_idx}'
            else:
                label_text = ''

            if score is not None:
                label_text += f' {score:.2f}'

            ax.text(x1, y1 - 2, label_text, fontsize=8, color='white',
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='square,pad=0.1', facecolor=color,
                              alpha=0.8, edgecolor='none'))
