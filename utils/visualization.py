"""
Visualization utilities for inference results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as transforms


def visualize_predictions(image, seg_pred=None, det_pred=None, cls_pred=None,
                          seg_gt=None, det_gt=None, cls_gt=None,
                          save_path=None, num_classes=4):
    """
    Create a composite visualization showing all three task predictions.
    
    Args:
        image: Input image tensor, shape (3, H, W) or PIL Image
        seg_pred: Segmentation prediction, shape (num_classes, H, W) or (H, W)
        det_pred: Detection predictions (dict with boxes, labels, scores)
        cls_pred: Classification prediction (float)
        seg_gt: Ground truth segmentation mask (optional)
        det_gt: Ground truth detection boxes (optional)
        cls_gt: Ground truth classification value (optional)
        save_path: Path to save the visualization
        num_classes: Number of segmentation classes
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        # Denormalize if needed
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_np = image * std + mean
        image_np = torch.clamp(image_np, 0, 1)
        image_np = image_np.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = np.array(image) / 255.0
    
    # Plot 1: Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot 2: Segmentation overlay
    axes[1].imshow(image_np)
    if seg_pred is not None:
        if isinstance(seg_pred, torch.Tensor):
            if seg_pred.dim() == 3:
                seg_pred = torch.argmax(seg_pred, dim=0)
            seg_pred = seg_pred.cpu().numpy()
        
        # Create colored mask
        colors = np.array([
            [0, 0, 0, 0],           # Background (transparent)
            [1, 0, 0, 0.5],         # Pan (red)
            [0, 1, 0, 0.5],         # Stirrer (green)
            [0, 0, 1, 0.5],         # Inlet pipes (blue)
        ])
        
        mask_colored = colors[seg_pred]
        axes[1].imshow(mask_colored)
    
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    # Plot 3: Detection + Classification
    axes[2].imshow(image_np)
    
    # Draw detection boxes
    if det_pred is not None:
        if isinstance(det_pred, dict):
            # Extract boxes from detection output
            # In a full implementation, this would decode from feature maps
            # For now, we'll just show a placeholder
            pass
        elif isinstance(det_pred, list) and len(det_pred) > 0:
            # Assume det_pred is list of [x, y, w, h] boxes in normalized coordinates
            for box in det_pred:
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy()
                x, y, w, h = box
                # Convert from center format to corner format
                x1 = (x - w/2) * image_np.shape[1]
                y1 = (y - h/2) * image_np.shape[0]
                width = w * image_np.shape[1]
                height = h * image_np.shape[0]
                
                rect = patches.Rectangle((x1, y1), width, height,
                                       linewidth=2, edgecolor='yellow', facecolor='none')
                axes[2].add_patch(rect)
    
    # Add classification text
    title_text = 'Detection + Classification\n'
    if cls_pred is not None:
        if isinstance(cls_pred, torch.Tensor):
            cls_pred = cls_pred.item() if cls_pred.numel() == 1 else str(cls_pred.cpu().numpy()[0])
        title_text += f'Class: {cls_pred}'
    if cls_gt is not None:
        if isinstance(cls_gt, torch.Tensor):
            cls_gt = cls_gt.item() if cls_gt.numel() == 1 else str(cls_gt.cpu().numpy()[0])
        title_text += f' (GT: {cls_gt})'
    
    axes[2].set_title(title_text)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

