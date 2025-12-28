"""
Training script for multi-task learning model.
Implements combined loss training and two-phase training.
"""

import os
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from model import MultiTaskModel
from data import SegmentationDataset, DetectionDataset, ClassificationDataset
from utils import (
    MultiTaskLoss,
    compute_seg_metrics,
    compute_det_metrics,
    compute_cls_metrics,
    decode_detections_batch
)
from config import Config

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")


def collate_detection(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for detection batches."""
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'image': images,
        'boxes': boxes,
        'labels': labels,
        'task_type': 'det'
    }


def create_data_loaders(config: Config) -> Tuple[Tuple[DataLoader, ...], Tuple[DataLoader, ...]]:
    """Create data loaders for all three tasks (train and validation)."""
    # Segmentation datasets
    seg_dataset_train = SegmentationDataset(
        data_root=config.DATA_ROOT,
        split='train',
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED,
        image_size=config.IMAGE_SIZE
    )
    seg_dataset_val = SegmentationDataset(
        data_root=config.DATA_ROOT,
        split='val',
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED,
        image_size=config.IMAGE_SIZE
    )
    seg_loader_train = DataLoader(
        seg_dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    seg_loader_val = DataLoader(
        seg_dataset_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # Detection datasets
    det_dataset_train = DetectionDataset(
        data_root=config.DATA_ROOT,
        split='train',
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED,
        image_size=config.IMAGE_SIZE
    )
    det_dataset_val = DetectionDataset(
        data_root=config.DATA_ROOT,
        split='val',
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED,
        image_size=config.IMAGE_SIZE
    )
    det_loader_train = DataLoader(
        det_dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_detection
    )
    det_loader_val = DataLoader(
        det_dataset_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_detection
    )

    # Classification datasets
    cls_dataset_train = ClassificationDataset(
        data_root=config.DATA_ROOT,
        split='train',
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED,
        image_size=config.IMAGE_SIZE
    )
    cls_dataset_val = ClassificationDataset(
        data_root=config.DATA_ROOT,
        split='val',
        val_split=config.VAL_SPLIT,
        random_seed=config.RANDOM_SEED,
        image_size=config.IMAGE_SIZE
    )
    cls_loader_train = DataLoader(
        cls_dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    cls_loader_val = DataLoader(
        cls_dataset_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    train_loaders = (seg_loader_train, det_loader_train, cls_loader_train)
    val_loaders = (seg_loader_val, det_loader_val, cls_loader_val)

    return train_loaders, val_loaders


def train_epoch(
    model: nn.Module,
    seg_loader: DataLoader,
    det_loader: DataLoader,
    cls_loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: Config,
    phase: int = 1
) -> Dict[str, float]:
    """
    Train for one epoch using combined loss backpropagation.

    All task losses are computed and summed, then a single backward pass
    updates the shared backbone and task-specific heads together.

    Args:
        model: The multi-task model.
        seg_loader: Segmentation data loader.
        det_loader: Detection data loader.
        cls_loader: Classification data loader.
        criterion: Multi-task loss function.
        optimizer: Optimizer.
        scaler: Gradient scaler for mixed precision.
        config: Configuration object.
        phase: 1 for phase 1 (backbone frozen), 2 for phase 2 (end-to-end)

    Returns:
        Dictionary with training losses.
    """
    model.train()

    total_loss = 0.0
    loss_seg_total = 0.0
    loss_det_total = 0.0
    loss_cls_total = 0.0

    # Create iterators

    seg_iter = iter(seg_loader)
    det_iter = iter(det_loader)
    cls_iter = iter(cls_loader)

    # Determine number of iterations (use the longest dataset)
    max_iter = max(len(seg_loader), len(det_loader), len(cls_loader))
    device_type = 'cuda' if 'cuda' in config.DEVICE else 'cpu'

    pbar = tqdm(range(max_iter), desc=f'Phase {phase} Training')

    for step in pbar:
        # Get batches from all tasks (restart iterator if exhausted)

        try:
            seg_batch = next(seg_iter)
        except StopIteration:
            seg_iter = iter(seg_loader)
            seg_batch = next(seg_iter)

        try:
            det_batch = next(det_iter)
        except StopIteration:
            det_iter = iter(det_loader)
            det_batch = next(det_iter)

        try:
            cls_batch = next(cls_iter)
        except StopIteration:
            cls_iter = iter(cls_loader)
            cls_batch = next(cls_iter)

        # Move data to device
        seg_images = seg_batch['image'].to(config.DEVICE)
        seg_masks = seg_batch['mask'].to(config.DEVICE)

        det_images = det_batch['image'].to(config.DEVICE)
        det_boxes = det_batch['boxes']
        det_labels = det_batch['labels']

        cls_images = cls_batch['image'].to(config.DEVICE)
        cls_labels = cls_batch['class_label'].to(config.DEVICE)

        # Zero gradients once
        optimizer.zero_grad()

        with autocast(device_type=device_type):
            # Forward pass for all tasks
            seg_outputs = model(seg_images, task='seg')
            det_outputs = model(det_images, task='det')
            cls_outputs = model(cls_images, task='cls')
            # Compute individual losses
            loss_seg = criterion.compute_seg_loss(
                seg_outputs['seg'], seg_masks, num_classes=config.NUM_SEG_CLASSES)

            loss_det = criterion.compute_det_loss(
                det_outputs['det'],
                targets={'boxes': det_boxes, 'labels': det_labels}
            )
            loss_cls = criterion.compute_cls_loss(
                cls_outputs['cls'], cls_labels,
                use_focal=config.USE_FOCAL_LOSS,
                gamma=config.FOCAL_GAMMA,
                alpha=config.FOCAL_ALPHA
            )

            # Combine losses with weights
            combined_loss = (config.WEIGHT_SEG * loss_seg +
                             config.WEIGHT_DET * loss_det +
                             config.WEIGHT_CLS * loss_cls)

        # Single backward pass and optimizer step
        scaler.scale(combined_loss).backward()

        # Gradient clipping to prevent NaN (unscale first for proper clipping)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.GRAD_CLIP_MAX_NORM)

        scaler.step(optimizer)
        scaler.update()

        # Track losses
        loss_seg_total += loss_seg.item()
        loss_det_total += loss_det.item()
        loss_cls_total += loss_cls.item()
        total_loss += combined_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{combined_loss.item():.4f}',
            'seg': f'{loss_seg.item():.4f}',
            'det': f'{loss_det.item():.4f}',
            'cls': f'{loss_cls.item():.4f}'
        })

    return {
        'total_loss': total_loss / max_iter,
        'seg_loss': loss_seg_total / max_iter,
        'det_loss': loss_det_total / max_iter,
        'cls_loss': loss_cls_total / max_iter
    }


def validate(
    model: nn.Module,
    seg_loader: DataLoader,
    det_loader: DataLoader,
    cls_loader: DataLoader,
    criterion: MultiTaskLoss,
    config: Config,
    cls_idx_to_class: Optional[Dict[int, str]] = None,
    det_class_names: Optional[List[str]] = None,
    num_det_classes: Optional[int] = None
) -> Dict[str, Any]:
    """Validate the model on all three tasks with per-class metrics."""
    model.eval()

    total_loss = 0.0

    # Accumulate predictions for metrics
    all_seg_metrics = []
    all_det_pred_boxes = []
    all_det_pred_labels = []
    all_det_pred_scores = []
    all_det_gt_boxes = []
    all_det_gt_labels = []
    all_cls_preds = []
    all_cls_labels = []

    with torch.no_grad():
        # Segmentation validation
        for batch in seg_loader:
            images = batch['image'].to(config.DEVICE)
            masks = batch['mask'].to(config.DEVICE)

            outputs = model(images, task='seg')
            loss_seg = criterion.compute_seg_loss(
                outputs['seg'], masks, num_classes=config.NUM_SEG_CLASSES)
            total_loss += config.WEIGHT_SEG * loss_seg.item()

            # Compute metrics with correct num_classes
            seg_result = compute_seg_metrics(
                outputs['seg'], masks, num_classes=config.NUM_SEG_CLASSES)
            all_seg_metrics.append(seg_result)

        # Detection validation
        for batch in det_loader:
            images = batch['image'].to(config.DEVICE)
            boxes = batch['boxes']
            labels = batch['labels']

            outputs = model(images, task='det')
            loss_det = criterion.compute_det_loss(
                outputs['det'],
                targets={'boxes': boxes, 'labels': labels}
            )
            total_loss += config.WEIGHT_DET * loss_det.item()

            # Decode predictions using shared utility function
            pred_boxes, pred_labels, pred_scores = decode_detections_batch(
                outputs['det'],
                score_threshold=config.DETECTION_SCORE_THRESHOLD,
                image_size=config.IMAGE_SIZE,
                strides=config.FPN_STRIDES,
                normalize_coords=True,
                apply_nms_filter=True
            )

            all_det_pred_boxes.extend(pred_boxes)
            all_det_pred_labels.extend(pred_labels)
            all_det_pred_scores.extend(pred_scores)
            all_det_gt_boxes.extend(boxes)
            all_det_gt_labels.extend(labels)

        # Classification validation
        for batch in cls_loader:
            images = batch['image'].to(config.DEVICE)
            class_labels = batch['class_label'].to(config.DEVICE)

            outputs = model(images, task='cls')
            loss_cls = criterion.compute_cls_loss(
                outputs['cls'], class_labels,
                use_focal=config.USE_FOCAL_LOSS,
                gamma=config.FOCAL_GAMMA,
                alpha=config.FOCAL_ALPHA
            )
            total_loss += config.WEIGHT_CLS * loss_cls.item()

            all_cls_preds.append(outputs['cls'])
            all_cls_labels.append(class_labels)

    # Aggregate segmentation metrics
    if all_seg_metrics:
        seg_miou = np.mean([m['miou'] for m in all_seg_metrics])
        # Aggregate per-class IoU
        seg_per_class = {}
        for key in all_seg_metrics[0]['per_class_iou']:
            seg_per_class[key] = np.mean(
                [m['per_class_iou'][key] for m in all_seg_metrics])
    else:
        seg_miou = 0.0
        seg_per_class = {}

    # Compute detection metrics
    if all_det_gt_boxes:
        det_result = compute_det_metrics(
            all_det_pred_boxes, all_det_pred_labels, all_det_pred_scores,
            all_det_gt_boxes, all_det_gt_labels,
            num_classes=num_det_classes,
            class_names=det_class_names
        )
        det_map50 = det_result['map50']
        det_per_class = det_result['per_class_ap']
    else:
        det_map50 = 0.0
        det_per_class = {}

    # Compute classification metrics
    if all_cls_preds:
        all_preds = torch.cat(all_cls_preds, dim=0)
        all_labels = torch.cat(all_cls_labels, dim=0)
        cls_result = compute_cls_metrics(
            all_preds, all_labels, class_names=cls_idx_to_class)
        cls_accuracy = cls_result['accuracy']
        cls_per_class = cls_result['per_class_accuracy']
    else:
        cls_accuracy = 0.0
        cls_per_class = {}

    results = {
        'total_loss': total_loss / (len(seg_loader) + len(det_loader) + len(cls_loader)),
        'seg_miou': seg_miou,
        'seg_per_class': seg_per_class,
        'det_map50': det_map50,
        'det_per_class': det_per_class,
        'cls_accuracy': cls_accuracy,
        'cls_per_class': cls_per_class,
    }

    return results


def init_wandb(
    config: Config,
    num_det_classes: int,
    num_cls_classes: int,
    det_class_names: List[str],
    cls_idx_to_class: Dict[int, str]
) -> Optional[Any]:
    """Initialize Weights & Biases logging."""
    if not WANDB_AVAILABLE or not config.USE_WANDB:
        return None

    try:
        # Get config dict and add dynamic values
        config_dict = config.to_dict()
        config_dict['num_det_classes'] = num_det_classes
        config_dict['num_cls_classes'] = num_cls_classes
        config_dict['det_class_names'] = det_class_names
        config_dict['cls_idx_to_class'] = cls_idx_to_class

        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config=config_dict,
            reinit=True
        )

        print(f"Initialized wandb run: {run.name}")
        return run
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        print("Continuing without wandb logging. Run 'wandb login' to enable.")
        return None


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
    """Log metrics to wandb if available."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    log_dict = {}
    for key, value in metrics.items():
        # Handle nested dicts (per-class metrics)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                # Convert numpy types to Python types
                if hasattr(sub_value, 'item'):
                    sub_value = sub_value.item()
                log_dict[f"{prefix}{key}/{sub_key}"] = sub_value
        else:
            # Convert numpy types to Python types
            if hasattr(value, 'item'):
                value = value.item()
            log_dict[f"{prefix}{key}"] = value

    wandb.log(log_dict, step=step)


def main() -> None:
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train multi-task model')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to dataset root directory (default: from config)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoints (default: from config)')
    args = parser.parse_args()

    config = Config()

    # Override config with command-line arguments if provided
    if args.data_root:
        Config.DATA_ROOT = args.data_root
        print(f"Using data root: {args.data_root}")
    if args.checkpoint_dir:
        Config.CHECKPOINT_DIR = args.checkpoint_dir
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Using checkpoint directory: {args.checkpoint_dir}")

    # Save config at the start of training
    config_path = config.save()

    # Create data loaders
    print("Loading datasets...")
    (seg_loader_train, det_loader_train, cls_loader_train), \
        (seg_loader_val, det_loader_val, cls_loader_val) = create_data_loaders(config)

    print(f"Train samples - Seg: {len(seg_loader_train.dataset)}, "
          f"Det: {len(det_loader_train.dataset)}, "
          f"Cls: {len(cls_loader_train.dataset)}")
    print(f"Val samples - Seg: {len(seg_loader_val.dataset)}, "
          f"Det: {len(det_loader_val.dataset)}, "
          f"Cls: {len(cls_loader_val.dataset)}")

    # Get number of classes from datasets
    num_cls_classes = cls_loader_train.dataset.num_classes
    cls_idx_to_class = cls_loader_train.dataset.idx_to_class
    num_det_classes = det_loader_train.dataset.num_classes
    det_class_names = det_loader_train.dataset.class_names  # Get from dataset

    print(f"Number of detection classes: {num_det_classes}")
    print(f"Detection classes: {det_class_names}")
    print(f"Number of classification classes: {num_cls_classes}")
    print(f"Classification class mapping: {cls_idx_to_class}")

    # Initialize wandb
    wandb_run = init_wandb(config, num_det_classes, num_cls_classes,
                           det_class_names, cls_idx_to_class)

    # Create model
    print("Creating model...")
    model = MultiTaskModel(
        num_seg_classes=config.NUM_SEG_CLASSES,
        num_det_classes=num_det_classes,
        num_cls_classes=num_cls_classes,
        pretrained_backbone=True,
        fpn_channels=config.FPN_CHANNELS,
    )
    model = model.to(config.DEVICE)

    # Loss function
    criterion = MultiTaskLoss(
        weight_seg=config.WEIGHT_SEG,
        weight_det=config.WEIGHT_DET,
        weight_cls=config.WEIGHT_CLS,
        fpn_strides=config.FPN_STRIDES
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR_PHASE1)

    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None

    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')

    def save_best_model(model, optimizer, epoch, val_loss, phase):
        """Save the best model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'phase': phase,
            'num_det_classes': num_det_classes,
            'num_cls_classes': num_cls_classes,
            'det_class_names': det_class_names,
            'cls_idx_to_class': cls_idx_to_class
        }
        torch.save(checkpoint, best_model_path)
        print(f"Saved best model (val_loss: {val_loss:.4f})")

    # Phase 1: Freeze backbone
    print("\n=== Phase 1: Training FPN + Heads (Backbone Frozen) ===")
    model.freeze_backbone()

    for epoch in range(config.NUM_EPOCHS_PHASE1):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS_PHASE1}")
        train_metrics = train_epoch(
            model, seg_loader_train, det_loader_train, cls_loader_train,
            criterion, optimizer, scaler, config, phase=1
        )

        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Seg: {train_metrics['seg_loss']:.4f}, "
              f"Det: {train_metrics['det_loss']:.4f}, "
              f"Cls: {train_metrics['cls_loss']:.4f})")

        # Log training metrics to wandb
        global_step = epoch + 1
        log_wandb({
            'train/loss': train_metrics['total_loss'],
            'train/seg_loss': train_metrics['seg_loss'],
            'train/det_loss': train_metrics['det_loss'],
            'train/cls_loss': train_metrics['cls_loss'],
            'phase': 1,
            'epoch': global_step
        }, step=global_step)

        # Validation
        if (epoch + 1) % config.VALIDATION_FREQ == 0:
            val_metrics = validate(
                model, seg_loader_val, det_loader_val, cls_loader_val,
                criterion, config, cls_idx_to_class, det_class_names, num_det_classes
            )
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(
                f"  Seg mIoU: {val_metrics['seg_miou']:.4f} | Per-class: {val_metrics['seg_per_class']}")
            print(
                f"  Det mAP@0.5: {val_metrics['det_map50']:.4f} | Per-class: {val_metrics['det_per_class']}")
            print(
                f"  Cls Accuracy: {val_metrics['cls_accuracy']:.4f} | Per-class: {val_metrics['cls_per_class']}")

            # Log validation metrics to wandb
            log_wandb({
                'val/loss': val_metrics['total_loss'],
                'val/seg_miou': val_metrics['seg_miou'],
                'val/det_map50': val_metrics['det_map50'],
                'val/cls_accuracy': val_metrics['cls_accuracy'],
                'val/seg_per_class': val_metrics['seg_per_class'],
                'val/det_per_class': val_metrics['det_per_class'],
                'val/cls_per_class': val_metrics['cls_per_class'],
                'best_loss': best_loss
            }, step=global_step)

            # Check for improvement
            if val_metrics['total_loss'] < best_loss:
                best_loss = val_metrics['total_loss']
                patience_counter = 0
                save_best_model(model, optimizer, epoch, best_loss, phase=1)
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            # Early stopping check
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping triggered after {epoch + 1} epochs (Phase 1)")
                break

    # Phase 2: Unfreeze backbone, end-to-end training
    print("\n=== Phase 2: End-to-End Training (All Layers) ===")
    model.unfreeze_backbone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR_PHASE2)
    phase_2_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.LR_GAMMA)

    # Reset patience counter for phase 2 but keep best_loss
    patience_counter = 0

    # Calculate global step offset from phase 1
    phase1_epochs = config.NUM_EPOCHS_PHASE1

    for epoch in range(config.NUM_EPOCHS_PHASE2):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS_PHASE2}")
        train_metrics = train_epoch(
            model, seg_loader_train, det_loader_train, cls_loader_train,
            criterion, optimizer, scaler, config, phase=2
        )

        phase_2_lr_scheduler.step()

        print(f"Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Seg: {train_metrics['seg_loss']:.4f}, "
              f"Det: {train_metrics['det_loss']:.4f}, "
              f"Cls: {train_metrics['cls_loss']:.4f})")

        # Log training metrics to wandb
        global_step = phase1_epochs + epoch + 1
        log_wandb({
            'train/loss': train_metrics['total_loss'],
            'train/seg_loss': train_metrics['seg_loss'],
            'train/det_loss': train_metrics['det_loss'],
            'train/cls_loss': train_metrics['cls_loss'],
            'phase': 2,
            'epoch': global_step,
            'learning_rate': phase_2_lr_scheduler.get_last_lr()[0]
        }, step=global_step)

        # Validation
        if (epoch + 1) % config.VALIDATION_FREQ == 0:
            val_metrics = validate(
                model, seg_loader_val, det_loader_val, cls_loader_val,
                criterion, config, cls_idx_to_class, det_class_names, num_det_classes
            )
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(
                f"  Seg mIoU: {val_metrics['seg_miou']:.4f} | Per-class: {val_metrics['seg_per_class']}")
            print(
                f"  Det mAP@0.5: {val_metrics['det_map50']:.4f} | Per-class: {val_metrics['det_per_class']}")
            print(
                f"  Cls Accuracy: {val_metrics['cls_accuracy']:.4f} | Per-class: {val_metrics['cls_per_class']}")

            # Log validation metrics to wandb
            log_wandb({
                'val/loss': val_metrics['total_loss'],
                'val/seg_miou': val_metrics['seg_miou'],
                'val/det_map50': val_metrics['det_map50'],
                'val/cls_accuracy': val_metrics['cls_accuracy'],
                'val/seg_per_class': val_metrics['seg_per_class'],
                'val/det_per_class': val_metrics['det_per_class'],
                'val/cls_per_class': val_metrics['cls_per_class'],
                'best_loss': best_loss
            }, step=global_step)

            # Check for improvement
            if val_metrics['total_loss'] < best_loss:
                best_loss = val_metrics['total_loss']
                patience_counter = 0
                save_best_model(model, optimizer, epoch, best_loss, phase=2)
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            # Early stopping check
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping triggered after {epoch + 1} epochs (Phase 2)")
                break

    # Finish wandb run
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

    print(f"\nTraining completed! Best validation loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
