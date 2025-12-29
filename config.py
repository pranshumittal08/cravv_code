"""
Configuration file for hyperparameters and paths.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch


class Config:
    """Configuration class for multi-task learning system."""

    # ========== Paths ==========
    # Update with your dataset path
    DATA_ROOT: str = "/home/pranshu21/datasets/assignment-dataset"
    CHECKPOINT_DIR: str = 'save/trial_12'
    OUTPUT_DIR: str = 'outputs'

    # ========== Weights & Biases ==========
    USE_WANDB: bool = True  # Enable/disable wandb logging
    WANDB_PROJECT: str = "multi-task-cooking-pan"
    # Set to your wandb username/team, or None for default
    WANDB_ENTITY: str = "pranshu21"

    # ========== Model ==========
    BACKBONE: str = 'resnet50'
    FPN_CHANNELS: int = 256
    NUM_SEG_CLASSES: int = 1  # Binary segmentation (foreground/background)
    # Will be determined from COCO annotations
    NUM_DET_CLASSES: Optional[int] = None
    # Will be determined from dataset (number of unique class folders)
    NUM_CLS_CLASSES: Optional[int] = None

    # ========== Detection Head Constants ==========
    # Feature map strides for each FPN level (P3, P4, P5)
    FPN_STRIDES: Tuple[int, ...] = (8, 16, 32)
    # Score threshold for filtering detections during inference
    DETECTION_SCORE_THRESHOLD: float = 0.3
    # IoU threshold for Non-Maximum Suppression
    NMS_IOU_THRESHOLD: float = 0.5

    # ========== Training ==========
    BATCH_SIZE: int = 16
    NUM_EPOCHS_PHASE1: int = 0  # Freeze backbone
    NUM_EPOCHS_PHASE2: int = 40  # End-to-end training
    LR_PHASE1: float = 1e-3
    LR_PHASE2: float = 1e-4
    LR_GAMMA: float = 0.95

    # Loss weights
    WEIGHT_SEG: float = 1.0
    WEIGHT_DET: float = 1.0
    WEIGHT_CLS: float = 2.0

    # Focal loss parameters (for imbalanced classification)
    # Use focal loss for classification (sigmoid-based)
    USE_FOCAL_LOSS: bool = True
    # Focusing parameter (higher = more focus on hard examples)
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.25  # Balancing factor for focal loss

    # ========== Data ==========
    IMAGE_SIZE: int = 256
    NUM_WORKERS: int = 10
    PIN_MEMORY: bool = True
    VAL_SPLIT: float = 0.2  # Fraction of data for validation (80/20 split)
    RANDOM_SEED: int = 42  # Random seed for reproducible train/val split

    # ========== ImageNet Normalization Constants ==========
    IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # ========== Training settings ==========
    USE_AMP: bool = True  # Mixed precision training
    VALIDATION_FREQ: int = 1  # Validate every N epochs
    EARLY_STOPPING_PATIENCE: int = 5  # Stop if no improvement for N epochs
    # Gradient clipping max norm (prevents NaN)
    GRAD_CLIP_MAX_NORM: float = 1.0

    # ========== Device ==========
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== Logging ==========
    LOG_INTERVAL: int = 10  # Log every N iterations

    def __init__(self) -> None:
        """Initialize config and create directories."""
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary (excludes non-serializable items)."""
        config_dict = {}
        for key in dir(cls):
            if key.startswith('_') or callable(getattr(cls, key)):
                continue
            value = getattr(cls, key)
            # Skip non-serializable values
            if isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                # Convert tuples to lists for JSON serialization
                if isinstance(value, tuple):
                    value = list(value)
                config_dict[key] = value
        return config_dict

    @classmethod
    def save(cls, filepath: Optional[str] = None) -> str:
        """Save config to JSON file.

        Args:
            filepath: Path to save config. If None, saves to CHECKPOINT_DIR with timestamp.

        Returns:
            filepath: Path where config was saved.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                cls.CHECKPOINT_DIR, f'config_{timestamp}.json')

        config_dict = cls.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Config saved to: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Load config from JSON file.

        Args:
            filepath: Path to the config JSON file.

        Returns:
            Config instance with loaded values.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Update class attributes with loaded values
        for key, value in config_dict.items():
            if hasattr(cls, key):
                # Convert lists back to tuples for tuple-typed attributes
                current_value = getattr(cls, key)
                if isinstance(current_value, tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(cls, key, value)

        print(f"Config loaded from: {filepath}")
        return cls()

    @classmethod
    def find_in_directory(cls, directory: str) -> Optional[str]:
        """Find the most recent config file in a directory.

        Args:
            directory: Directory to search for config files.

        Returns:
            Path to the config file, or None if not found.
        """
        if not os.path.isdir(directory):
            return None

        config_files = [
            f for f in os.listdir(directory)
            if f.startswith('config_') and f.endswith('.json')
        ]

        if not config_files:
            return None

        # Sort by filename (which includes timestamp) to get most recent
        config_files.sort(reverse=True)
        return os.path.join(directory, config_files[0])
