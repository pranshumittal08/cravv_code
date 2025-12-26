"""
Configuration file for hyperparameters and paths.
"""

import os
import json
from datetime import datetime
import torch


class Config:
    """Configuration class for multi-task learning system."""

    # ========== Paths ==========
    # Update with your dataset path
    DATA_ROOT = "/home/pranshu21/datasets/assignment-dataset"
    CHECKPOINT_DIR = 'save/trial_5'
    OUTPUT_DIR = 'outputs'

    # Create directories if they don't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== Weights & Biases ==========
    USE_WANDB = True  # Enable/disable wandb logging
    WANDB_PROJECT = "multi-task-cooking-pan"
    WANDB_ENTITY = "pranshu21"  # Set to your wandb username/team, or None for default

    # ========== Model ==========
    BACKBONE = 'resnet50'
    FPN_CHANNELS = 256
    NUM_SEG_CLASSES = 1  # Binary segmentation (foreground/background)
    # Will be determined from COCO annotations
    NUM_DET_CLASSES = None
    # Will be determined from dataset (number of unique class folders)
    NUM_CLS_CLASSES = None

    # ========== Training ==========
    BATCH_SIZE = 16
    NUM_EPOCHS_PHASE1 = 10  # Freeze backbone
    NUM_EPOCHS_PHASE2 = 30  # End-to-end training
    LR_PHASE1 = 1e-3
    LR_PHASE2 = 1e-4
    LR_GAMMA = 0.95

    # Loss weights
    WEIGHT_SEG = 1.0
    WEIGHT_DET = 1.0
    WEIGHT_CLS = 0.5

    # Focal loss parameters (for imbalanced classification)
    USE_FOCAL_LOSS = True  # Use focal loss for classification (sigmoid-based)
    # Focusing parameter (higher = more focus on hard examples)
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25  # Balancing factor for focal loss

    # ========== Data ==========
    IMAGE_SIZE = 256
    NUM_WORKERS = 10
    PIN_MEMORY = True
    VAL_SPLIT = 0.2  # Fraction of data for validation (80/20 split)
    RANDOM_SEED = 42  # Random seed for reproducible train/val split

    # ========== Training settings ==========
    USE_AMP = True  # Mixed precision training
    VALIDATION_FREQ = 1  # Validate every N epochs
    EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for N epochs
    GRAD_CLIP_MAX_NORM = 1.0  # Gradient clipping max norm (prevents NaN)

    # ========== Device ==========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== Logging ==========
    LOG_INTERVAL = 10  # Log every N iterations

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary (excludes non-serializable items)."""
        config_dict = {}
        for key in dir(cls):
            if key.startswith('_') or callable(getattr(cls, key)):
                continue
            value = getattr(cls, key)
            # Skip non-serializable values
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config_dict[key] = value
        return config_dict

    @classmethod
    def save(cls, filepath=None):
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
