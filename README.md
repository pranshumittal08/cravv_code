# Multi-Task Learning Vision System

A multi-task deep learning model for cooking pan monitoring with shared ResNet-50 backbone and three task heads: segmentation, detection, and classification.

## Project Overview

This system performs three related computer vision tasks simultaneously:
1. **Segmentation**: Binary segmentation of cooking pan region
2. **Detection**: Detect pan, stirrer, inlet pipes using bounding boxes (FCOS-style anchor-free detection)
3. **Classification**: Classify onion cooking states

All tasks share the same scene type but use different image samples from separate datasets.

## Architecture

### Backbone + FPN

```
Input (256×256 RGB)
  ↓
ResNet-50 (pretrained ImageNet) → Extract [C2, C3, C4, C5]
  ↓
FPN → Generate [P2, P3, P4, P5] (256 channels each)
  ↓
Three Task Heads (parallel)
```

### Task Heads

1. **Segmentation Head**: 
   - Input: P2 features (highest resolution, 1/4 scale)
   - Architecture: 2 transposed conv layers + classifier
   - Output: 256×256 binary segmentation mask
   - Loss: BCE with Logits

2. **Detection Head**:
   - Input: P3, P4, P5 (multi-scale features)
   - Architecture: Anchor-free FCOS-style detection
   - Output: Bounding boxes + class labels + centerness
   - Loss: Focal Loss (classification) + Smooth L1 (regression) + BCE (centerness)

3. **Classification Head**:
   - Input: C5 features (before FPN, 2048 channels)
   - Architecture: Global Average Pooling → FC layers
   - Output: Cooking state class
   - Loss: Focal Loss

### Multi-Task Loss

```
L_total = 1.0 * L_seg + 1.0 * L_det + 2.0 * L_cls
```

## Dataset Structure

The system expects three separate datasets organized as follows:

```
data/
├── segmentation/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── masks/
│       ├── image1.png
│       └── ...
├── detection/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── _annotations.coco.json
└── classification/
    └── images/
        ├── class_0/
        │   ├── image1.jpg
        │   └── ...
        ├── class_1/
        │   └── ...
        └── ...
```

- **Segmentation**: RGB images and corresponding binary masks (same filenames)
- **Detection**: RGB images and COCO-format JSON annotations
- **Classification**: RGB images organized in folders named by class

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Training

### Run Training

```bash
python train.py --data_root /path/to/dataset --checkpoint_dir save/exp_001
```

**Arguments:**
- `--data_root`: Path to dataset root directory (overrides config)
- `--checkpoint_dir`: Directory to save checkpoints (overrides config)

### Training Features

- Two-phase training: freeze backbone → end-to-end fine-tuning
- Mixed precision training (AMP) for faster training
- Automatic checkpointing with best model selection
- Early stopping based on validation loss
- Gradient clipping for training stability
- Weights & Biases integration for experiment tracking
- Config saved with checkpoint for reproducibility

### Checkpoints

Training saves the following to the checkpoint directory:
- `best_model.pth`: Best model based on validation loss
- `config_<timestamp>.json`: Configuration used for training

The checkpoint includes:
- Model weights
- Number of classes for each task
- Class name mappings for detection and classification

## Inference

### Command Line Inference

Process all images in a directory:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input_dir /path/to/images \
    --output_dir outputs \
    --visualize \
    --score_threshold 0.5 \
    --nms_threshold 0.4
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint (required)
- `--input_dir`: Directory containing images for inference (required)
- `--output_dir`: Directory for output visualizations (default: `outputs`)
- `--config`: Path to config JSON (auto-detected from checkpoint dir if not specified)
- `--visualize`: Generate and save visualizations (optional flag)
- `--score_threshold`: Detection confidence threshold (default: from config)
- `--nms_threshold`: NMS IoU threshold (default: from config)

### Programmatic API

For integration in applications with streaming data:

```python
from inference import load_model, infer_image
from config import Config

# Load model once
config = Config.load('checkpoints/config_20241228.json')
# Or auto-detect config from checkpoint directory
config = Config()

model, checkpoint_info = load_model(
    'checkpoints/best_model.pth',
    config,
    config.DEVICE
)

# Run inference on a single image
# Accepts: file path (str), PIL.Image, or numpy array (H, W, C) in RGB
result = infer_image(
    model=model,
    image=frame,  # numpy array from camera/video
    checkpoint_info=checkpoint_info,
    config=config,
    score_threshold=0.5,
    nms_threshold=0.4
)

# Access outputs
seg_mask = result['segmentation']['mask']       # (H, W) numpy array, values 0 or 255
boxes = result['detection']['boxes']            # List of [x1, y1, x2, y2]
labels = result['detection']['labels']          # List of class indices
scores = result['detection']['scores']          # List of confidence scores
class_names = result['detection']['class_names'] # List of class names
pred_class = result['classification']['class_name']  # Predicted class string
pred_idx = result['classification']['class_idx']     # Predicted class index
```

### Batch Processing

```python
from inference import load_model, run_inference
from config import Config

config = Config()
model, checkpoint_info = load_model('checkpoints/best_model.pth', config, config.DEVICE)

# Process all images in a directory
results = run_inference(
    model=model,
    image_dir='/path/to/images',
    checkpoint_info=checkpoint_info,
    config=config,
    output_dir='outputs',
    visualize=True,  # Set to False to skip visualization
    score_threshold=0.5,
    nms_threshold=0.4
)

# results is a list of dicts, one per image
for result in results:
    print(f"Image: {result['image_name']}")
    print(f"  Detections: {len(result['detection']['boxes'])}")
    print(f"  Classification: {result['classification']['class_name']}")
```

## Configuration

Key parameters in `config.py`:

```python
# Model
NUM_SEG_CLASSES = 1          # Binary segmentation
FPN_CHANNELS = 256

# Detection
FPN_STRIDES = (8, 16, 32)    # P3, P4, P5 strides
DETECTION_SCORE_THRESHOLD = 0.3
NMS_IOU_THRESHOLD = 0.5

# Training
BATCH_SIZE = 16
NUM_EPOCHS_PHASE1 = 3        # Freeze backbone phase
NUM_EPOCHS_PHASE2 = 10       # End-to-end phase
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4

# Loss weights
WEIGHT_SEG = 1.0
WEIGHT_DET = 1.0
WEIGHT_CLS = 2.0

# Focal loss
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
```

### Config Persistence

Config is automatically saved with checkpoints and can be loaded during inference:

```python
# Save config
Config.save('path/to/config.json')

# Load config
config = Config.load('path/to/config.json')

# Auto-find config in checkpoint directory
config_path = Config.find_in_directory('checkpoints/')
```

## Project Structure

```
project/
├── model/
│   ├── __init__.py
│   ├── backbone.py          # ResNet50 feature extractor
│   ├── fpn.py               # Feature Pyramid Network
│   ├── heads.py             # Segmentation, Detection, Classification heads
│   └── multitask_model.py   # Main model combining all components
├── data/
│   ├── __init__.py
│   ├── base_dataset.py      # Base dataset class
│   ├── seg_dataset.py       # Segmentation dataset
│   ├── det_dataset.py       # Detection dataset (COCO format)
│   └── cls_dataset.py       # Classification dataset
├── utils/
│   ├── __init__.py
│   ├── losses.py            # Multi-task loss computation
│   ├── fcos_losses.py       # FCOS detection losses
│   ├── detection_utils.py   # Detection decoding & NMS
│   ├── metrics.py           # Evaluation metrics (mIoU, mAP, accuracy)
│   └── visualization.py     # Inference visualization
├── train.py                 # Training script
├── inference.py             # Inference API & CLI
├── config.py                # Configuration class
├── requirements.txt
└── README.md
```

## Implementation Details

### Data Augmentation

Training augmentations:
- Random horizontal flip (50% probability)
- Color jitter (brightness, contrast, saturation, hue)
- Resize to 256×256
- ImageNet normalization

### Training Strategy

1. **Phase 1** (optional): Freeze backbone, train heads only
2. **Phase 2**: End-to-end training with lower learning rate

### Memory Efficiency

- Mixed precision training (AMP) reduces memory usage
- Gradient clipping prevents gradient explosion
- Batch size configurable in `config.py`

### Evaluation Metrics

- **Segmentation**: mIoU (mean Intersection over Union)
- **Detection**: mAP@0.5 with per-class breakdown
- **Classification**: Accuracy with per-class breakdown

## Weights & Biases Integration

Enable experiment tracking:

```python
# In config.py
USE_WANDB = True
WANDB_PROJECT = "multi-task-cooking-pan"
WANDB_ENTITY = "your-username"
```

Logged metrics include:
- Training/validation losses per task
- mIoU, mAP, accuracy
- Learning rate schedules
- Configuration hyperparameters
