"""
Inference script for multi-task model.
Provides both single-image inference API and batch processing for folders.
"""

import os
from typing import Optional, List, Dict, Any, Tuple, Union
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import MultiTaskModel
from utils import visualize_predictions, decode_detection_output
from config import Config


def get_image_files(
    directory: str,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
) -> List[str]:
    """Get list of image files from a directory.

    Args:
        directory: Directory to search for images.
        extensions: Tuple of valid image extensions.

    Returns:
        List of image file paths.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")

    image_files = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(extensions):
            image_files.append(os.path.join(directory, filename))

    return image_files


def load_model(
    checkpoint_path: str,
    config: "Config",
    device: str
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        config: Configuration object.
        device: Device to load the model on.

    Returns:
        model: Loaded model
        checkpoint_info: Dict with class info from checkpoint
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

    # Get class info from checkpoint (required)
    num_cls_classes = checkpoint.get('num_cls_classes')
    num_det_classes = checkpoint.get('num_det_classes')
    det_class_names = checkpoint.get('det_class_names')
    cls_idx_to_class = checkpoint.get('cls_idx_to_class')

    if num_cls_classes is None or num_det_classes is None:
        raise ValueError(
            "Checkpoint missing class information. "
            "Ensure the checkpoint was saved with 'num_cls_classes' and 'num_det_classes'."
        )

    if det_class_names is None or cls_idx_to_class is None:
        print("Warning: Checkpoint missing class name mappings. Using generic names.")
        det_class_names = [f"det_class_{i}" for i in range(num_det_classes)]
        cls_idx_to_class = {i: f"cls_{i}" for i in range(num_cls_classes)}

    model = MultiTaskModel(
        num_seg_classes=config.NUM_SEG_CLASSES,
        num_det_classes=num_det_classes,
        num_cls_classes=num_cls_classes,
        pretrained_backbone=False,
        fpn_channels=getattr(config, 'FPN_CHANNELS', 256),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    checkpoint_info = {
        'num_det_classes': num_det_classes,
        'num_cls_classes': num_cls_classes,
        'det_class_names': det_class_names,
        'cls_idx_to_class': cls_idx_to_class
    }

    return model, checkpoint_info


def preprocess_image(
    image: Union[str, Image.Image, np.ndarray],
    image_size: int = 256,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess an image for inference.

    Args:
        image: Image path, PIL Image, or numpy array (H, W, C) in RGB format.
        image_size: Target image size.
        mean: Normalization mean values.
        std: Normalization std values.

    Returns:
        image_tensor: Preprocessed image tensor (1, 3, H, W)
        image_display: Tensor for display (3, H, W) - normalized but not batched
    """
    if isinstance(image, str):
        pil_image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image.convert('RGB')
    else:
        raise TypeError(
            f"Expected str, PIL.Image, or np.ndarray, got {type(image)}"
        )

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    image_tensor = transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor


def infer_image(
    model: torch.nn.Module,
    image: Union[str, Image.Image, np.ndarray],
    checkpoint_info: Dict[str, Any],
    config: "Config",
    score_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Run inference on a single image.

    This function is designed for integration in applications with streaming data.
    It processes a single image and returns structured outputs from all three heads.

    Args:
        model: Trained model (must be in eval mode)
        image: Input image - can be:
            - str: Path to image file
            - PIL.Image: PIL Image object
            - np.ndarray: Numpy array (H, W, C) in RGB format
        checkpoint_info: Dict with class names from checkpoint
        config: Configuration object
        score_threshold: Detection score threshold (overrides config if provided)
        nms_threshold: NMS IoU threshold (overrides config if provided)

    Returns:
        Dict containing:
            - 'segmentation': Segmentation mask tensor (H, W) with class indices
            - 'detection': Dict with 'boxes', 'labels', 'scores', 'class_names'
            - 'classification': Dict with 'class_idx', 'class_name'
            - 'image_tensor': Preprocessed image tensor for visualization
    """
    device = config.DEVICE

    # Use provided thresholds or fall back to config
    det_score_threshold = (
        score_threshold if score_threshold is not None
        else config.DETECTION_SCORE_THRESHOLD
    )
    det_nms_threshold = (
        nms_threshold if nms_threshold is not None
        else config.NMS_IOU_THRESHOLD
    )

    # Get class mappings from checkpoint
    det_class_names = checkpoint_info['det_class_names']
    cls_idx_to_class = checkpoint_info['cls_idx_to_class']

    # Handle different input types
    image_tensor = preprocess_image(
        image,
        image_size=config.IMAGE_SIZE,
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD
    )
    image_tensor = image_tensor.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)

    # Process segmentation output
    seg_logits = outputs['seg'][0]  # (1, H, W)
    seg_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    seg_mask = (seg_mask > 0.5).astype(np.uint8) * 255

    # Process classification output
    cls_logits = outputs['cls'][0]  # (num_classes,)
    pred_class_idx = torch.argmax(cls_logits).item()
    pred_class_name = cls_idx_to_class[pred_class_idx]

    # Process detection output
    det_pred = outputs['det']
    boxes, labels, scores = decode_detection_output(
        det_pred,
        image_size=config.IMAGE_SIZE,
        score_threshold=det_score_threshold,
        nms_threshold=det_nms_threshold,
        strides=config.FPN_STRIDES
    )

    # Map detection labels to class names
    det_class_names_pred = [det_class_names[l]
                            for l in labels] if labels else []

    result = {
        'segmentation': {
            'mask': seg_mask,
        },
        'detection': {
            'boxes': boxes,  # List of [x1, y1, x2, y2]
            'labels': labels,  # List of class indices
            'scores': scores,  # List of confidence scores
            'class_names': det_class_names_pred  # List of class names
        },
        'classification': {
            'class_idx': pred_class_idx,
            'class_name': pred_class_name,
        }
    }
    return result


def run_inference(
    model: torch.nn.Module,
    image_dir: str,
    checkpoint_info: Dict[str, Any],
    config: "Config",
    output_dir: Optional[str] = None,
    visualize: bool = False,
    score_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Run inference on all images in a directory.

    Args:
        model: Trained model
        image_dir: Directory containing input images
        checkpoint_info: Dict with class names from checkpoint
        config: Configuration object
        output_dir: Directory to save visualizations (required if visualize=True)
        visualize: Whether to generate and save visualizations
        score_threshold: Detection score threshold (overrides config if provided)
        nms_threshold: NMS IoU threshold (overrides config if provided)

    Returns:
        List of result dictionaries, one per image
    """
    # Get class mappings from checkpoint
    det_class_names = checkpoint_info['det_class_names']

    # Find all images in the input directory
    image_files = get_image_files(image_dir)
    if not image_files:
        print(f"No images found in {image_dir}")
        return []

    print(f"Found {len(image_files)} images in {image_dir}")

    if visualize:
        if output_dir is None:
            raise ValueError("output_dir is required when visualize=True")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Visualizations will be saved to: {output_dir}")

    results = []

    # Process each image
    for i, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path)
        print(f"Processing [{i+1}/{len(image_files)}]: {image_name}")

        image = cv2.imread(image_path)
        # Run inference on single image
        result = infer_image(
            model=model,
            image=image,
            checkpoint_info=checkpoint_info,
            config=config,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold
        )

        # Add metadata
        result['image_path'] = image_path
        result['image_name'] = image_name
        results.append(result)

        # Visualize if requested
        if visualize:
            output_name = os.path.splitext(image_name)[0] + '_pred.png'
            output_path = os.path.join(output_dir, output_name)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
            visualize_predictions(
                image=image,
                seg_pred=result['segmentation']['mask'],
                det_pred=result['detection']['boxes'],
                det_labels=result['detection']['labels'],
                det_scores=result['detection']['scores'],
                det_class_names=det_class_names,
                cls_pred=result['classification']['class_name'],
                save_path=output_path
            )

    print(f"\nProcessed {len(image_files)} images")
    if visualize:
        print(f"Saved visualizations to: {output_dir}")

    return results


def main():
    """Main inference function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run inference on multi-task model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images for inference')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save output visualizations')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file (auto-detected from checkpoint dir if not specified)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate and save visualizations')
    parser.add_argument('--score_threshold', type=float, default=None,
                        help='Detection score threshold (default: from config)')
    parser.add_argument('--nms_threshold', type=float, default=None,
                        help='NMS IoU threshold (default: from config)')
    args = parser.parse_args()

    # Load config from checkpoint directory or use default
    checkpoint_dir = os.path.dirname(args.checkpoint)

    if args.config:
        config = Config.load(args.config)
    else:
        config_path = Config.find_in_directory(checkpoint_dir)
        if config_path:
            print(f"Found config in checkpoint directory: {config_path}")
            config = Config.load(config_path)
        else:
            print("No config found in checkpoint directory, using default config")
            config = Config()

    # Load model and get class info from checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model, checkpoint_info = load_model(args.checkpoint, config, config.DEVICE)
    print(f"Detection classes: {checkpoint_info['det_class_names']}")
    print(f"Classification classes: {checkpoint_info['cls_idx_to_class']}")

    # Run inference
    print(f"\nRunning inference on images in: {args.input_dir}")
    results = run_inference(
        model=model,
        image_dir=args.input_dir,
        checkpoint_info=checkpoint_info,
        config=config,
        output_dir=args.output_dir,
        visualize=args.visualize,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold
    )

    print(f"\nInference completed! Processed {len(results)} images.")


if __name__ == '__main__':
    main()
