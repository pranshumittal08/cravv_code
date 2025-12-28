"""
Dataset loader for detection task (COCO format).
Loads images and bounding box annotations from COCO JSON.
"""

import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from .base_dataset import BaseDataset


class DetectionDataset(BaseDataset):
    """
    Dataset for detection task using COCO format annotations.

    Expected structure:
        detection/
            images/
                image1.jpg
                image2.jpg
                ...
            _annotations.coco.json
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Any] = None,
        val_split: float = 0.2,
        random_seed: int = 42,
        image_size: int = 256,
    ) -> None:
        """
        Args:
            data_root: Root directory containing 'detection' folder
            split: 'train' or 'val' (default: 'train')
            transform: Optional transform to apply to images
            val_split: Fraction of data to use for validation (default: 0.2)
            random_seed: Random seed for reproducible splitting (default: 42)
            image_size: Target image size (default: 256)
        """
        # Initialize base class with transform=None (we handle transforms ourselves)
        task_data_root = os.path.join(data_root, 'detection')
        super(DetectionDataset, self).__init__(
            data_root=task_data_root,
            split=split,
            transform=None,  # We'll handle transforms in __getitem__
            val_split=val_split,
            random_seed=random_seed,
            image_size=image_size
        )

        # Store flip probability for training
        self.flip_prob = 0.5 if split == 'train' else 0.0

        # Color jitter for training
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        ) if split == 'train' else None

        # Setup task-specific paths
        self.images_dir = os.path.join(self.data_root, 'images')
        self.annotations_file = os.path.join(
            self.data_root, '_annotations.coco.json')

        # Load COCO annotations
        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)

        # Create mappings
        self.image_id_to_info = {
            img['id']: img for img in self.coco_data['images']}
        self.category_id_to_name = {cat['id']: cat['name']
                                    for cat in self.coco_data['categories']}
        self.category_name_to_id = {cat['name']: cat['id']
                                    for cat in self.coco_data['categories']}

        # Extract class names from COCO categories (sorted by category ID for consistency)
        # Filter out generic categories like 'objects' if present
        categories_sorted = sorted(
            self.coco_data['categories'], key=lambda x: x['id'])
        self.class_names = [cat['name'] for cat in categories_sorted
                            if cat['name'].lower() != 'objects']
        self.name_to_class_idx = {
            name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)

        # Get list of image IDs and split into train/val
        all_image_ids = sorted(list(self.image_id_to_info.keys()))
        self.image_ids = self._split_data(all_image_ids)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _apply_paired_transforms(
        self,
        image,
        boxes: List[List[float]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms to both image and bounding boxes consistently.

        Args:
            image: PIL Image
            boxes: List of [cx, cy, w, h] in normalized coordinates

        Returns:
            image: Transformed tensor
            boxes: Transformed boxes tensor
        """
        # 1. Resize image
        image = TF.resize(image, (self.image_size, self.image_size))

        # 2. Random horizontal flip - same decision for both image and boxes
        do_flip = self.split == 'train' and random.random() < self.flip_prob
        if do_flip:
            image = TF.hflip(image)
            # Flip boxes: for center format [cx, cy, w, h], new cx = 1 - cx
            boxes = [[1.0 - cx, cy, w, h] for cx, cy, w, h in boxes]

        # 3. Color jitter (image only, doesn't affect boxes)
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        # 4. Convert image to tensor
        image = TF.to_tensor(image)

        # 5. Normalize image
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        # 6. Convert boxes to tensor
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)

        return image, boxes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with image, boxes, labels, and task_type
        """
        img_id = self.image_ids[idx]
        img_info = self.image_id_to_info[img_id]

        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = self._load_image(img_path)

        # Get original size
        orig_width, orig_height = image.size

        # Get annotations for this image
        annotations = self.image_annotations.get(img_id, [])

        boxes = []
        labels = []

        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Normalize to [0, 1] based on original image size
            x_norm = x / orig_width
            y_norm = y / orig_height
            w_norm = w / orig_width
            h_norm = h / orig_height

            # Convert to center format: [center_x, center_y, width, height]
            center_x = x_norm + w_norm / 2
            center_y = y_norm + h_norm / 2

            # Get class index
            category_name = self.category_id_to_name[ann['category_id']]
            if category_name in self.name_to_class_idx:
                class_idx = self.name_to_class_idx[category_name]
                boxes.append([center_x, center_y, w_norm, h_norm])
                labels.append(class_idx)

        # Handle images with no annotations
        if len(boxes) == 0:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Apply paired transforms to image and boxes
        image, boxes = self._apply_paired_transforms(image, boxes)

        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'task_type': 'det'
        }
