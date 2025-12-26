"""
Dataset loader for detection task (COCO format).
Loads images and bounding box annotations from COCO JSON.
"""

import os
import json
import random
import torch
from torchvision import transforms
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

    def __init__(self, data_root, split='train', transform=None, val_split=0.2,
                 random_seed=42, image_size=256):
        """
        Args:
            data_root: Root directory containing 'detection' folder
            split: 'train' or 'val' (default: 'train')
            transform: Optional transform to apply to images
            val_split: Fraction of data to use for validation (default: 0.2)
            random_seed: Random seed for reproducible splitting (default: 42)
            image_size: Target image size (default: 256)
        """
        # Initialize base class
        task_data_root = os.path.join(data_root, 'detection')
        super(DetectionDataset, self).__init__(
            data_root=task_data_root,
            split=split,
            transform=transform,
            val_split=val_split,
            random_seed=random_seed,
            image_size=image_size
        )

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

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.image_id_to_info[img_id]

        # Load image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = self._load_image(img_path)

        # Get original size
        orig_width, orig_height = image.size

        # Transform image
        image = self.transform(image)

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

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            # No objects in image - return empty tensors
            self.__getitem__(random.ranint(0, len(self)-1))

        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'task_type': 'det'
        }
