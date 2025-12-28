"""
Dataset loader for classification task.
Loads images organized by score folders.
"""

import os
from typing import Dict, Optional, Any

import torch
from .base_dataset import BaseDataset


class ClassificationDataset(BaseDataset):
    """
    Dataset for classification task.

    Expected structure:
        classification/
            images/
                0/
                    image1.jpg
                    image2.jpg
                    ...
                0.25/
                    image1.jpg
                    ...
                0.5/
                    ...
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Any] = None,
        val_split: float = 0.2,
        random_seed: int = 42,
        image_size: int = 256
    ) -> None:
        """
        Args:
            data_root: Root directory containing 'classification' folder
            split: 'train' or 'val' (default: 'train')
            transform: Optional transform to apply to images
            val_split: Fraction of data to use for validation (default: 0.2)
            random_seed: Random seed for reproducible splitting (default: 42)
            image_size: Target image size (default: 256)
        """
        # Initialize base class
        task_data_root = os.path.join(data_root, 'classification')
        super(ClassificationDataset, self).__init__(
            data_root=task_data_root,
            split=split,
            transform=transform,
            val_split=val_split,
            random_seed=random_seed,
            image_size=image_size
        )

        # Setup task-specific paths
        self.images_dir = os.path.join(self.data_root, 'images')

        # Get all label folders (treat folder names as string class labels)
        gt_label_folders = sorted([
            d for d in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, d))
        ])

        # Create class mapping from folder names to class indices
        unique_labels = sorted(set(gt_label_folders))
        self.class_to_idx = {
            label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_class = {
            idx: label for label, idx in self.class_to_idx.items()}
        self.num_classes = len(unique_labels)

        # Build list of (image_path, class_label_string, class_idx) pairs
        all_samples = []
        all_labels = []  # Track labels for stratified splitting
        for gt_label_folder in gt_label_folders:
            folder_path = os.path.join(self.images_dir, gt_label_folder)
            image_files = self._get_image_files(folder_path)

            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                class_idx = self.class_to_idx[gt_label_folder]
                all_samples.append((img_path, gt_label_folder, class_idx))
                all_labels.append(class_idx)

        # Split into train/val using stratified splitting to maintain class balance
        self.samples = self._split_data_stratified(all_samples, all_labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with image, class_label, class_label_str, and task_type
        """
        img_path, gt_label_str, class_idx = self.samples[idx]

        # Load image using base class method
        image = self._load_image(img_path)
        image = self.transform(image)

        # Convert class index to tensor
        class_label = torch.tensor(class_idx, dtype=torch.long)

        return {
            'image': image,
            'class_label': class_label,
            'class_label_str': gt_label_str,  # Keep string for reference
            'task_type': 'cls'
        }
