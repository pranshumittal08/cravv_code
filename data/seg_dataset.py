"""
Dataset loader for segmentation task.
Loads RGB images and corresponding masks.
"""

import os
import random
from typing import Dict, Any, Optional, Tuple

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from .base_dataset import BaseDataset


class SegmentationDataset(BaseDataset):
    """
    Dataset for segmentation task.

    Expected structure:
        segmentation/
            images/
                image1.jpg
                image2.jpg
                ...
            masks/
                image1.png
                image2.png
                ...
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
            data_root: Root directory containing 'segmentation' folder
            split: 'train' or 'val' (default: 'train')
            transform: Optional transform to apply to images (ignored, uses custom paired transforms)
            val_split: Fraction of data to use for validation (default: 0.2)
            random_seed: Random seed for reproducible splitting (default: 42)
            image_size: Target image size (default: 256)
        """
        # Initialize base class (we'll override transforms for segmentation)
        task_data_root = os.path.join(data_root, 'segmentation')
        super(SegmentationDataset, self).__init__(
            data_root=task_data_root,
            split=split,
            transform=transform,
            val_split=val_split,
            random_seed=random_seed,
            image_size=image_size
        )

        # Setup task-specific paths
        self.images_dir = os.path.join(self.data_root, 'images')
        self.masks_dir = os.path.join(self.data_root, 'masks')

        # Get list of image files and split
        all_image_files = self._get_image_files(self.images_dir)
        self.image_files = self._split_data(all_image_files)

        # Augmentation settings
        self.flip_prob = 0.5 if split == 'train' else 0.0
        self.use_color_jitter = (split == 'train')

        # Color jitter (only for images, not masks)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        )

        # Normalization (only for images)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def _apply_paired_transforms(
        self,
        image: Image.Image,
        mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms to both image and mask with spatial consistency.
        Geometric transforms are applied to both; color transforms only to image.
        """
        # 1. Resize both to configured image size
        size = (self.image_size, self.image_size)
        image = TF.resize(image, size)
        mask = TF.resize(mask, size,
                         interpolation=TF.InterpolationMode.NEAREST)

        # 2. Random horizontal flip (same decision for both)
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 3. Color jitter (image only)
        if self.use_color_jitter:
            image = self.color_jitter(image)

        # 4. Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # 5. Normalize image only
        image = self.normalize(image)

        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with image, mask, and task_type
        """
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load mask (try different extensions)
        mask_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(self.masks_dir, mask_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break

        if mask_path is None:
            return self.__getitem__(random.randint(0, len(self)-1))

        # Load images
        image = self._load_image(img_path)
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        # Apply paired transforms (flip applied to both consistently)
        image, mask = self._apply_paired_transforms(image, mask)

        # Convert mask: remove channel dim, binarize, convert to long
        mask = (mask > 0).long().squeeze(0)

        return {
            'image': image,
            'mask': mask,
            'task_type': 'seg'
        }
