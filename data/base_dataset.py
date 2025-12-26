"""
Base dataset class with common functionality for all task-specific datasets.
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    """
    Base dataset class providing common functionality:
    - Train/val splitting
    - Transform setup
    - Image loading utilities
    """

    def __init__(self, data_root, split='train', transform=None, val_split=0.2,
                 random_seed=42, image_size=256):
        """
        Args:
            data_root: Root directory containing task-specific folder
            split: 'train' or 'val' (default: 'train')
            transform: Optional transform to apply to images
            val_split: Fraction of data to use for validation (default: 0.2)
            random_seed: Random seed for reproducible splitting (default: 42)
            image_size: Target image size (default: 256)
        """
        self.data_root = data_root
        self.split = split
        self.val_split = val_split
        self.random_seed = random_seed
        self.image_size = image_size

        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transforms(split)
        else:
            self.transform = transform

    def _get_default_transforms(self, split):
        """
        Get default transforms for train or validation.

        Args:
            split: 'train' or 'val'

        Returns:
            transform: torchvision transform
        """
        size = (self.image_size, self.image_size)

        if split == 'train':
            return transforms.Compose([
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
        else:  # 'val'
            return transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

    def _split_data(self, data_list):
        """
        Split data into train/val sets.

        Args:
            data_list: List of data items to split

        Returns:
            split_data: Data for the current split (train or val)
        """
        random.seed(self.random_seed)
        shuffled_data = data_list.copy()
        random.shuffle(shuffled_data)

        split_idx = int(len(shuffled_data) * (1 - self.val_split))

        if self.split == 'train':
            return shuffled_data[:split_idx]
        else:  # 'val'
            return shuffled_data[split_idx:]

    def _split_data_stratified(self, data_list, labels):
        """
        Split data into train/val sets with stratified sampling to maintain class balance.

        Args:
            data_list: List of data items to split
            labels: List of class labels corresponding to each item in data_list

        Returns:
            split_data: Data for the current split (train or val)
        """
        from collections import defaultdict

        random.seed(self.random_seed)

        # Group samples by class label
        class_samples = defaultdict(list)
        for item, label in zip(data_list, labels):
            class_samples[label].append(item)

        train_data = []
        val_data = []

        # Split each class proportionally
        for label in sorted(class_samples.keys()):
            samples = class_samples[label]
            random.shuffle(samples)

            # Calculate split point for this class
            split_idx = int(len(samples) * (1 - self.val_split))

            # Ensure at least 1 sample in each split if possible
            if split_idx == 0 and len(samples) > 1:
                split_idx = 1
            elif split_idx == len(samples) and len(samples) > 1:
                split_idx = len(samples) - 1

            train_data.extend(samples[:split_idx])
            val_data.extend(samples[split_idx:])

        # Shuffle the combined splits to mix classes
        random.shuffle(train_data)
        random.shuffle(val_data)

        if self.split == 'train':
            return train_data
        else:  # 'val'
            return val_data

    def _load_image(self, image_path):
        """
        Load and convert image to RGB.

        Args:
            image_path: Path to image file

        Returns:
            image: PIL Image in RGB format
        """
        return Image.open(image_path).convert('RGB')

    def _get_image_files(self, directory, extensions=('.jpg', '.jpeg', '.png')):
        """
        Get list of image files from directory.

        Args:
            directory: Directory to search
            extensions: Tuple of file extensions to include

        Returns:
            image_files: Sorted list of image file names
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        image_files = [
            f for f in os.listdir(directory)
            if f.lower().endswith(extensions)
        ]
        return sorted(image_files)

    def __len__(self):
        """Must be implemented by child classes."""
        raise NotImplementedError

    def __getitem__(self, idx):
        """Must be implemented by child classes."""
        raise NotImplementedError
