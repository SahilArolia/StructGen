"""
StructGAN Dataset Loader

Handles loading of paired architectural-structural images for training.
Compatible with both original StructGAN dataset and processed RPLAN data.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class StructGANDataset(Dataset):
    """
    Dataset for StructGAN training.

    Expects paired images where:
    - Input: Architectural floor plan
    - Target: Structural layout (shear walls, columns)

    Supports two formats:
    1. Side-by-side format: Single image with input|target concatenated horizontally
    2. Separate folders: input/ and target/ folders with matching filenames
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 256,
        transform: Optional[Callable] = None,
        paired_format: str = "side_by_side"  # or "separate"
    ):
        """
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size for resizing
            transform: Optional transforms to apply
            paired_format: 'side_by_side' or 'separate'
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.paired_format = paired_format

        self.image_paths = self._load_image_paths()
        print(f"Loaded {len(self.image_paths)} images for {split} split")

    def _load_image_paths(self) -> List[Path]:
        """Load all image paths from the dataset directory."""
        paths = []
        
        # Check for different directory formats
        split_dir = self.root_dir / self.split  # e.g., root/train
        split_dir_A = self.root_dir / f"{self.split}_A"  # e.g., root/train_A
        split_dir_B = self.root_dir / f"{self.split}_B"  # e.g., root/train_B
        
        if split_dir_A.exists() and split_dir_B.exists():
             # Pix2pix format detected
            search_dir = split_dir_A
            self.paired_format = "pix2pix"
            self.target_dir = split_dir_B
        elif split_dir.exists():
            search_dir = split_dir
        else:
            search_dir = self.root_dir
            
        # Support multiple image formats
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    
        for ext in extensions:
            paths.extend(search_dir.glob(ext))
    
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.paired_format == "side_by_side":
            return self._load_side_by_side(idx)
        elif self.paired_format == "pix2pix":
            return self._load_pix2pix(idx)
        else:
            return self._load_separate(idx)
    
    def _load_pix2pix(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load paired images from pix2pix format (train_A/train_B)."""
        input_path = self.image_paths[idx]
        target_path = self.target_dir / input_path.name
      
        if not target_path.exists():
            raise FileNotFoundError(f"Target image not found: {target_path}")
    
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
    
        input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        target_img = target_img.resize((self.image_size, self.image_size), Image.BILINEAR)
    
        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)
        else:
            to_tensor = transforms.ToTensor()
            input_img = to_tensor(input_img)
            target_img = to_tensor(target_img)
            input_img = input_img * 2 - 1
            target_img = target_img * 2 - 1
    
        return input_img, target_img

    def _load_separate(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load paired images from separate folders."""
        input_path = self.image_paths[idx]

        # Construct target path
        target_dir = input_path.parent.parent / "target"
        target_path = target_dir / input_path.name

        if not target_path.exists():
            raise FileNotFoundError(f"Target image not found: {target_path}")

        # Load images
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Resize
        input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        target_img = target_img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Apply transforms
        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)
        else:
            to_tensor = transforms.ToTensor()
            input_img = to_tensor(input_img)
            target_img = to_tensor(target_img)
            input_img = input_img * 2 - 1
            target_img = target_img * 2 - 1

        return input_img, target_img


def get_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 4,
    image_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    paired_format: str = "side_by_side"
) -> DataLoader:
    """
    Create a DataLoader for the StructGAN dataset.

    Args:
        root_dir: Root directory of the dataset
        split: Dataset split
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
        paired_format: Format of paired images

    Returns:
        DataLoader instance
    """
    dataset = StructGANDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
        paired_format=paired_format
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )


# Utility functions for dataset exploration
def count_images(root_dir: str) -> dict:
    """Count images in a dataset directory."""
    root = Path(root_dir)
    counts = {}

    for split in ['train', 'val', 'test']:
        split_dir = root / split
        if split_dir.exists():
            count = len(list(split_dir.glob('*.png'))) + len(list(split_dir.glob('*.jpg')))
            counts[split] = count

    # Count in root if no splits found
    if not counts:
        count = len(list(root.glob('*.png'))) + len(list(root.glob('*.jpg')))
        counts['all'] = count

    return counts


def visualize_sample(dataset: StructGANDataset, idx: int = 0, save_path: Optional[str] = None):
    """Visualize a sample from the dataset."""
    import matplotlib.pyplot as plt

    input_img, target_img = dataset[idx]

    # Convert from [-1, 1] to [0, 1]
    input_img = (input_img + 1) / 2
    target_img = (target_img + 1) / 2

    # Convert to numpy for plotting
    input_np = input_img.permute(1, 2, 0).numpy()
    target_np = target_img.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(input_np)
    axes[0].set_title('Input: Architectural Floor Plan')
    axes[0].axis('off')

    axes[1].imshow(target_np)
    axes[1].set_title('Target: Structural Layout')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()
