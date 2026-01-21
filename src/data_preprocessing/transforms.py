"""
Data Transforms for StructGAN

Custom transforms that apply the same augmentation to both
input (architectural) and target (structural) images.
"""

import random
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class PairedTransform:
    """
    Apply identical transforms to paired images.
    Ensures input and target receive the same random augmentation.
    """

    def __init__(
        self,
        image_size: int = 256,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation: bool = True,
        rotation_angles: list = [0, 90, 180, 270],
        color_jitter: bool = False,
        normalize: bool = True
    ):
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.rotation_angles = rotation_angles
        self.color_jitter = color_jitter
        self.normalize = normalize

    def __call__(
        self,
        input_img: Image.Image,
        target_img: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transforms to paired images.

        Args:
            input_img: PIL Image of architectural floor plan
            target_img: PIL Image of structural layout

        Returns:
            Tuple of transformed tensors
        """
        # Resize
        input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        target_img = target_img.resize((self.image_size, self.image_size), Image.NEAREST)

        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)

        # Random vertical flip
        if self.vertical_flip and random.random() > 0.5:
            input_img = TF.vflip(input_img)
            target_img = TF.vflip(target_img)

        # Random rotation (90-degree increments)
        if self.rotation:
            angle = random.choice(self.rotation_angles)
            if angle != 0:
                input_img = TF.rotate(input_img, angle)
                target_img = TF.rotate(target_img, angle)

        # Color jitter (only for input, not target)
        if self.color_jitter:
            jitter = transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            )
            input_img = jitter(input_img)

        # Convert to tensor
        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        # Normalize to [-1, 1]
        if self.normalize:
            input_tensor = input_tensor * 2 - 1
            target_tensor = target_tensor * 2 - 1

        return input_tensor, target_tensor


class PairedTestTransform:
    """
    Transform for testing/inference - no augmentation, just resize and normalize.
    """

    def __init__(self, image_size: int = 256, normalize: bool = True):
        self.image_size = image_size
        self.normalize = normalize

    def __call__(
        self,
        input_img: Image.Image,
        target_img: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize
        input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        target_img = target_img.resize((self.image_size, self.image_size), Image.NEAREST)

        # Convert to tensor
        input_tensor = TF.to_tensor(input_img)
        target_tensor = TF.to_tensor(target_img)

        # Normalize
        if self.normalize:
            input_tensor = input_tensor * 2 - 1
            target_tensor = target_tensor * 2 - 1

        return input_tensor, target_tensor


def get_transforms(
    split: str = "train",
    image_size: int = 256,
    augmentation: bool = True
) -> PairedTransform:
    """
    Get appropriate transforms for a given split.

    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size
        augmentation: Whether to apply data augmentation

    Returns:
        Transform callable
    """
    if split == "train" and augmentation:
        return PairedTransform(
            image_size=image_size,
            horizontal_flip=True,
            vertical_flip=True,
            rotation=True,
            color_jitter=False  # Usually off for structural data
        )
    else:
        return PairedTestTransform(image_size=image_size)
