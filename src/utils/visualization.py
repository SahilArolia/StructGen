"""
Visualization Utilities for StructGAN

Tools for visualizing training progress, model outputs, and comparisons.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to a displayable numpy image.

    Args:
        tensor: Tensor of shape [C, H, W] or [B, C, H, W] with values in [-1, 1]

    Returns:
        Numpy array of shape [H, W, C] with values in [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch element

    # Move to CPU if needed
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2

    # Clamp values
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy [H, W, C]
    img = tensor.permute(1, 2, 0).numpy()

    # Convert to uint8
    img = (img * 255).astype(np.uint8)

    return img


def visualize_results(
    input_img: torch.Tensor,
    generated_img: torch.Tensor,
    target_img: Optional[torch.Tensor] = None,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Visualize input, generated, and optionally target images.

    Args:
        input_img: Input architectural floor plan tensor
        generated_img: Generated structural layout tensor
        target_img: Optional ground truth structural layout tensor
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
    """
    n_cols = 3 if target_img is not None else 2

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Input
    axes[0].imshow(tensor_to_image(input_img))
    axes[0].set_title('Input: Architectural Plan')
    axes[0].axis('off')

    # Generated
    axes[1].imshow(tensor_to_image(generated_img))
    axes[1].set_title('Generated: Structural Layout')
    axes[1].axis('off')

    # Target (if provided)
    if target_img is not None:
        axes[2].imshow(tensor_to_image(target_img))
        axes[2].set_title('Target: Ground Truth')
        axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()


def save_comparison_grid(
    inputs: List[torch.Tensor],
    generated: List[torch.Tensor],
    targets: Optional[List[torch.Tensor]] = None,
    save_path: str = "comparison_grid.png",
    max_samples: int = 8
) -> None:
    """
    Save a grid of comparison images.

    Args:
        inputs: List of input tensors
        generated: List of generated tensors
        targets: Optional list of target tensors
        save_path: Path to save the grid
        max_samples: Maximum number of samples to include
    """
    n_samples = min(len(inputs), max_samples)
    n_cols = 3 if targets is not None else 2

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(5 * n_cols, 5 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Input
        axes[i, 0].imshow(tensor_to_image(inputs[i]))
        if i == 0:
            axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        # Generated
        axes[i, 1].imshow(tensor_to_image(generated[i]))
        if i == 0:
            axes[i, 1].set_title('Generated')
        axes[i, 1].axis('off')

        # Target
        if targets is not None:
            axes[i, 2].imshow(tensor_to_image(targets[i]))
            if i == 0:
                axes[i, 2].set_title('Target')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison grid to {save_path}")


def plot_training_curves(
    losses: dict,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training loss curves.

    Args:
        losses: Dictionary with loss names as keys and lists of values
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, len(losses), figsize=(5 * len(losses), 4))

    if len(losses) == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, losses.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()


def create_overlay(
    arch_img: np.ndarray,
    struct_img: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create an overlay of architectural and structural layouts.

    Args:
        arch_img: Architectural floor plan image
        struct_img: Structural layout image
        alpha: Blending factor (0 = only arch, 1 = only struct)

    Returns:
        Blended overlay image
    """
    # Ensure same size
    if arch_img.shape != struct_img.shape:
        struct_img = np.array(Image.fromarray(struct_img).resize(
            (arch_img.shape[1], arch_img.shape[0])
        ))

    # Create overlay
    overlay = ((1 - alpha) * arch_img + alpha * struct_img).astype(np.uint8)

    return overlay


def visualize_structural_elements(
    struct_img: Union[torch.Tensor, np.ndarray],
    colors: dict = None,
    save_path: Optional[str] = None
) -> dict:
    """
    Analyze and visualize structural elements in the output.

    Args:
        struct_img: Structural layout image
        colors: Dictionary mapping element types to RGB colors
        save_path: Path to save the visualization

    Returns:
        Dictionary with element statistics
    """
    if isinstance(struct_img, torch.Tensor):
        struct_img = tensor_to_image(struct_img)

    if colors is None:
        colors = {
            'shear_wall': (255, 0, 0),
            'column': (0, 0, 255),
            'background': (255, 255, 255)
        }

    stats = {}
    h, w, _ = struct_img.shape
    total_pixels = h * w

    fig, axes = plt.subplots(1, len(colors), figsize=(5 * len(colors), 5))

    for i, (name, color) in enumerate(colors.items()):
        # Create mask for this color
        tolerance = 30
        lower = np.array([max(0, c - tolerance) for c in color])
        upper = np.array([min(255, c + tolerance) for c in color])

        mask = np.all((struct_img >= lower) & (struct_img <= upper), axis=2)

        # Calculate statistics
        pixel_count = np.sum(mask)
        coverage = pixel_count / total_pixels * 100

        stats[name] = {
            'pixel_count': int(pixel_count),
            'coverage_percent': float(coverage)
        }

        # Visualize mask
        axes[i].imshow(mask, cmap='gray')
        axes[i].set_title(f'{name}\n({coverage:.1f}% coverage)')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()

    return stats
