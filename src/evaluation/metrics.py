"""
Evaluation Metrics for StructGAN

Metrics for evaluating structural layout generation quality:
- Pixel Accuracy (PA)
- Intersection over Union (IoU)
- Wall-specific IoU (WIoU)
- Structural element IoU (SIoU)
- FID Score
"""

import time
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for metric computation."""
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)

    return tensor.numpy()


def get_structural_mask(
    img: np.ndarray,
    element_type: str = "wall",
    threshold: float = 0.3
) -> np.ndarray:
    """
    Extract binary mask for structural elements.

    Args:
        img: Image array of shape [C, H, W] or [H, W, C] with values in [0, 1]
        element_type: Type of element ('wall', 'column', 'all')
        threshold: Threshold for binary mask

    Returns:
        Binary mask of shape [H, W]
    """
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        # [C, H, W] -> [H, W, C]
        img = np.transpose(img, (1, 2, 0))

    if element_type == "wall":
        # Walls are typically red in StructGAN output
        # Red channel high, green and blue low
        if img.shape[-1] == 3:
            mask = (img[:, :, 0] > threshold) & (img[:, :, 1] < 0.5) & (img[:, :, 2] < 0.5)
        else:
            mask = img.squeeze() > threshold

    elif element_type == "column":
        # Columns are typically blue
        if img.shape[-1] == 3:
            mask = (img[:, :, 2] > threshold) & (img[:, :, 0] < 0.5) & (img[:, :, 1] < 0.5)
        else:
            mask = img.squeeze() > threshold

    elif element_type == "all":
        # All structural elements (not white/background)
        if img.shape[-1] == 3:
            # Non-white pixels
            mask = ~((img[:, :, 0] > 0.9) & (img[:, :, 1] > 0.9) & (img[:, :, 2] > 0.9))
        else:
            mask = img.squeeze() > threshold

    else:
        raise ValueError(f"Unknown element type: {element_type}")

    return mask.astype(np.float32)


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute pixel accuracy.

    PA = (TP + TN) / Total Pixels

    Args:
        pred: Predicted tensor [B, C, H, W] or [C, H, W]
        target: Target tensor [B, C, H, W] or [C, H, W]
        threshold: Threshold for binary conversion

    Returns:
        Pixel accuracy as float
    """
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)

    # Convert to grayscale if RGB
    if pred_np.ndim == 4:
        pred_gray = np.mean(pred_np, axis=1)
        target_gray = np.mean(target_np, axis=1)
    elif pred_np.ndim == 3 and pred_np.shape[0] == 3:
        pred_gray = np.mean(pred_np, axis=0)
        target_gray = np.mean(target_np, axis=0)
    else:
        pred_gray = pred_np
        target_gray = target_np

    # Binarize
    pred_binary = (pred_gray > threshold).astype(np.float32)
    target_binary = (target_gray > threshold).astype(np.float32)

    # Compute accuracy
    correct = np.sum(pred_binary == target_binary)
    total = pred_binary.size

    return correct / total


def compute_iou(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Compute Intersection over Union.

    IoU = Intersection / Union

    Args:
        pred_mask: Predicted binary mask
        target_mask: Target binary mask
        eps: Small epsilon for numerical stability

    Returns:
        IoU score
    """
    intersection = np.sum(pred_mask * target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask) - intersection

    return (intersection + eps) / (union + eps)


def wall_iou(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Compute IoU specifically for wall elements.

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        Wall IoU score
    """
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)

    pred_mask = get_structural_mask(pred_np, element_type="wall")
    target_mask = get_structural_mask(target_np, element_type="wall")

    return compute_iou(pred_mask, target_mask)


def structural_iou(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Compute IoU for all structural elements.

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        Structural IoU score
    """
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)

    pred_mask = get_structural_mask(pred_np, element_type="all")
    target_mask = get_structural_mask(target_np, element_type="all")

    return compute_iou(pred_mask, target_mask)


def column_iou(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Compute IoU specifically for column elements.

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        Column IoU score
    """
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)

    pred_mask = get_structural_mask(pred_np, element_type="column")
    target_mask = get_structural_mask(target_np, element_type="column")

    return compute_iou(pred_mask, target_mask)


def shear_wall_ratio(img: np.ndarray) -> float:
    """
    Compute shear wall ratio (wall area / total area).

    Args:
        img: Image array

    Returns:
        Shear wall ratio
    """
    if isinstance(img, torch.Tensor):
        img = tensor_to_numpy(img)

    wall_mask = get_structural_mask(img, element_type="wall")
    return np.sum(wall_mask) / wall_mask.size


def wall_connectivity(img: np.ndarray) -> float:
    """
    Measure wall connectivity (penalize disconnected wall segments).

    Args:
        img: Image array

    Returns:
        Connectivity score (1.0 = fully connected, lower = more fragments)
    """
    if isinstance(img, torch.Tensor):
        img = tensor_to_numpy(img)

    wall_mask = get_structural_mask(img, element_type="wall")

    if np.sum(wall_mask) == 0:
        return 0.0

    # Label connected components
    labeled, num_features = ndimage.label(wall_mask)

    # Ideal: 1 connected component
    # Score decreases with more components
    return 1.0 / num_features


def boundary_alignment(
    pred: torch.Tensor,
    input_img: torch.Tensor
) -> float:
    """
    Measure how well structural elements align with architectural boundaries.

    Args:
        pred: Predicted structural layout
        input_img: Input architectural floor plan

    Returns:
        Alignment score
    """
    pred_np = tensor_to_numpy(pred)
    input_np = tensor_to_numpy(input_img)

    # Extract boundaries from input (walls in architectural plan)
    if input_np.ndim == 3 and input_np.shape[0] == 3:
        # Find dark pixels (walls) in input
        input_gray = np.mean(input_np, axis=0)
    else:
        input_gray = input_np.squeeze()

    arch_walls = input_gray < 0.3  # Dark pixels are walls

    # Get structural walls
    struct_walls = get_structural_mask(pred_np, element_type="wall")

    # Measure overlap
    overlap = np.sum(arch_walls & struct_walls)
    total_struct = np.sum(struct_walls)

    if total_struct == 0:
        return 0.0

    return overlap / total_struct


def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    input_img: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Evaluate a batch of predictions.

    Args:
        pred: Predicted batch [B, C, H, W]
        target: Target batch [B, C, H, W]
        input_img: Optional input batch [B, C, H, W]

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'pixel_accuracy': pixel_accuracy(pred, target),
        'wall_iou': wall_iou(pred, target),
        'structural_iou': structural_iou(pred, target),
        'column_iou': column_iou(pred, target)
    }

    if input_img is not None:
        metrics['boundary_alignment'] = boundary_alignment(pred, input_img)

    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate a model on a full dataset.

    Args:
        model: The generator model
        dataloader: DataLoader for evaluation data
        device: Device to run on

    Returns:
        Dictionary of averaged metrics
    """
    model.eval()

    all_metrics = {
        'pixel_accuracy': [],
        'wall_iou': [],
        'structural_iou': [],
        'column_iou': [],
        'boundary_alignment': [],
        'generation_time': []
    }

    with torch.no_grad():
        for batch in dataloader:
            input_img, target_img = batch
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            # Generate
            start_time = time.time()
            generated = model(input_img)
            gen_time = time.time() - start_time

            # Compute metrics
            batch_metrics = evaluate_batch(generated, target_img, input_img)

            for key, value in batch_metrics.items():
                all_metrics[key].append(value)

            all_metrics['generation_time'].append(gen_time / input_img.size(0))

    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    return avg_metrics


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Print metrics in a formatted way."""
    print(f"\n{prefix}Evaluation Metrics:")
    print("-" * 40)
    print(f"  Pixel Accuracy:     {metrics.get('pixel_accuracy', 0):.4f}")
    print(f"  Wall IoU:           {metrics.get('wall_iou', 0):.4f}")
    print(f"  Structural IoU:     {metrics.get('structural_iou', 0):.4f}")
    print(f"  Column IoU:         {metrics.get('column_iou', 0):.4f}")

    if 'boundary_alignment' in metrics:
        print(f"  Boundary Alignment: {metrics.get('boundary_alignment', 0):.4f}")

    if 'generation_time' in metrics:
        print(f"  Gen Time (avg):     {metrics.get('generation_time', 0)*1000:.1f} ms")

    print("-" * 40)
