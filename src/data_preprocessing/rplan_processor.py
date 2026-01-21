"""
RPLAN Dataset Processor

Converts RPLAN floor plan dataset to StructGAN-compatible format.
RPLAN contains 80,000+ residential floor plans with semantic annotations.

Reference:
- Wu et al. (2019) "Data-driven Interior Plan Generation for Residential Buildings"
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


class RPLANProcessor:
    """
    Process RPLAN dataset for use with StructGAN.

    RPLAN Format:
    - 256x256 PNG images with semantic color coding
    - Each room type has a specific RGB color
    - Walls, doors, and windows are annotated

    Output Format:
    - Paired images: architectural input -> structural target
    - Side-by-side format compatible with pix2pix/pix2pixHD
    """

    # RPLAN semantic color mapping (RGB)
    RPLAN_COLORS = {
        'living_room': (244, 242, 229),      # Beige
        'master_room': (224, 207, 215),      # Light pink
        'kitchen': (208, 216, 135),          # Yellow-green
        'bathroom': (190, 220, 222),         # Cyan
        'dining_room': (253, 232, 173),      # Light orange
        'child_room': (224, 207, 215),       # Light pink
        'study_room': (239, 226, 200),       # Tan
        'second_room': (224, 207, 215),      # Light pink
        'guest_room': (224, 207, 215),       # Light pink
        'balcony': (249, 222, 189),          # Peach
        'entrance': (239, 233, 221),         # Cream
        'storage': (225, 225, 225),          # Light gray
        'wall': (0, 0, 0),                   # Black
        'external_area': (255, 255, 255),    # White
        'exterior_wall': (120, 120, 120),    # Dark gray
        'front_door': (255, 0, 0),           # Red
        'interior_door': (255, 100, 100),    # Light red
        'window': (0, 0, 255),               # Blue
    }

    # StructGAN output colors for structural elements
    STRUCTURAL_COLORS = {
        'shear_wall': (255, 0, 0),           # Red
        'column': (0, 0, 255),               # Blue
        'beam': (0, 255, 0),                 # Green
        'background': (255, 255, 255),       # White
    }

    def __init__(
        self,
        rplan_dir: str,
        output_dir: str,
        image_size: int = 256
    ):
        """
        Args:
            rplan_dir: Path to RPLAN dataset
            output_dir: Path to output directory
            image_size: Output image size
        """
        self.rplan_dir = Path(rplan_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

    def process_all(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        max_samples: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Process all RPLAN images.

        Args:
            train_ratio: Ratio of training samples
            val_ratio: Ratio of validation samples (test = 1 - train - val)
            max_samples: Maximum number of samples to process (None for all)

        Returns:
            Dictionary with counts per split
        """
        # Find all floor plan images
        image_paths = list(self.rplan_dir.glob('*.png'))

        if max_samples:
            image_paths = image_paths[:max_samples]

        print(f"Found {len(image_paths)} RPLAN images")

        # Shuffle for random split
        np.random.seed(42)
        np.random.shuffle(image_paths)

        # Calculate split indices
        n_total = len(image_paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            'train': image_paths[:n_train],
            'val': image_paths[n_train:n_train + n_val],
            'test': image_paths[n_train + n_val:]
        }

        counts = {}

        for split_name, paths in splits.items():
            print(f"\nProcessing {split_name} split ({len(paths)} images)...")
            count = 0

            for img_path in tqdm(paths, desc=split_name):
                try:
                    paired_img = self.process_single(img_path)
                    if paired_img is not None:
                        output_path = self.output_dir / split_name / f"{img_path.stem}.png"
                        paired_img.save(output_path)
                        count += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            counts[split_name] = count

        return counts

    def process_single(self, image_path: Path) -> Optional[Image.Image]:
        """
        Process a single RPLAN image to create a paired image.

        Args:
            image_path: Path to RPLAN floor plan image

        Returns:
            PIL Image with input|target side by side, or None if processing fails
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract architectural layout
        arch_layout = self.extract_architectural(img)

        # Generate structural layout (rule-based)
        struct_layout = self.generate_structural(img)

        # Resize to target size
        arch_layout = cv2.resize(arch_layout, (self.image_size, self.image_size))
        struct_layout = cv2.resize(struct_layout, (self.image_size, self.image_size))

        # Create side-by-side image
        combined = np.concatenate([arch_layout, struct_layout], axis=1)

        return Image.fromarray(combined)

    def extract_architectural(self, img: np.ndarray) -> np.ndarray:
        """
        Extract architectural floor plan representation.
        Simplifies RPLAN colors to a cleaner architectural view.
        """
        h, w, _ = img.shape
        output = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Create masks for different elements
        wall_mask = self._get_color_mask(img, self.RPLAN_COLORS['wall'])
        ext_wall_mask = self._get_color_mask(img, self.RPLAN_COLORS['exterior_wall'])
        door_mask = self._get_color_mask(img, self.RPLAN_COLORS['front_door']) | \
                    self._get_color_mask(img, self.RPLAN_COLORS['interior_door'])

        # Draw rooms (simplified - all same color)
        room_colors = ['living_room', 'master_room', 'kitchen', 'bathroom',
                       'dining_room', 'child_room', 'study_room', 'second_room',
                       'guest_room', 'balcony', 'entrance', 'storage']

        room_mask = np.zeros((h, w), dtype=bool)
        for room_type in room_colors:
            if room_type in self.RPLAN_COLORS:
                room_mask |= self._get_color_mask(img, self.RPLAN_COLORS[room_type])

        # Fill rooms with light gray
        output[room_mask] = (240, 240, 240)

        # Draw walls (black)
        output[wall_mask | ext_wall_mask] = (0, 0, 0)

        # Draw doors (brown)
        output[door_mask] = (139, 69, 19)

        return output

    def generate_structural(self, img: np.ndarray) -> np.ndarray:
        """
        Generate structural layout using rule-based approach.

        Rules based on structural engineering principles:
        1. Exterior walls become shear walls
        2. Long interior walls may need columns at ends
        3. Corners of exterior need columns
        4. Large span areas need column support
        """
        h, w, _ = img.shape
        output = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Get wall masks
        wall_mask = self._get_color_mask(img, self.RPLAN_COLORS['wall'])
        ext_wall_mask = self._get_color_mask(img, self.RPLAN_COLORS['exterior_wall'])

        # Rule 1: Exterior walls become shear walls (red)
        if ext_wall_mask.any():
            # Dilate exterior wall slightly for visibility
            kernel = np.ones((3, 3), np.uint8)
            ext_wall_thick = cv2.dilate(ext_wall_mask.astype(np.uint8), kernel, iterations=1)
            output[ext_wall_thick > 0] = self.STRUCTURAL_COLORS['shear_wall']

        # Rule 2: Find corners and add columns (blue)
        combined_wall = wall_mask | ext_wall_mask
        corners = self._find_corners(combined_wall)

        for (cx, cy) in corners:
            # Draw column as small square
            col_size = max(5, h // 50)
            y1, y2 = max(0, cy - col_size), min(h, cy + col_size)
            x1, x2 = max(0, cx - col_size), min(w, cx + col_size)
            output[y1:y2, x1:x2] = self.STRUCTURAL_COLORS['column']

        # Rule 3: Interior shear walls at strategic locations
        interior_mask = wall_mask & ~ext_wall_mask
        if interior_mask.any():
            # Find long interior walls
            contours, _ = cv2.findContours(
                interior_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Check if wall is long enough
                area = cv2.contourArea(contour)
                if area > (h * w * 0.005):  # Significant wall
                    cv2.drawContours(
                        output, [contour], -1,
                        self.STRUCTURAL_COLORS['shear_wall'], -1
                    )

        return output

    def _get_color_mask(
        self,
        img: np.ndarray,
        color: Tuple[int, int, int],
        tolerance: int = 10
    ) -> np.ndarray:
        """Create a boolean mask for pixels matching a specific color."""
        lower = np.array([max(0, c - tolerance) for c in color])
        upper = np.array([min(255, c + tolerance) for c in color])
        mask = cv2.inRange(img, lower, upper)
        return mask > 0

    def _find_corners(self, wall_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find corner points in wall layout."""
        corners = []

        # Use Harris corner detection
        wall_img = wall_mask.astype(np.uint8) * 255

        corners_detected = cv2.cornerHarris(wall_img, 9, 3, 0.04)
        corners_detected = cv2.dilate(corners_detected, None)

        # Threshold for corner detection
        threshold = 0.01 * corners_detected.max()
        corner_points = np.where(corners_detected > threshold)

        # Convert to list of (x, y) tuples and filter duplicates
        seen = set()
        grid_size = max(10, wall_mask.shape[0] // 20)

        for y, x in zip(corner_points[0], corner_points[1]):
            grid_key = (x // grid_size, y // grid_size)
            if grid_key not in seen:
                seen.add(grid_key)
                corners.append((x, y))

        return corners


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentation_factor: int = 4
) -> int:
    """
    Augment a dataset with rotations and flips.

    Args:
        input_dir: Directory containing paired images
        output_dir: Output directory for augmented images
        augmentation_factor: How many augmented versions per image

    Returns:
        Total number of augmented images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_path.glob('*.png'))
    count = 0

    for img_path in tqdm(image_paths, desc="Augmenting"):
        img = Image.open(img_path)
        w, h = img.size

        # Original
        img.save(output_path / f"{img_path.stem}_orig.png")
        count += 1

        # Rotations
        for angle in [90, 180, 270]:
            rotated = img.rotate(angle, expand=True)
            rotated.save(output_path / f"{img_path.stem}_rot{angle}.png")
            count += 1

        # Horizontal flip
        flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_h.save(output_path / f"{img_path.stem}_fliph.png")
        count += 1

        # Vertical flip
        flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_v.save(output_path / f"{img_path.stem}_flipv.png")
        count += 1

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process RPLAN dataset")
    parser.add_argument("--rplan_dir", type=str, required=True, help="RPLAN dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--image_size", type=int, default=256, help="Output image size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")

    args = parser.parse_args()

    processor = RPLANProcessor(
        rplan_dir=args.rplan_dir,
        output_dir=args.output_dir,
        image_size=args.image_size
    )

    counts = processor.process_all(max_samples=args.max_samples)

    print("\nProcessing complete!")
    print(f"Train: {counts.get('train', 0)}")
    print(f"Val: {counts.get('val', 0)}")
    print(f"Test: {counts.get('test', 0)}")
