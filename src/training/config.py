"""
Training Configuration for StructGAN

Configurable parameters for training pix2pix/pix2pixHD models.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class Config:
    """Training configuration dataclass."""

    # Experiment
    name: str = "structgan_baseline"
    seed: int = 42

    # Data
    dataroot: str = "./data/structgan_original"
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4

    # Model Architecture
    input_nc: int = 3  # Input channels
    output_nc: int = 3  # Output channels
    ngf: int = 64  # Generator filters
    ndf: int = 64  # Discriminator filters
    netG: str = "global"  # Generator type: 'global' or 'local'
    netD: str = "multiscale"  # Discriminator type
    n_downsample_global: int = 4  # Downsampling layers in global generator
    n_blocks_global: int = 9  # ResNet blocks in global generator
    n_local_enhancers: int = 1  # Local enhancer networks
    n_blocks_local: int = 3  # ResNet blocks in local enhancer
    norm: str = "instance"  # Normalization: 'instance', 'batch', 'none'

    # Discriminator
    num_D: int = 3  # Number of discriminators (multi-scale)
    n_layers_D: int = 3  # Layers in each discriminator
    no_ganFeat_loss: bool = False  # Disable feature matching loss
    no_vgg_loss: bool = False  # Disable VGG perceptual loss

    # Training
    niter: int = 100  # Epochs with initial learning rate
    niter_decay: int = 100  # Epochs with decaying learning rate
    lr: float = 0.0002  # Initial learning rate
    beta1: float = 0.5  # Adam beta1
    beta2: float = 0.999  # Adam beta2

    # Loss weights
    lambda_feat: float = 10.0  # Feature matching loss weight
    lambda_vgg: float = 10.0  # VGG perceptual loss weight
    lambda_l1: float = 100.0  # L1 reconstruction loss weight (optional)

    # Checkpoints
    checkpoints_dir: str = "./models/checkpoints"
    save_epoch_freq: int = 10  # Save every N epochs
    save_latest_freq: int = 1000  # Save latest every N iterations

    # Logging
    print_freq: int = 100  # Print loss every N iterations
    display_freq: int = 500  # Display images every N iterations
    tensorboard: bool = True

    # Hardware
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    fp16: bool = False  # Use mixed precision

    # Resume
    continue_train: bool = False
    which_epoch: str = "latest"

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.image_size in [256, 512, 1024], "Image size must be 256, 512, or 1024"
        assert self.netG in ["global", "local"], "Generator must be 'global' or 'local'"
        assert self.norm in ["instance", "batch", "none"], "Invalid normalization"

    @property
    def experiment_dir(self) -> Path:
        """Get the experiment directory path."""
        return Path(self.checkpoints_dir) / self.name

    def save(self, path: Optional[str] = None):
        """Save configuration to YAML file."""
        import yaml

        if path is None:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            path = self.experiment_dir / "config.yaml"

        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        print(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        import yaml

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)


def get_config(preset: str = "baseline") -> Config:
    """
    Get a preset configuration.

    Args:
        preset: Configuration preset name

    Returns:
        Config object
    """
    if preset == "baseline":
        return Config(
            name="structgan_baseline",
            image_size=256,
            batch_size=4,
            netG="global",
            niter=100,
            niter_decay=100
        )

    elif preset == "high_res":
        return Config(
            name="structgan_highres",
            image_size=512,
            batch_size=2,
            netG="local",
            n_local_enhancers=1,
            niter=100,
            niter_decay=100
        )

    elif preset == "fast":
        return Config(
            name="structgan_fast",
            image_size=256,
            batch_size=8,
            netG="global",
            ngf=32,
            niter=50,
            niter_decay=50
        )

    elif preset == "debug":
        return Config(
            name="structgan_debug",
            image_size=256,
            batch_size=2,
            netG="global",
            niter=2,
            niter_decay=2,
            print_freq=10,
            save_epoch_freq=1
        )

    else:
        raise ValueError(f"Unknown preset: {preset}")


# Default configuration file content
DEFAULT_CONFIG_YAML = """
# StructGAN Training Configuration
# ================================

# Experiment Settings
name: structgan_baseline
seed: 42

# Data Settings
dataroot: ./data/structgan_original
image_size: 256
batch_size: 4
num_workers: 4

# Generator Architecture
input_nc: 3
output_nc: 3
ngf: 64
netG: global  # 'global' for 256px, 'local' for 512px+
n_downsample_global: 4
n_blocks_global: 9
norm: instance

# Discriminator Architecture
ndf: 64
netD: multiscale
num_D: 3
n_layers_D: 3

# Training Schedule
niter: 100  # Epochs at initial LR
niter_decay: 100  # Epochs with LR decay
lr: 0.0002
beta1: 0.5
beta2: 0.999

# Loss Weights
lambda_feat: 10.0  # Feature matching
lambda_vgg: 10.0   # VGG perceptual
lambda_l1: 100.0   # L1 reconstruction

# Checkpoints
checkpoints_dir: ./models/checkpoints
save_epoch_freq: 10
save_latest_freq: 1000

# Logging
print_freq: 100
display_freq: 500
tensorboard: true

# Hardware
gpu_ids: [0]
fp16: false

# Resume Training
continue_train: false
which_epoch: latest
"""


def create_default_config(path: str = "config.yaml") -> None:
    """Create a default configuration file."""
    with open(path, 'w') as f:
        f.write(DEFAULT_CONFIG_YAML)
    print(f"Created default configuration at {path}")


if __name__ == "__main__":
    # Create default config when run directly
    create_default_config()
