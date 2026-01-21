"""
Baseline Training Script for StructGAN

This script provides a simplified training loop for reproducing
the baseline StructGAN results using a pix2pix-style architecture.

For the full pix2pixHD training, use the original repository:
    cd StructGAN_v1/2_pix2pixHD_adopted
    python train.py --name structgan --dataroot ../0_datasets/Group7-H2
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_preprocessing.dataset import StructGANDataset, get_dataloader
from src.data_preprocessing.transforms import get_transforms
from src.evaluation.metrics import evaluate_batch, print_metrics
from src.utils.visualization import visualize_results, save_comparison_grid
from src.utils.helpers import set_seed, get_device, AverageMeter, ensure_dir
from src.training.config import Config, get_config


# =============================================================================
# Model Definitions (Simplified pix2pix for baseline)
# =============================================================================

class UNetDown(nn.Module):
    """Downsampling block for U-Net generator."""

    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block for U-Net generator."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    """
    U-Net Generator for pix2pix.

    Architecture:
        Encoder -> Bottleneck -> Decoder with skip connections
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Encoder (downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)

        # Decoder (upsampling with skip connections)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.

    Classifies 70x70 overlapping patches as real/fake.
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    optimizer_G: optim.Optimizer,
    optimizer_D: optim.Optimizer,
    criterion_GAN: nn.Module,
    criterion_pixelwise: nn.Module,
    device: torch.device,
    epoch: int,
    config: Config,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Train for one epoch."""

    generator.train()
    discriminator.train()

    loss_G_meter = AverageMeter("G_loss")
    loss_D_meter = AverageMeter("D_loss")
    loss_pixel_meter = AverageMeter("Pixel_loss")

    for i, (input_img, target_img) in enumerate(dataloader):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # 1. Generate fake image
        fake_img = generator(input_img)

        # 2. Get discriminator prediction on fake image
        # This fixes the NameError: we use 'fake_img' and 'input_img' which are defined
        pred_fake = discriminator(fake_img, input_img)

        # 3. Create ground truth labels dynamically
        # This fixes the Shape Mismatch: 'valid' and 'fake' automatically match pred_fake's shape (e.g., 15x15)
        valid = torch.ones_like(pred_fake, device=device)
        fake = torch.zeros_like(pred_fake, device=device)

        # 4. Calculate Generator Loss
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_pixel = criterion_pixelwise(fake_img, target_img)
        
        # Total generator loss
        loss_G = loss_GAN + config.lambda_l1 * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # 1. Real loss (Real Target + Input)
        pred_real = discriminator(target_img, input_img)
        loss_real = criterion_GAN(pred_real, valid)

        # 2. Fake loss (Generated Image + Input)
        # We must detach fake_img to avoid calculating gradients for the Generator here
        pred_fake_d = discriminator(fake_img.detach(), input_img)
        loss_fake = criterion_GAN(pred_fake_d, fake)

        # 3. Total discriminator loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # Update meters
        loss_G_meter.update(loss_G.item())
        loss_D_meter.update(loss_D.item())
        loss_pixel_meter.update(loss_pixel.item())

        # Logging
        global_step = epoch * len(dataloader) + i

        if i % config.print_freq == 0:
            print(f"  Epoch [{epoch}][{i}/{len(dataloader)}] "
                  f"G: {loss_G.item():.4f} D: {loss_D.item():.4f} "
                  f"Pixel: {loss_pixel.item():.4f}")

        if writer and i % config.display_freq == 0:
            writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
            writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)
            writer.add_scalar('Loss/Pixel', loss_pixel.item(), global_step)

    return {
        'loss_G': loss_G_meter.avg,
        'loss_D': loss_D_meter.avg,
        'loss_pixel': loss_pixel_meter.avg
    }


def validate(
    generator: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    generator.eval()

    all_metrics = []

    with torch.no_grad():
        for input_img, target_img in dataloader:
            input_img = input_img.to(device)
            target_img = target_img.to(device)

            fake_img = generator(input_img)

            metrics = evaluate_batch(fake_img, target_img, input_img)
            all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return avg_metrics


def train_baseline(config: Optional[Config] = None):
    """
    Main training function for baseline StructGAN.

    Args:
        config: Training configuration (uses default if None)
    """
    if config is None:
        config = get_config("baseline")

    print("\n" + "=" * 60)
    print("StructGAN Baseline Training")
    print("=" * 60)
    print(f"Experiment: {config.name}")
    print(f"Data: {config.dataroot}")
    print(f"Image size: {config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.niter + config.niter_decay}")
    print("=" * 60 + "\n")

    # Setup
    set_seed(config.seed)
    device = get_device()

    # Create directories
    experiment_dir = ensure_dir(config.experiment_dir)
    images_dir = ensure_dir(experiment_dir / "images")

    # Save configuration
    config.save()

    # Data loaders
    print("Loading datasets...")
    train_loader = get_dataloader(
        config.dataroot,
        split="train",
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        shuffle=True
    )

    val_loader = get_dataloader(
        config.dataroot,
        split="val",
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        shuffle=False
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Models
    print("\nInitializing models...")
    generator = GeneratorUNet(config.input_nc, config.output_nc).to(device)
    discriminator = Discriminator(config.input_nc + config.output_nc).to(device)

    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2)
    )

    # Learning rate schedulers
    def lambda_lr(epoch):
        """Linear decay after niter epochs."""
        return 1.0 - max(0, epoch - config.niter) / config.niter_decay

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_lr)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_lr)

    # TensorBoard
    writer = None
    if config.tensorboard:
        writer = SummaryWriter(experiment_dir / "logs")

    # Training loop
    print("\nStarting training...")
    best_metric = 0
    total_epochs = config.niter + config.niter_decay

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_one_epoch(
            generator, discriminator, train_loader,
            optimizer_G, optimizer_D,
            criterion_GAN, criterion_pixelwise,
            device, epoch, config, writer
        )

        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()

        # Validate
        val_metrics = validate(generator, val_loader, device)
        print_metrics(val_metrics, prefix="Validation ")

        if writer:
            for key, value in val_metrics.items():
                writer.add_scalar(f'Val/{key}', value, epoch)

        # Save best model
        if val_metrics['pixel_accuracy'] > best_metric:
            best_metric = val_metrics['pixel_accuracy']
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'metrics': val_metrics
            }, experiment_dir / "best_model.pth")
            print(f"Saved best model (PA: {best_metric:.4f})")

        # Save checkpoint
        if (epoch + 1) % config.save_epoch_freq == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, experiment_dir / f"checkpoint_epoch_{epoch + 1}.pth")

        # Save sample images
        if (epoch + 1) % config.save_epoch_freq == 0:
            generator.eval()
            with torch.no_grad():
                sample_input, sample_target = next(iter(val_loader))
                sample_input = sample_input.to(device)
                sample_output = generator(sample_input)

                save_comparison_grid(
                    [sample_input[i] for i in range(min(4, sample_input.size(0)))],
                    [sample_output[i] for i in range(min(4, sample_output.size(0)))],
                    [sample_target[i] for i in range(min(4, sample_target.size(0)))],
                    save_path=str(images_dir / f"epoch_{epoch + 1}.png")
                )

    # Save final model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, experiment_dir / "final_model.pth")

    if writer:
        writer.close()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Pixel Accuracy: {best_metric:.4f}")
    print(f"Models saved to: {experiment_dir}")
    print("=" * 60)

    return generator, best_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StructGAN baseline")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--preset", type=str, default="baseline",
                        choices=["baseline", "high_res", "fast", "debug"])
    parser.add_argument("--dataroot", type=str, help="Override data directory")
    parser.add_argument("--name", type=str, help="Experiment name")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.load(args.config)
    else:
        config = get_config(args.preset)

    # Override with command line args
    if args.dataroot:
        config.dataroot = args.dataroot
    if args.name:
        config.name = args.name

    # Train
    train_baseline(config)
