#!/bin/bash
# Script to run the original StructGAN pix2pixHD training
# This provides exact reproduction of the paper's results

set -e

# Configuration
DATASET="Group7-H2"
NAME="structgan_original"
EPOCHS=200
BATCH_SIZE=1

echo "=========================================="
echo "StructGAN Original Training"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Experiment: $NAME"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Check if StructGAN repo exists
if [ ! -d "StructGAN_v1" ]; then
    echo "Error: StructGAN_v1 directory not found."
    echo "Please run ./setup.sh first to clone the repository."
    exit 1
fi

# Check if dataset exists
if [ ! -d "StructGAN_v1/0_datasets/$DATASET" ]; then
    echo "Error: Dataset $DATASET not found."
    echo "Available datasets:"
    ls StructGAN_v1/0_datasets/ 2>/dev/null | grep "Group" || echo "  (none found)"
    exit 1
fi

# Navigate to pix2pixHD directory
cd StructGAN_v1/2_pix2pixHD_adopted

# Check if required files exist
if [ ! -f "train.py" ]; then
    echo "Error: train.py not found in pix2pixHD directory."
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

# Run training
python train.py \
    --name "$NAME" \
    --dataroot "../0_datasets/$DATASET" \
    --model pix2pixHD \
    --netG global \
    --ngf 64 \
    --ndf 64 \
    --n_downsample_global 4 \
    --n_blocks_global 9 \
    --num_D 3 \
    --n_layers_D 3 \
    --niter 100 \
    --niter_decay 100 \
    --lambda_feat 10.0 \
    --batchSize $BATCH_SIZE \
    --gpu_ids 0 \
    --save_epoch_freq 10 \
    --print_freq 100

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Checkpoints saved to: StructGAN_v1/2_pix2pixHD_adopted/checkpoints/$NAME"
echo ""
echo "To test the model, run:"
echo "  cd StructGAN_v1/2_pix2pixHD_adopted"
echo "  python test.py --name $NAME --dataroot ../0_datasets/$DATASET --which_epoch latest"
