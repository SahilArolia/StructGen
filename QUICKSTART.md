# StructGAN Quick Start Guide

This guide will help you set up and reproduce the baseline StructGAN model.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: RTX 3080 or similar)
- 16GB+ RAM
- 50GB+ disk space

## Step 1: Environment Setup

```bash
# Make the setup script executable and run it
chmod +x setup.sh
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Clone StructGAN repository
git clone https://github.com/wenjie-liao/StructGAN_v1.git
```

## Step 2: Verify Installation

```bash
# Activate the virtual environment
source venv/bin/activate

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check dataset exists
ls StructGAN_v1/0_datasets/
```

## Step 3: Explore the Dataset

```bash
# Start Jupyter
jupyter notebook notebooks/01_data_exploration.ipynb
```

Or use the command line:

```python
from src.data_preprocessing.dataset import StructGANDataset, visualize_sample

dataset = StructGANDataset(
    root_dir="StructGAN_v1/0_datasets/Group7-H2",
    image_size=256
)
print(f"Dataset size: {len(dataset)}")
visualize_sample(dataset, idx=0, save_path="sample.png")
```

## Step 4: Train Baseline Model

### Option A: Use our simplified training script

```bash
# Train with default config
python src/training/train_baseline.py --preset baseline

# Train with custom config
python src/training/train_baseline.py --config config.yaml

# Quick debug run
python src/training/train_baseline.py --preset debug
```

### Option B: Use original pix2pixHD (recommended for reproduction)

```bash
cd StructGAN_v1/2_pix2pixHD_adopted

# Train
python train.py \
    --name structgan_baseline \
    --dataroot ../0_datasets/Group7-H2 \
    --model pix2pixHD \
    --netG global \
    --ngf 64 \
    --n_downsample_global 4 \
    --n_blocks_global 9 \
    --niter 100 \
    --niter_decay 100

# Test
python test.py \
    --name structgan_baseline \
    --dataroot ../0_datasets/Group7-H2 \
    --which_epoch latest
```

## Step 5: Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir models/checkpoints/structgan_baseline/logs

# Open http://localhost:6006 in browser
```

## Step 6: Evaluate Results

```python
from src.evaluation.metrics import evaluate_model, print_metrics
from src.data_preprocessing.dataset import get_dataloader

# Load test data
test_loader = get_dataloader("StructGAN_v1/0_datasets/Group7-H2", split="test")

# Evaluate
metrics = evaluate_model(model, test_loader, device)
print_metrics(metrics)
```

## Expected Results

After 200 epochs, you should achieve approximately:
- **Pixel Accuracy**: ~85%
- **Wall IoU**: ~70%
- **Structural IoU**: ~65%

## Project Structure

```
StructGen/
├── config.yaml              # Training configuration
├── requirements.txt         # Python dependencies
├── setup.sh                 # Environment setup script
├── QUICKSTART.md           # This file
├── README.md               # Project overview
│
├── StructGAN_v1/           # Cloned StructGAN repository
│   ├── 0_datasets/         # Original datasets
│   ├── 1_pix2pix/          # Basic pix2pix
│   └── 2_pix2pixHD_adopted/# pix2pixHD implementation
│
├── data/                   # Your processed data
│   ├── structgan_original/ # Links to StructGAN datasets
│   ├── rplan_processed/    # Processed RPLAN data
│   └── augmented/          # Augmented dataset
│
├── models/
│   └── checkpoints/        # Saved model weights
│
├── src/
│   ├── data_preprocessing/ # Data loading & preprocessing
│   ├── training/           # Training scripts & configs
│   ├── evaluation/         # Metrics & evaluation
│   └── utils/              # Utility functions
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
└── webapp/                 # Streamlit interface (Week 7)
```

## Troubleshooting

### CUDA out of memory
- Reduce batch size: `batch_size: 2` in config.yaml
- Use smaller image size: `image_size: 128`

### Dataset not found
- Ensure StructGAN_v1 is cloned: `ls StructGAN_v1/0_datasets/`
- Check data path in config.yaml

### Slow training
- Enable mixed precision: `fp16: true`
- Reduce image size for testing

### Import errors
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall requirements: `pip install -r requirements.txt`

## Next Steps

After reproducing the baseline:

1. **Week 3-4**: Process RPLAN dataset
   ```bash
   python src/data_preprocessing/rplan_processor.py \
       --rplan_dir path/to/rplan \
       --output_dir data/rplan_processed
   ```

2. **Week 5-6**: Add physics constraints (see README.md)

3. **Week 7**: Build Streamlit interface
   ```bash
   streamlit run webapp/streamlit_app.py
   ```

## References

- [StructGAN Paper](https://www.sciencedirect.com/science/article/pii/S0926580521003526)
- [pix2pixHD Paper](https://arxiv.org/abs/1711.11585)
- [Original Repository](https://github.com/wenjie-liao/StructGAN_v1)
