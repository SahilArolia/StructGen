#!/bin/bash
# StructGAN Project Setup Script
# This script sets up the environment and downloads necessary resources

set -e

echo "=========================================="
echo "StructGAN Project Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
    print_status "Python version $python_version is compatible"
else
    print_error "Python 3.8+ is required. Found: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Clone StructGAN repository if not exists
if [ ! -d "StructGAN_v1" ]; then
    print_status "Cloning StructGAN repository..."
    git clone https://github.com/wenjie-liao/StructGAN_v1.git
    print_status "StructGAN repository cloned"
else
    print_status "StructGAN repository already exists"
fi

# Create symlinks to StructGAN dataset in our data folder
if [ -d "StructGAN_v1/0_datasets" ]; then
    print_status "Linking StructGAN datasets..."

    # Link each dataset group
    for dataset in StructGAN_v1/0_datasets/Group*; do
        if [ -d "$dataset" ]; then
            dataset_name=$(basename "$dataset")
            if [ ! -L "data/structgan_original/$dataset_name" ]; then
                ln -sf "../../$dataset" "data/structgan_original/$dataset_name"
                print_status "Linked $dataset_name"
            fi
        fi
    done
fi

# Check for GPU availability
print_status "Checking GPU availability..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('No GPU detected - will use CPU')"

# Create __init__.py files for Python packages
print_status "Setting up Python package structure..."
touch src/__init__.py
touch src/data_preprocessing/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py

echo ""
echo "=========================================="
print_status "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Explore the dataset: jupyter notebook notebooks/01_data_exploration.ipynb"
echo "3. Run baseline training: python src/training/train_baseline.py"
echo ""
echo "StructGAN dataset location: StructGAN_v1/0_datasets/"
echo "Available dataset groups:"
if [ -d "StructGAN_v1/0_datasets" ]; then
    ls -1 StructGAN_v1/0_datasets/ 2>/dev/null | grep "Group" || echo "  (none found yet - clone may still be in progress)"
fi
