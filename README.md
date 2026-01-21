# StructGAN Project Plan
## Deep Learning-Based Automated Structural Layout Generation from Architectural Floor Plans

---

## 📋 Project Overview

| Attribute | Details |
|-----------|---------|
| **Project Title** | Automated Structural Layout Generation using GANs |
| **Duration** | 8 Weeks |
| **Domain** | AI/ML + Structural Engineering |
| **Core Technology** | Generative Adversarial Networks (pix2pixHD) |
| **Input** | Architectural Floor Plan (Image) |
| **Output** | Structural Layout with Shear Walls & Columns |

### Problem Statement
Traditional structural design is a manual, iterative process where engineers interpret architectural floor plans and design structural layouts through trial-and-error. This takes 2-4 weeks per building and relies heavily on individual experience.

### Solution
Use deep learning (GANs) to automatically generate structural layouts from architectural floor plans, reducing design time from weeks to seconds while maintaining code compliance.

---

## 🎯 Project Objectives

1. **Reproduce Baseline**: Successfully run StructGAN on the provided dataset
2. **Understand Architecture**: Deep dive into pix2pix and pix2pixHD architectures
3. **Expand Dataset**: Integrate RPLAN dataset (80,000+ floor plans)
4. **Improve Model**: Add physics-based constraints and Indian code compliance
5. **Build Interface**: Create a web-based tool for practical use
6. **Document & Evaluate**: Comprehensive evaluation and technical report

---

## 📁 Repository Structure

```
StructGAN_Project/
├── 📂 data/
│   ├── structgan_original/      # Original StructGAN dataset
│   ├── rplan_processed/         # Processed RPLAN data
│   └── augmented/               # Augmented dataset
├── 📂 models/
│   ├── pix2pix/                 # Basic pix2pix implementation
│   ├── pix2pixHD/               # High-resolution GAN
│   └── checkpoints/             # Saved model weights
├── 📂 src/
│   ├── data_preprocessing/      # Data processing scripts
│   ├── training/                # Training scripts
│   ├── evaluation/              # Evaluation metrics
│   └── utils/                   # Utility functions
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── 📂 webapp/
│   └── streamlit_app.py         # Web interface
├── 📂 docs/
│   ├── literature_review.md
│   └── technical_report.md
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Information

### Primary Dataset: StructGAN
| Attribute | Value |
|-----------|-------|
| Source | [GitHub - StructGAN_v1](https://github.com/wenjie-liao/StructGAN_v1) |
| Location | `0_datasets/` folder |
| Format | Paired images (256×256 or 512×512) |
| Content | Architectural → Structural layout pairs |

### Secondary Dataset: RPLAN
| Attribute | Value |
|-----------|-------|
| Source | [RPLAN Request Form](https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform) |
| Size | 80,788 floor plans |
| Format | 256×256 PNG images with semantic annotations |
| Channels | Multi-channel (room types, walls, doors) |

### Data Format (StructGAN)
```
Input Image (Architectural):
- Room boundaries (different colors per room type)
- Doors and windows
- Building outline

Output Image (Structural):
- Shear wall positions (specific color)
- Column positions (specific color)
- Wall thickness encoding
```

---

## 🏗️ Model Architecture

### pix2pixHD Architecture (Primary)

```
┌─────────────────────────────────────────────────────────────┐
│                      GENERATOR (G)                          │
├─────────────────────────────────────────────────────────────┤
│  Global Generator (G1):                                     │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐                │
│  │ Encoder │ -> │ ResBlocks│ -> │ Decoder │                │
│  │ (Down)  │    │ (9 blocks)│   │  (Up)   │                │
│  └─────────┘    └──────────┘    └─────────┘                │
│                                                             │
│  Local Enhancer (G2) - for high resolution:                 │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐                │
│  │ Encoder │ -> │ ResBlocks│ -> │ Decoder │ + G1 output    │
│  │ (Down)  │    │ (3 blocks)│   │  (Up)   │                │
│  └─────────┘    └──────────┘    └─────────┘                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   DISCRIMINATOR (D)                         │
├─────────────────────────────────────────────────────────────┤
│  Multi-scale Discriminator:                                 │
│  D1: Original resolution                                    │
│  D2: Downsampled 2x                                         │
│  D3: Downsampled 4x                                         │
│                                                             │
│  Each discriminator: PatchGAN (70×70 receptive field)       │
└─────────────────────────────────────────────────────────────┘
```

### Loss Functions
```python
# Total Loss
L_total = L_GAN + λ_feat * L_feat + λ_VGG * L_VGG

# Where:
# L_GAN   = Adversarial loss (hinge loss)
# L_feat  = Feature matching loss (discriminator features)
# L_VGG   = Perceptual loss (VGG19 features)
```

### Key Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 1-4 | Limited by GPU memory |
| `lr_G` | 0.0002 | Generator learning rate |
| `lr_D` | 0.0002 | Discriminator learning rate |
| `beta1` | 0.5 | Adam optimizer |
| `lambda_feat` | 10.0 | Feature matching weight |
| `n_epochs` | 200 | Training epochs |
| `input_nc` | 3 | Input channels (RGB) |
| `output_nc` | 3 | Output channels (RGB) |

---

## 📅 Week-by-Week Plan

### Week 1: Environment Setup & Literature Review

#### Tasks
- [ ] Set up development environment
- [ ] Clone StructGAN repository
- [ ] Install dependencies
- [ ] Read and understand the core paper (Liao et al. 2021)
- [ ] Read supporting papers (pix2pix, pix2pixHD)
- [ ] Explore the dataset structure

#### Commands
```bash
# Clone repository
git clone https://github.com/wenjie-liao/StructGAN_v1.git
cd StructGAN_v1

# Create virtual environment
conda create -n structgan python=3.8
conda activate structgan

# Install dependencies
pip install torch==1.9.0 torchvision==0.10.0
pip install tensorflow==1.15.0
pip install opencv-python scikit-image scipy numpy matplotlib
pip install pillow tqdm tensorboard
```

#### Reading List
1. **Core Paper**: Liao et al. (2021) - "Automated structural design of shear wall residential buildings using GANs"
2. **pix2pix**: Isola et al. (2017) - "Image-to-Image Translation with Conditional Adversarial Networks"
3. **pix2pixHD**: Wang et al. (2018) - "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"

#### Deliverables
- [ ] Working environment with all dependencies
- [ ] Literature review notes (1-2 pages)
- [ ] Dataset exploration notebook

---

### Week 2: Reproduce Baseline Results

#### Tasks
- [ ] Understand the data preprocessing pipeline
- [ ] Run training on original StructGAN dataset
- [ ] Reproduce the results from the paper
- [ ] Analyze the generated outputs
- [ ] Document any issues and solutions

#### Training Command
```bash
# Navigate to pix2pixHD folder
cd 2_pix2pixHD_adopted

# Train the model
python train.py --name structgan_baseline \
    --dataroot ../0_datasets/Group7-H2 \
    --model pix2pixHD \
    --netG global \
    --ngf 64 \
    --n_downsample_global 4 \
    --n_blocks_global 9 \
    --niter 100 \
    --niter_decay 100

# Test the model
python test.py --name structgan_baseline \
    --dataroot ../0_datasets/Group7-H2 \
    --model pix2pixHD
```

#### Evaluation Metrics to Compute
```python
# Pixel Accuracy (PA)
PA = (True Positives + True Negatives) / Total Pixels

# Intersection over Union (IoU)
IoU = Intersection / Union

# Wall IoU (WIoU) - specific to wall regions
WIoU = Wall_Intersection / Wall_Union

# Structural IoU (SIoU) - specific to structural elements
SIoU = Struct_Intersection / Struct_Union

# Shear Wall Ratio Difference
SW_diff = |SW_ratio_generated - SW_ratio_groundtruth|
```

#### Deliverables
- [ ] Trained baseline model
- [ ] Evaluation results matching paper (~85% PA)
- [ ] Sample generated images (10-20)
- [ ] Training logs and loss curves

---

### Week 3: Data Preprocessing & Expansion

#### Tasks
- [ ] Request and download RPLAN dataset
- [ ] Analyze RPLAN data format
- [ ] Create preprocessing pipeline to convert RPLAN to StructGAN format
- [ ] Generate structural layouts for RPLAN (rule-based or manual annotation)
- [ ] Implement data augmentation

#### RPLAN Data Processing Script
```python
# rplan_to_structgan.py

import numpy as np
import cv2
from pathlib import Path

class RPLANProcessor:
    """Convert RPLAN format to StructGAN format"""
    
    # RPLAN color mapping
    ROOM_TYPES = {
        0: "background",
        1: "living_room",
        2: "master_room", 
        3: "kitchen",
        4: "bathroom",
        5: "dining_room",
        6: "child_room",
        7: "study_room",
        8: "second_room",
        9: "guest_room",
        10: "balcony",
        11: "entrance",
        12: "storage",
        13: "wall",
        14: "external_area",
        15: "exterior_wall",
        16: "front_door",
        17: "interior_door"
    }
    
    def __init__(self, rplan_path, output_path):
        self.rplan_path = Path(rplan_path)
        self.output_path = Path(output_path)
        
    def process_single(self, image_path):
        """Process a single RPLAN image"""
        img = cv2.imread(str(image_path))
        
        # Extract architectural elements
        arch_layout = self.extract_architectural(img)
        
        # Generate structural layout (rule-based)
        struct_layout = self.generate_structural(img)
        
        return arch_layout, struct_layout
    
    def extract_architectural(self, img):
        """Extract architectural floor plan from RPLAN"""
        # Implementation based on RPLAN format
        # Extract rooms, doors, windows
        pass
    
    def generate_structural(self, img):
        """Generate structural layout using rules"""
        # Rule 1: External walls become shear walls
        # Rule 2: Long internal walls may need columns
        # Rule 3: Corners need columns
        pass
    
    def augment(self, arch_img, struct_img):
        """Apply data augmentation"""
        augmented = []
        
        # Rotation (90, 180, 270 degrees)
        for angle in [90, 180, 270]:
            rot_arch = self.rotate(arch_img, angle)
            rot_struct = self.rotate(struct_img, angle)
            augmented.append((rot_arch, rot_struct))
        
        # Flip (horizontal, vertical)
        for flip in ['h', 'v']:
            flip_arch = self.flip(arch_img, flip)
            flip_struct = self.flip(struct_img, flip)
            augmented.append((flip_arch, flip_struct))
        
        return augmented
```

#### Data Augmentation Techniques
| Technique | Description | Multiplier |
|-----------|-------------|------------|
| Rotation | 90°, 180°, 270° | 4x |
| Flipping | Horizontal, Vertical | 3x |
| Scaling | 0.8x - 1.2x | 3x |
| Color Jitter | Brightness/Contrast | 2x |
| **Total** | | **~72x** |

#### Deliverables
- [ ] RPLAN dataset downloaded
- [ ] Preprocessing pipeline code
- [ ] Expanded dataset (5,000+ paired samples)
- [ ] Data augmentation pipeline

---

### Week 4: Dataset Annotation & Validation

#### Tasks
- [ ] Create/refine structural annotations for RPLAN
- [ ] Validate data quality
- [ ] Split data into train/val/test sets
- [ ] Create data loading utilities
- [ ] Visualize dataset statistics

#### Data Split Strategy
```python
# Stratified split based on building characteristics
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for validation
test_ratio = 0.1   # 10% for testing

# Ensure similar distribution of:
# - Number of rooms
# - Building area
# - Building shape complexity
```

#### Data Validation Checks
```python
def validate_pair(arch_img, struct_img):
    """Validate architectural-structural pair"""
    checks = {
        "size_match": arch_img.shape == struct_img.shape,
        "has_walls": np.any(struct_img > 0),
        "boundary_match": check_boundary_alignment(arch_img, struct_img),
        "no_overflow": struct_within_boundary(arch_img, struct_img),
        "min_elements": count_structural_elements(struct_img) >= 4
    }
    return all(checks.values()), checks
```

#### Deliverables
- [ ] Validated dataset with quality checks
- [ ] Train/Val/Test splits
- [ ] Dataset statistics report
- [ ] Data loader implementation

---

### Week 5: Model Training on Expanded Dataset

#### Tasks
- [ ] Train model on expanded dataset
- [ ] Implement learning rate scheduling
- [ ] Monitor training with TensorBoard
- [ ] Compare with baseline results
- [ ] Tune hyperparameters

#### Training Configuration
```python
# config.py

CONFIG = {
    # Data
    "dataroot": "./data/expanded_dataset",
    "image_size": 512,
    "batch_size": 2,
    
    # Model
    "netG": "global",  # or "local" for higher resolution
    "ngf": 64,
    "n_downsample_global": 4,
    "n_blocks_global": 9,
    
    # Training
    "niter": 100,
    "niter_decay": 100,
    "lr": 0.0002,
    "beta1": 0.5,
    
    # Loss weights
    "lambda_feat": 10.0,
    "lambda_vgg": 10.0,
    
    # Checkpoints
    "save_epoch_freq": 10,
    "print_freq": 100
}
```

#### TensorBoard Monitoring
```bash
# Start TensorBoard
tensorboard --logdir ./checkpoints/structgan_expanded/logs

# Monitor:
# - Generator loss
# - Discriminator loss
# - Feature matching loss
# - VGG perceptual loss
# - Sample images
```

#### Deliverables
- [ ] Trained model on expanded dataset
- [ ] Training curves and logs
- [ ] Model checkpoints
- [ ] Comparison report (baseline vs expanded)

---

### Week 6: Physics Constraints & Code Compliance

#### Tasks
- [ ] Study Indian structural codes (IS 456, IS 13920)
- [ ] Implement physics-based loss functions
- [ ] Add structural validity constraints
- [ ] Retrain model with constraints
- [ ] Validate structural outputs

#### Indian Code Requirements (IS 456 / IS 13920)

| Parameter | Requirement | Implementation |
|-----------|-------------|----------------|
| Minimum Wall Thickness | 150mm for load-bearing | Check pixel ratio |
| Column Spacing | Max 7.5m typically | Distance constraint |
| Shear Wall Length | Min 4x thickness | Aspect ratio check |
| Symmetry | Preferred for seismic | Symmetry loss |
| Wall Distribution | Along both directions | Coverage loss |

#### Physics-Based Loss Functions
```python
class PhysicsLoss:
    """Physics-based constraints for structural design"""
    
    def __init__(self):
        self.min_wall_thickness = 0.05  # As ratio of image
        self.max_span = 0.3  # Maximum unsupported span
        
    def wall_thickness_loss(self, generated):
        """Penalize walls that are too thin"""
        walls = extract_walls(generated)
        thickness = compute_thickness(walls)
        violation = torch.relu(self.min_wall_thickness - thickness)
        return violation.mean()
    
    def symmetry_loss(self, generated):
        """Encourage symmetric layouts for seismic resistance"""
        flipped_h = torch.flip(generated, dims=[3])
        flipped_v = torch.flip(generated, dims=[2])
        
        sym_h = F.mse_loss(generated, flipped_h)
        sym_v = F.mse_loss(generated, flipped_v)
        
        return (sym_h + sym_v) / 2
    
    def coverage_loss(self, generated, boundary):
        """Ensure walls cover building adequately"""
        wall_coverage = compute_wall_coverage(generated)
        min_coverage = 0.15  # At least 15% wall coverage
        violation = torch.relu(min_coverage - wall_coverage)
        return violation
    
    def continuity_loss(self, generated):
        """Penalize discontinuous walls"""
        # Walls should be connected, not floating
        pass

# Combined loss
def total_loss(real, fake, discriminator):
    gan_loss = adversarial_loss(discriminator, real, fake)
    feat_loss = feature_matching_loss(discriminator, real, fake)
    vgg_loss = perceptual_loss(real, fake)
    physics_loss = PhysicsLoss()(fake)
    
    return (gan_loss + 
            10 * feat_loss + 
            10 * vgg_loss + 
            5 * physics_loss)  # New physics term
```

#### Deliverables
- [ ] Physics loss implementation
- [ ] Code compliance checker
- [ ] Retrained model with constraints
- [ ] Structural validation report

---

### Week 7: Web Interface Development

#### Tasks
- [ ] Design user interface
- [ ] Implement Streamlit app
- [ ] Add file upload functionality
- [ ] Integrate trained model
- [ ] Add visualization features

#### Streamlit App Structure
```python
# webapp/streamlit_app.py

import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()
    return model

def main():
    st.title("🏗️ AI Structural Layout Generator")
    st.markdown("Upload an architectural floor plan to generate structural layout")
    
    # Sidebar
    st.sidebar.header("Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.8)
    show_overlay = st.sidebar.checkbox("Show Overlay", True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Architectural Floor Plan",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input: Architectural Plan")
            input_image = Image.open(uploaded_file)
            st.image(input_image)
        
        if st.button("🚀 Generate Structural Layout"):
            with st.spinner("Generating..."):
                # Preprocess
                input_tensor = preprocess(input_image)
                
                # Generate
                model = load_model()
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Postprocess
                output_image = postprocess(output)
                
            with col2:
                st.subheader("Output: Structural Layout")
                st.image(output_image)
            
            # Metrics
            st.subheader("📊 Analysis")
            metrics = analyze_structure(output_image)
            
            col3, col4, col5 = st.columns(3)
            col3.metric("Wall Coverage", f"{metrics['coverage']:.1%}")
            col4.metric("Symmetry Score", f"{metrics['symmetry']:.2f}")
            col5.metric("Elements Count", metrics['element_count'])
            
            # Download
            st.download_button(
                "📥 Download Result",
                data=output_image.tobytes(),
                file_name="structural_layout.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
```

#### UI Features
- [x] File upload for floor plan images
- [x] Real-time generation
- [x] Side-by-side comparison
- [x] Overlay visualization
- [x] Structural metrics display
- [x] Download generated layout
- [ ] Batch processing (optional)
- [ ] DXF export (optional)

#### Deliverables
- [ ] Working Streamlit application
- [ ] User documentation
- [ ] Demo screenshots/video

---

### Week 8: Evaluation, Documentation & Final Submission

#### Tasks
- [ ] Comprehensive model evaluation
- [ ] Compare with existing methods
- [ ] Write technical report
- [ ] Prepare presentation
- [ ] Create demo video
- [ ] Final code cleanup

#### Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Pixel Accuracy (PA) | (TP + TN) / Total | > 80% |
| Wall IoU (WIoU) | Wall ∩ / Wall ∪ | > 70% |
| Structural IoU (SIoU) | Struct ∩ / Struct ∪ | > 65% |
| FID Score | Fréchet Inception Distance | < 50 |
| Generation Time | seconds/image | < 1s |

#### Evaluation Script
```python
# evaluation/evaluate.py

import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model(model, test_loader):
    results = {
        'pixel_accuracy': [],
        'wall_iou': [],
        'structural_iou': [],
        'generation_time': []
    }
    
    for batch in test_loader:
        input_img, target_img = batch
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            generated = model(input_img)
        gen_time = time.time() - start_time
        
        # Compute metrics
        pa = pixel_accuracy(generated, target_img)
        wiou = wall_iou(generated, target_img)
        siou = structural_iou(generated, target_img)
        
        results['pixel_accuracy'].append(pa)
        results['wall_iou'].append(wiou)
        results['structural_iou'].append(siou)
        results['generation_time'].append(gen_time)
    
    # Aggregate
    return {k: np.mean(v) for k, v in results.items()}
```

#### Technical Report Outline
```
1. Introduction
   1.1 Problem Statement
   1.2 Objectives
   1.3 Contributions

2. Literature Review
   2.1 Traditional Structural Design
   2.2 AI in Structural Engineering
   2.3 GANs for Image Translation

3. Methodology
   3.1 Dataset Preparation
   3.2 Model Architecture
   3.3 Physics-Based Constraints
   3.4 Training Strategy

4. Implementation
   4.1 Technical Stack
   4.2 Data Pipeline
   4.3 Model Training
   4.4 Web Interface

5. Results & Discussion
   5.1 Quantitative Evaluation
   5.2 Qualitative Analysis
   5.3 Comparison with Baselines
   5.4 Ablation Studies

6. Conclusion & Future Work

References
Appendix
```

#### Deliverables
- [ ] Evaluation report with metrics
- [ ] Technical report (10-15 pages)
- [ ] Presentation slides (15-20 slides)
- [ ] Demo video (3-5 minutes)
- [ ] Clean, documented codebase
- [ ] README with setup instructions

---

## 🛠️ Technical Stack

### Software Requirements
```
Python >= 3.8
PyTorch >= 1.9.0
TensorFlow >= 1.15.0 (for original pix2pix)
CUDA >= 11.0
```

### Python Packages
```txt
# requirements.txt

# Deep Learning
torch==1.9.0
torchvision==0.10.0
tensorflow==1.15.0

# Image Processing
opencv-python==4.5.5.64
scikit-image==0.18.3
Pillow==8.4.0

# Data Processing
numpy==1.21.0
pandas==1.3.0
scipy==1.7.0

# Visualization
matplotlib==3.4.3
seaborn==0.11.2
tensorboard==2.7.0

# Web Interface
streamlit==1.12.0
gradio==3.0.0  # Alternative

# Utilities
tqdm==4.62.0
pyyaml==6.0
```

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1060 6GB | RTX 3080 10GB |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB | 100 GB SSD |

---

## 📚 References

### Primary Papers
1. Liao, W., Lu, X., Huang, Y., Zheng, Z., & Lin, Y. (2021). **Automated structural design of shear wall residential buildings using generative adversarial networks**. *Automation in Construction*, 132, 103931.

2. Lu, X., Liao, W., Zhang, Y., & Huang, Y. (2022). **Intelligent structural design of shear wall residence using physics-enhanced generative adversarial networks**. *Earthquake Engineering & Structural Dynamics*, 51(7), 1657-1676.

3. Gu, M., et al. (2024). **Intelligent design of shear wall layout based on diffusion models**. *Computer-Aided Civil and Infrastructure Engineering*, 39(11), 1-18.

### GAN Architecture Papers
4. Isola, P., Zhu, J.Y., Zhou, T., & Efros, A.A. (2017). **Image-to-image translation with conditional adversarial networks**. *CVPR*.

5. Wang, T.C., Liu, M.Y., Zhu, J.Y., Tao, A., Kautz, J., & Catanzaro, B. (2018). **High-resolution image synthesis and semantic manipulation with conditional GANs**. *CVPR*.

### Dataset Papers
6. Wu, W., Fu, X.M., Tang, R., Wang, Y., Qi, Y.H., & Liu, L. (2019). **Data-driven interior plan generation for residential buildings**. *ACM Transactions on Graphics (SIGGRAPH Asia)*, 38(6), 1-12.

7. Nauata, N., Chang, K.H., Cheng, C.Y., Mori, G., & Furukawa, Y. (2021). **House-GAN++: Generative Adversarial Layout Refinement Networks**. *CVPR*.

---

## ✅ Project Checklist

### Week 1-2: Foundation
- [ ] Environment setup complete
- [ ] Literature review done
- [ ] Baseline model reproduced
- [ ] Dataset explored and understood

### Week 3-4: Data
- [ ] RPLAN downloaded
- [ ] Preprocessing pipeline created
- [ ] Data augmentation implemented
- [ ] Dataset validated and split

### Week 5-6: Model
- [ ] Model trained on expanded data
- [ ] Physics constraints added
- [ ] Code compliance integrated
- [ ] Model optimized and evaluated

### Week 7-8: Deployment
- [ ] Web interface built
- [ ] Comprehensive evaluation done
- [ ] Technical report written
- [ ] Presentation prepared
- [ ] Code documented and cleaned

---

## 📞 Resources & Support

### Useful Links
- [StructGAN GitHub](https://github.com/wenjie-liao/StructGAN_v1)
- [pix2pixHD GitHub](https://github.com/NVIDIA/pix2pixHD)
- [RPLAN Toolbox](https://github.com/zzilch/RPLAN-Toolbox)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Indian Code References
- IS 456:2000 - Plain and Reinforced Concrete
- IS 13920:2016 - Ductile Design and Detailing of RC Structures

---

*Last Updated: January 2026*
*Author: Sahil Arolia*
*Project Guide: [To be added]*
