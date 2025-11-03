# Super Resolution GAN (SRGAN) for Binary Classification - Midterm Project

## Project Overview

This project implements a Super Resolution Generative Adversarial Network (SRGAN) to generate high-resolution images from low-resolution inputs, and compares the performance of binary classifiers trained on original vs. super-resolved images.

### Objectives
- Train SRGAN to generate 128×128 images from 32×32 inputs
- Build binary classifier A on original 128×128 images
- Build binary classifier B on SRGAN-generated 128×128 images
- Compare model performance using multiple metrics

## Architecture

### SRGAN Components
- **Generator**: 16 residual blocks with pixel-shuffle upsampling
- **Discriminator**: CNN with increasing filters and binary classification
- **Perceptual Loss**: VGG19-based feature matching
- **Adversarial Training**: Combined generator-discriminator optimization

### Binary Classifiers
- **Model A**: ResNet152V2trained on original 128x128 images
- **Model B**: ResNet152V2 trained on SRGAN-generated images

## 📁 Project Structure

```
midterm-project/
│
├── data/
│   ├── processed_128/          # Original 128×128 images
│   └── lowres_32/              # Downscaled 32×32 images
|   └── raw/                    # Origianl 150x150 Images 
├── models/                    # Trained model weights
├── notebooks/                 # Notebook for classifiers A, SRGAN and B
|   └── logs/                  # Logs for SRGAN training
├── checkpoints/               # Model weights during training
├── output/                    # SRGAN-generated images
└── README.md                  # This file
```

## Implementation Steps

### 1. Data Preparation
```python
# Create paired dataset with 70/30 split
highres_path = "/content/drive/MyDrive/midterm/data/processed_128"
lowres_path = "/content/drive/MyDrive/midterm/data/lowres_32"

train_ds, test_ds = create_paired_dataset(highres_path, lowres_path, batch_size=16)
```
### 2. Binary Classifier Training
- **Model A**: Trained on downscaled 128x128 Images of cats and dogs
- **Architecture**: ResNet125V2 with transfer learning

### 2. SRGAN Training
- **Example Scaled Images**: To confirm the images scaled properly, 9x9 plots of images from each folder are shown in the SRGAN file
- **Epochs**: 150+ (as required)
- **Batch Size**: 32
- **Optimizer**: Adam (1e-4)
- **Checkpoints**: Saved every epoch
- **Loss Components**:
  - Adversarial loss (1e-3 weight)
  - Perceptual loss (VGG features)

### 3. Binary Classifier Training
- **Model B**: SRGAN-generated 128×128 images
- **Architecture**: EfficientNetB0 with transfer learning
- **Metrics**: Accuracy, F1-Score, AUC

### 4. Evaluation & Comparison
- Comprehensive metrics comparison
- Visualization of super-resolved images
- Performance analysis

## Technical Requirements

### Dependencies
```python
tensorflow>=2.8.0
keras
matplotlib
numpy
pandas
scikit-learn
Pillow
seaborn
```

### Hardware
- GPU: NVIDIA A100-SXM4-40GB (Google Colab)
- RAM: 40GB GPU Memory
- Storage: Google Drive for model persistence

## Dataset Specifications

- **Original Resolution**: 128×128×3
- **Low Resolution**: 32×32×3
- **Scaling Factor**: 4×
- **Train-Test Split**: 70%-30%
- **Normalization**: [-1, 1] range for GAN training

## Key Functions

### Data Processing
- `create_paired_dataset()`: Creates HR-LR image pairs
- `prepare_srgan_data()`: Prepares data for SRGAN training
- `verify_paired_dataset()`: Validates data pairing

### Model Building
- `build_generator()`: SRGAN generator with residual blocks
- `build_discriminator()`: SRGAN discriminator
- `build_vgg_loss()`: VGG19 for perceptual loss
- `build_classifier_a/b()`: Binary classification models

### Training & Evaluation
- `train_srgan()`: SRGAN training loop
- `generate_sr_images()`: Creates super-resolved images
- `evaluate_models()`: Compares Model A vs Model B

## Performance Metrics

### Evaluation Metrics
- **Accuracy**: Overall classification correctness
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area Under ROC Curve
- **Training Loss**: Generator and discriminator loss curves

### Expected Results
- SRGAN should generate visually plausible 128×128 images
- Model A (original) typically performs better
- Model B (SR-generated) shows SRGAN's effectiveness

## Model Persistence

### Checkpoint Strategy
```python
# Save weights every epoch
checkpoint_cb = ModelCheckpoint(
    'srgan_weights_epoch_{epoch:02d}.h5',
    save_weights_only=True,
    save_freq='epoch'
)
```

### Final Models Saved
- `srgan_generator.h5`: Trained SRGAN generator
- `classifier_a.h5`: Model A (original images)
- `classifier_b.h5`: Model B (SR-generated images)

## Visualization

### Generated Samples
- Low-res input (32×32)
- Super-resolved output (128×128)
- Original high-res reference (128×128)

### Training Progress
- Generator vs Discriminator loss
- Validation accuracy curves
- Model comparison charts

### Training Considerations
- SRGAN training is computationally intensive
- Monitor for mode collapse
- Adjust learning rates if training unstable
- Use smaller batch sizes if memory limited

## Running the Project

### Step-by-Step Execution
1. **Mount Google Drive** and set up paths
2. **Prepare dataset** with proper HR-LR pairing
3. **Train SRGAN** for 150+ epochs with checkpointing
4. **Generate super-resolved** images for Model B
5. **Train both classifiers** (Model A and Model B)
6. **Evaluate and compare** model performance
7. **Save final models** and generate reports

### Quick Start
```python
# Complete pipeline execution
train_ds, test_ds = prepare_srgan_data()
srgan_model, history = train_srgan()
classifier_a = train_classifier_a()
classifier_b = train_classifier_b()
results = evaluate_models(classifier_a, classifier_b)
```

## Results Interpretation

### Successful SRGAN Training
- Decreasing generator and discriminator loss
- Visually coherent super-resolved images
- Stable training without mode collapse

### Model Comparison
- Model A baseline performance
- Model B demonstrates SRGAN quality
- Gap indicates SRGAN improvement potential

## 🔍 Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch size
2. **Training instability**: Adjust learning rates
3. **Poor SR quality**: Increase training epochs
4. **Dataset pairing**: Verify file correspondence

### Validation
- Use `verify_paired_dataset()` to check data alignment
- Monitor loss curves for training stability
- Regularly visualize generated samples

## Author: Eka Ebong
- **Course**: Applied AI
- **Assignment**: Midterm Exam
- **Focus**: SRGAN implementation and evaluation

---

*This implementation follows the midterm requirements including 150+ epoch training, 70-30 dataset split, proper normalization, and comprehensive model comparison.*