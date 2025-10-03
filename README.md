# Hyperspectral Material Classification Pipeline

Deep learning pipeline for classifying plastic materials using hyperspectral imaging data (26 bands, 450-700nm).

## Dataset Structure

- **Training data**: `data/bands/` (26 spectral band images) + `data/labels/` (segmentation labels)
- **Inference data**: `inference_data_set1/` and `inference_data_set2/`
- **11 material classes**: Background, 95PU, HIPS, HVDF-HFP, GPSS, PU, 75PU, 85PU, PETE, PET, PMMA
- **Label handling**: Pixels matching exact plastic RGB codes → plastic classes; all other pixels → Background (class 0)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

Train a model on the hyperspectral data:

```bash
# ULTRA FAST MODE - Use binning + sampling for quick testing (4x faster!)
python train.py --model spectral_cnn_2d --use_patches --patch_size 3 --epochs 100 --batch_size 512 \
    --augment --max_samples_per_class 5000 --bin_factor 2

# FAST MODE - Use sampling only
python train.py --model spectral_cnn_2d --use_patches --patch_size 3 --epochs 100 --batch_size 256 \
    --augment --max_samples_per_class 5000

# Train hybrid model
python train.py --model hybrid --use_patches --patch_size 5 --epochs 100 --batch_size 128 \
    --augment --max_samples_per_class 5000 --bin_factor 2

# Train ResNet-based model
python train.py --model resnet --use_patches --patch_size 3 --epochs 150 --batch_size 128 \
    --max_samples_per_class 5000

# Full resolution training (slower but potentially more accurate)
python train.py --model hybrid --use_patches --augment --epochs 100
```

### 2. Inference

Run inference on new data:

```bash
# Single dataset
python inference.py \
    --checkpoint outputs/spectral_cnn_2d_20231115_120000/best_model.pth \
    --model spectral_cnn_2d \
    --use_patches \
    --data_dir inference_data_set1

# Multiple datasets (parent directory)
python inference.py \
    --checkpoint outputs/spectral_cnn_2d_20231115_120000/best_model.pth \
    --model spectral_cnn_2d \
    --use_patches \
    --data_dir .
```

Output includes:
- `predictions.npy` - Raw class predictions
- `confidence.npy` - Prediction confidence scores
- `prediction_visualization.png` - Color-coded visualization
- `statistics.json` - Class distribution and confidence stats

## Model Architectures

1. **SpectralCNN1D**: 1D CNN on spectral signatures (pixel-wise)
2. **SpectralCNN2D**: 2D CNN using spatial patches with spectral channels
3. **HybridSpectralNet**: Combines spectral (1D) and spatial (2D) features
4. **SpectralResNet**: ResNet-based architecture with residual connections

## Files

- `dataset.py` - Dataset loaders for hyperspectral data
- `model.py` - Neural network architectures
- `train.py` - Training script
- `inference.py` - Inference script
- `labels.json` - Class mappings and colors

## Training Arguments

```
--data_dir                 Path to training data (default: data)
--model                    Model architecture (spectral_cnn_1d|spectral_cnn_2d|hybrid|resnet)
--use_patches              Use spatial patches for 2D models
--patch_size               Patch size (default: 3)
--epochs                   Number of epochs (default: 100)
--batch_size               Batch size (default: 256)
--lr                       Learning rate (default: 0.001)
--augment                  Enable data augmentation
--train_split              Train/val split ratio (default: 0.8)
--dropout                  Dropout rate (default: 0.3)
--max_samples_per_class    Max samples per class for faster training (e.g., 5000)
                           Use this to speed up training significantly!
--bin_factor               Pixel binning factor (e.g., 2, 4). Reduces image resolution
                           bin_factor=2 → 4x fewer pixels, bin_factor=4 → 16x fewer pixels
```

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (ultra fast mode with binning + sampling for quick testing)
python train.py --model hybrid --use_patches --augment --epochs 100 \
    --max_samples_per_class 5000 --bin_factor 2

# 3. Run inference
python inference.py \
    --checkpoint outputs/hybrid_TIMESTAMP/best_model.pth \
    --model hybrid \
    --use_patches \
    --data_dir inference_data_set1
```

## Performance Tips

### Speed Optimization (for quick testing):
- **Pixel binning**: `--bin_factor 2` (4x faster) or `--bin_factor 4` (16x faster)
- **Class sampling**: `--max_samples_per_class 5000` limits samples per class
- **Combine both**: `--bin_factor 2 --max_samples_per_class 5000` for maximum speed
- **Increase batch size**: Use larger batches when using binning (e.g., `--batch_size 512`)

### Accuracy Optimization (for final model):
- Use `--augment` for better generalization
- Increase `patch_size` (5, 7) for more spatial context
- Use `hybrid` or `resnet` models for best accuracy
- Train on full resolution (`--bin_factor 1`) with all samples
- Monitor per-class accuracy for class imbalance issues
