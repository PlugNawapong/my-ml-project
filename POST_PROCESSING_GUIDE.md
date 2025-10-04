# Post-Processing Guide: Reducing Noise in Predictions

## Problem
Background predictions on dataset 2 have noise (salt-and-pepper artifacts, misclassified pixels).

## Solution: Post-Processing Filters

Three filtering methods are applied sequentially:

### 1. **Majority Filter (Mode Filter)**
- Takes a window (e.g., 5×5) around each pixel
- Assigns the most common class in that window
- **Effect**: Smooths predictions, removes isolated pixels

### 2. **Morphological Operations**
- **Opening**: Removes small noise (erosion → dilation)
- **Closing**: Fills small holes (dilation → erosion)
- **Effect**: Cleans up object boundaries, removes speckles

### 3. **Confidence Thresholding** (Optional)
- Sets low-confidence predictions to background
- **Effect**: Removes uncertain predictions

---

## Usage

### Integrated (During Inference)
```bash
python inference.py \
    --checkpoint model.pth \
    --model spectral_cnn_1d \
    --data_dir inference_data_set2 \
    --norm_method percentile \
    --post_process \
    --majority_size 5 \
    --morph_size 3
```

**Parameters:**
- `--post_process`: Enable post-processing
- `--majority_size`: Window size for majority filter (default: 3, recommended: 5 for more smoothing)
- `--morph_size`: Kernel size for morphological operations (default: 3)

**Output:**
- `predictions.npy` - Raw predictions
- `predictions_filtered.npy` - Post-processed predictions
- `prediction_visualization.png` - Raw visualization
- `prediction_filtered_visualization.png` - Filtered visualization

### Standalone (After Inference)
```bash
python post_process.py \
    --input_dir predictions/inference_data_set2 \
    --majority_size 5 \
    --morph_size 3
```

---

## Recommended Settings

### Light Filtering (Preserve Details)
```bash
--post_process --majority_size 3 --morph_size 3
```

### Medium Filtering (Balanced) - **Recommended**
```bash
--post_process --majority_size 5 --morph_size 3
```

### Heavy Filtering (Maximum Smoothing)
```bash
--post_process --majority_size 7 --morph_size 5
```

---

## When to Use

✅ **Use post-processing when:**
- Background has salt-and-pepper noise
- Small isolated misclassified pixels
- Need smoother boundaries
- Working with 1D CNN (no spatial context during training)

❌ **Don't use if:**
- Predictions are already clean
- Need to preserve fine details
- Working with small objects (filtering might remove them)

---

## Google Colab

Post-processing is enabled by default for dataset 2 in the Colab notebook:

```python
!python inference.py \
    --checkpoint {latest_model} \
    --model {model_type} \
    --data_dir /content/my-ml-project/inference_data_set2 \
    --norm_method percentile \
    --post_process \
    --majority_size 5 \
    --morph_size 3 \
    --output_dir /content/drive/MyDrive/dl-plastics-predictions
```

Results are saved to Google Drive with both raw and filtered versions!

---

## Visual Comparison

**Before Post-Processing:**
- Noisy background
- Isolated misclassified pixels
- Rough boundaries

**After Post-Processing:**
- Clean background
- Smooth regions
- Better-defined boundaries

Check both `prediction_visualization.png` and `prediction_filtered_visualization.png` to compare!
