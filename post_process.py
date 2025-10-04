import numpy as np
from scipy.ndimage import median_filter, binary_opening, binary_closing
from scipy import ndimage

def majority_filter(prediction_map, size=3):
    """Apply majority (mode) filter to smooth predictions"""
    from scipy.stats import mode

    height, width = prediction_map.shape
    filtered = np.zeros_like(prediction_map)
    pad = size // 2

    padded = np.pad(prediction_map, pad, mode='edge')

    for i in range(height):
        for j in range(width):
            window = padded[i:i+size, j:j+size]
            filtered[i, j] = mode(window, axis=None, keepdims=False)[0]

    return filtered

def median_filter_predictions(prediction_map, size=3):
    """Apply median filter to remove salt-and-pepper noise"""
    return median_filter(prediction_map, size=size)

def morphological_clean(prediction_map, kernel_size=3):
    """Clean predictions using morphological operations"""
    # For each class, clean separately
    cleaned = np.zeros_like(prediction_map)

    for class_id in np.unique(prediction_map):
        # Create binary mask for this class
        mask = (prediction_map == class_id).astype(np.uint8)

        # Remove small noise (opening = erosion + dilation)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        cleaned_mask = binary_opening(mask, structure=kernel)

        # Fill small holes (closing = dilation + erosion)
        cleaned_mask = binary_closing(cleaned_mask, structure=kernel)

        # Apply to output
        cleaned[cleaned_mask > 0] = class_id

    return cleaned

def confidence_threshold_filter(prediction_map, confidence_map, threshold=0.7, default_class=0):
    """Set low-confidence predictions to background"""
    filtered = prediction_map.copy()
    filtered[confidence_map < threshold] = default_class
    return filtered

def apply_all_filters(prediction_map, confidence_map=None,
                      use_majority=True, majority_size=3,
                      use_morphological=True, morph_size=3,
                      use_confidence=False, conf_threshold=0.7):
    """Apply combination of filters"""
    result = prediction_map.copy()

    # Step 1: Confidence thresholding (if available)
    if use_confidence and confidence_map is not None:
        print(f'  Applying confidence threshold ({conf_threshold})...')
        result = confidence_threshold_filter(result, confidence_map, conf_threshold)

    # Step 2: Majority filter (smoothing)
    if use_majority:
        print(f'  Applying majority filter (size={majority_size})...')
        result = majority_filter(result, size=majority_size)

    # Step 3: Morphological cleaning
    if use_morphological:
        print(f'  Applying morphological operations (size={morph_size})...')
        result = morphological_clean(result, kernel_size=morph_size)

    return result

if __name__ == '__main__':
    import argparse
    import os
    from PIL import Image

    parser = argparse.ArgumentParser(description='Post-process predictions to reduce noise')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing predictions.npy and confidence.npy')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--majority_size', type=int, default=3,
                        help='Majority filter window size (default: 3)')
    parser.add_argument('--morph_size', type=int, default=3,
                        help='Morphological kernel size (default: 3)')
    parser.add_argument('--conf_threshold', type=float, default=0.7,
                        help='Confidence threshold (default: 0.7)')
    parser.add_argument('--use_confidence', action='store_true',
                        help='Apply confidence thresholding')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir

    # Load predictions
    pred_path = os.path.join(args.input_dir, 'predictions.npy')
    conf_path = os.path.join(args.input_dir, 'confidence.npy')

    print(f'Loading predictions from {pred_path}...')
    predictions = np.load(pred_path)

    confidence = None
    if os.path.exists(conf_path):
        print(f'Loading confidence from {conf_path}...')
        confidence = np.load(conf_path)

    print(f'\nOriginal prediction shape: {predictions.shape}')
    print(f'Unique classes: {np.unique(predictions)}')

    # Apply filters
    print('\nApplying post-processing filters...')
    filtered = apply_all_filters(
        predictions,
        confidence,
        use_majority=True,
        majority_size=args.majority_size,
        use_morphological=True,
        morph_size=args.morph_size,
        use_confidence=args.use_confidence,
        conf_threshold=args.conf_threshold
    )

    # Save filtered predictions
    output_path = os.path.join(args.output_dir, 'predictions_filtered.npy')
    print(f'\nSaving filtered predictions to {output_path}...')
    np.save(output_path, filtered)

    # Create visualization
    from inference import create_visualization
    viz_path = os.path.join(args.output_dir, 'prediction_filtered_visualization.png')
    create_visualization(filtered, confidence, viz_path)

    print(f'✓ Post-processing complete!')
    print(f'  Filtered predictions: {output_path}')
    print(f'  Visualization: {viz_path}')
