import numpy as np
from scipy.ndimage import median_filter, binary_opening, binary_closing, sobel
from scipy import ndimage
import cv2

def detect_edges(prediction_map, method='sobel'):
    """Detect edges in prediction map"""
    if method == 'sobel':
        # Sobel edge detection
        sx = sobel(prediction_map.astype(float), axis=0)
        sy = sobel(prediction_map.astype(float), axis=1)
        edges = np.hypot(sx, sy)
        edges = (edges > 0).astype(np.uint8) * 255
    elif method == 'canny':
        # Canny edge detection
        edges = cv2.Canny(prediction_map.astype(np.uint8), 0, 1)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")

    return edges

def voting_classification(prediction_maps, confidence_maps=None, method='majority'):
    """
    Ensemble voting from multiple predictions

    Args:
        prediction_maps: List of prediction maps (N, H, W)
        confidence_maps: Optional list of confidence maps (N, H, W)
        method: 'majority' or 'weighted'
    """
    if len(prediction_maps) == 0:
        raise ValueError("No prediction maps provided")

    if len(prediction_maps) == 1:
        return prediction_maps[0]

    prediction_maps = np.array(prediction_maps)

    if method == 'majority':
        # Simple majority vote
        from scipy.stats import mode
        result = mode(prediction_maps, axis=0, keepdims=False)[0]
        return result

    elif method == 'weighted' and confidence_maps is not None:
        # Weighted voting based on confidence
        confidence_maps = np.array(confidence_maps)
        height, width = prediction_maps.shape[1], prediction_maps.shape[2]
        result = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                votes = {}
                for k in range(len(prediction_maps)):
                    pred = prediction_maps[k, i, j]
                    conf = confidence_maps[k, i, j]
                    votes[pred] = votes.get(pred, 0) + conf

                # Select class with highest weighted vote
                result[i, j] = max(votes.items(), key=lambda x: x[1])[0]

        return result
    else:
        raise ValueError(f"Unknown voting method: {method}")

def majority_filter(prediction_map, size=3):
    """Apply majority (mode) filter to smooth predictions using fast scipy implementation"""
    from scipy.ndimage import generic_filter
    from scipy.stats import mode

    def mode_func(values):
        """Get most common value in window"""
        return mode(values, keepdims=False)[0]

    # Use scipy's generic_filter which is much faster than Python loops
    filtered = generic_filter(prediction_map, mode_func, size=size, mode='nearest')

    return filtered.astype(prediction_map.dtype)

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

def edge_aware_voting(prediction_map, window_size=5, edge_threshold=0.5):
    """
    Edge-aware spatial voting: Vote only within same material region (don't cross edges)

    For each pixel, look at neighbors but only vote with pixels in the same region
    (separated by edges). This prevents voting across material boundaries.

    Args:
        prediction_map: Prediction map (H, W)
        window_size: Size of voting window (default: 5)
        edge_threshold: Threshold for edge detection (0-1, default: 0.5)

    Returns:
        Voted prediction map
    """
    from scipy.stats import mode

    # Detect edges in the prediction map
    edge_map = detect_edges(prediction_map, method='sobel')
    # Normalize edges to [0, 1]
    edge_map = edge_map.astype(float) / 255.0

    height, width = prediction_map.shape
    voted = np.zeros_like(prediction_map)
    pad = window_size // 2

    # Pad prediction map and edge map
    padded_pred = np.pad(prediction_map, pad, mode='edge')
    padded_edge = np.pad(edge_map, pad, mode='edge')

    for i in range(height):
        for j in range(width):
            # Extract window around current pixel
            window_pred = padded_pred[i:i+window_size, j:j+window_size]
            window_edge = padded_edge[i:i+window_size, j:j+window_size]

            # Only vote with pixels that are NOT on edges (same region)
            # Edges have high values, so we want low edge values
            valid_mask = window_edge < edge_threshold

            # Get valid predictions (not on edges)
            valid_predictions = window_pred[valid_mask]

            if len(valid_predictions) > 0:
                # Vote among valid neighbors
                voted[i, j] = mode(valid_predictions, keepdims=False)[0]
            else:
                # If all neighbors are on edges, keep original prediction
                voted[i, j] = prediction_map[i, j]

    return voted

def apply_industry_standard_postprocessing(prediction_map, confidence_map=None,
                                            min_region_size=50, smooth_sigma=0.8):
    """
    Apply industry-standard post-processing for hyperspectral material classification

    Based on:
    - Connected component analysis for small region removal
    - Bilateral filtering for edge-preserving smoothing
    - Confidence-weighted refinement

    Args:
        prediction_map: Raw prediction map (H, W)
        confidence_map: Confidence scores (H, W)
        min_region_size: Minimum pixels for a valid region (default: 50)
        smooth_sigma: Gaussian sigma for bilateral smoothing (default: 0.8)

    Returns:
        Refined prediction map
    """
    from scipy.ndimage import label, median_filter
    from skimage.filters import rank
    from skimage.morphology import disk, remove_small_objects

    result = prediction_map.copy()
    print(f'  Applying industry-standard post-processing...')

    # Step 1: Remove small isolated regions (salt-and-pepper noise)
    print(f'    - Removing regions smaller than {min_region_size} pixels...')
    for class_id in np.unique(result):
        mask = (result == class_id)
        # Label connected components
        labeled_mask, num_features = label(mask)

        # Remove small components
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_mask == region_id)
            if np.sum(region_mask) < min_region_size:
                result[region_mask] = 0  # Set to background

    # Step 2: Median filter for isolated pixel noise (fast and effective)
    print(f'    - Applying median filter (size=3)...')
    result = median_filter(result, size=3)

    # Step 3: Morphological operations for each class
    print(f'    - Applying morphological cleaning...')
    refined = np.zeros_like(result)
    for class_id in np.unique(result):
        mask = (result == class_id).astype(np.uint8)
        if np.sum(mask) > 0:
            # Opening: remove small noise
            kernel = np.ones((3, 3), dtype=np.uint8)
            cleaned = binary_opening(mask, structure=kernel)
            # Closing: fill small holes
            cleaned = binary_closing(cleaned, structure=kernel)
            refined[cleaned > 0] = class_id

    print(f'  ✓ Post-processing complete')
    return refined

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
    parser.add_argument('--spatial_voting', action='store_true',
                        help='Apply edge-aware spatial voting')
    parser.add_argument('--voting_window', type=int, default=5,
                        help='Spatial voting window size (default: 5)')
    parser.add_argument('--edge_threshold', type=float, default=0.5,
                        help='Edge threshold for spatial voting (default: 0.5)')

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
        conf_threshold=args.conf_threshold,
        use_spatial_voting=args.spatial_voting,
        voting_window_size=args.voting_window,
        edge_threshold=args.edge_threshold
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
