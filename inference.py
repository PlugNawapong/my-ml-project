import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import json

from dataset import HyperspectralPatchDataset, CLASS_NAMES
from model import get_model
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# Color mapping for visualization (same as labels.json)
CLASS_COLORS = {
    0: (0, 0, 0),         # Background
    1: (255, 0, 0),       # 95PU
    2: (0, 0, 255),       # HIPS
    3: (255, 125, 125),   # HVDF-HFP
    4: (255, 255, 0),     # GPSS
    5: (0, 125, 125),     # PU
    6: (0, 200, 255),     # 75PU
    7: (255, 0, 255),     # 85PU
    8: (0, 255, 0),       # PETE
    9: (255, 125, 0),     # PET
    10: (255, 0, 100)     # PMMA
}


def load_model(checkpoint_path, model_name, num_bands, num_classes, patch_size, device):
    """Load trained model from checkpoint"""
    model = get_model(model_name, num_bands, num_classes, patch_size)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def predict_image(model, data_dir, device, use_patches=True, patch_size=3, num_bands=26, batch_size=256, norm_method='snv+minmax'):
    """Predict material classes for an entire image"""

    # Load dataset for inference
    if use_patches:
        dataset = HyperspectralPatchDataset(
            data_dir,
            patch_size=patch_size,
            num_bands=num_bands,
            transform=None,
            is_training=False,
            norm_method=norm_method
        )
    else:
        from dataset import HyperspectralDataset
        dataset = HyperspectralDataset(
            data_dir,
            num_bands=num_bands,
            transform=None,
            is_training=False,
            norm_method=norm_method
        )

    # Create prediction map
    height, width = dataset.height, dataset.width
    prediction_map = np.zeros((height, width), dtype=np.uint8)
    confidence_map = np.zeros((height, width), dtype=np.float32)

    # Batch processing
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    with torch.no_grad():
        for inputs, positions in tqdm(dataloader, desc='Predicting'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predictions = torch.max(probs, dim=1)

            # Place predictions in map
            for i in range(len(predictions)):
                y, x = positions[0][i].item(), positions[1][i].item()
                prediction_map[y, x] = predictions[i].cpu().numpy()
                confidence_map[y, x] = max_probs[i].cpu().numpy()

    return prediction_map, confidence_map


def create_visualization(prediction_map, confidence_map=None, save_path=None):
    """Create RGB visualization of predictions"""
    height, width = prediction_map.shape
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx, color in CLASS_COLORS.items():
        mask = prediction_map == class_idx
        vis_img[mask] = color

    if save_path:
        Image.fromarray(vis_img).save(save_path)
        print(f'Visualization saved to: {save_path}')

    return vis_img


def calculate_statistics(prediction_map, confidence_map):
    """Calculate prediction statistics"""
    stats = {
        'total_pixels': prediction_map.size,
        'class_distribution': {},
        'mean_confidence': float(np.mean(confidence_map)),
        'min_confidence': float(np.min(confidence_map)),
        'max_confidence': float(np.max(confidence_map))
    }

    for class_idx in range(len(CLASS_NAMES)):
        count = np.sum(prediction_map == class_idx)
        percentage = 100 * count / prediction_map.size
        mean_conf = float(np.mean(confidence_map[prediction_map == class_idx])) if count > 0 else 0.0

        stats['class_distribution'][CLASS_NAMES[class_idx]] = {
            'count': int(count),
            'percentage': float(percentage),
            'mean_confidence': mean_conf
        }

    return stats


def visualize_normalization_check(data_dir, output_dir, dataset_name='data'):
    """Visualize bands before and after normalization for verification"""
    bands_dir = os.path.join(data_dir, 'bands')
    band_files = sorted([f for f in os.listdir(bands_dir) if f.endswith('.png')])[:26]

    # Load bands
    bands_raw = []
    bands_norm = []

    for band_file in band_files:
        band_path = os.path.join(bands_dir, band_file)
        band_img = np.array(Image.open(band_path).convert('L'), dtype=np.float32)
        bands_raw.append(band_img)

        # Apply percentile normalization
        p_low = np.percentile(band_img, 2)
        p_high = np.percentile(band_img, 98)
        if p_high > p_low:
            band_norm = np.clip(band_img, p_low, p_high)
            band_norm = (band_norm - p_low) / (p_high - p_low) * 255.0
        else:
            band_norm = band_img
        bands_norm.append(band_norm)

    # Create visualization
    fig, axes = plt.subplots(2, 26, figsize=(26, 4))
    wavelengths = np.arange(450, 710, 10)

    for i in range(26):
        # Raw
        axes[0, i].imshow(bands_raw[i], cmap='gray', vmin=0, vmax=255)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'{wavelengths[i]}nm', fontsize=6)

        # Normalized
        axes[1, i].imshow(bands_norm[i], cmap='gray', vmin=0, vmax=255)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Raw', fontsize=8, fontweight='bold')
    axes[1, 0].set_ylabel('Normalized', fontsize=8, fontweight='bold')

    plt.suptitle(f'{dataset_name}: Percentile Normalization Check', fontsize=12, fontweight='bold')
    plt.tight_layout()

    viz_path = os.path.join(output_dir, f'normalization_check_{dataset_name}.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  ✓ Saved normalization check: {viz_path}')

    # Print statistics
    raw_mean = np.mean([b.mean() for b in bands_raw])
    norm_mean = np.mean([b.mean() for b in bands_norm])
    print(f'  Raw mean intensity: {raw_mean:.1f}, Normalized mean: {norm_mean:.1f}')


def main(args):
    # Device - prioritize MPS for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load model
    print(f'Loading model from: {args.checkpoint}')
    model = load_model(
        args.checkpoint,
        args.model,
        args.num_bands,
        args.num_classes,
        args.patch_size,
        device
    )
    print('Model loaded successfully')

    # Process single image or multiple images
    data_dirs = []
    if os.path.isdir(args.data_dir):
        # Check if it's a single dataset or contains multiple datasets
        if os.path.exists(os.path.join(args.data_dir, 'bands')):
            data_dirs.append(args.data_dir)
        else:
            # Look for subdirectories with bands
            for subdir in sorted(os.listdir(args.data_dir)):
                subdir_path = os.path.join(args.data_dir, subdir)
                if os.path.isdir(subdir_path) and os.path.exists(os.path.join(subdir_path, 'bands')):
                    data_dirs.append(subdir_path)

    if not data_dirs:
        data_dirs = [args.data_dir]

    print(f'Found {len(data_dirs)} dataset(s) to process')

    # Process each dataset
    for data_dir in data_dirs:
        dataset_name = os.path.basename(data_dir)
        print(f'\nProcessing: {dataset_name}')

        # Visualize normalization before inference
        if args.norm_method == 'percentile':
            print('\n' + '='*80)
            print('NORMALIZATION CHECK')
            print('='*80)
            print(f'\nVisualizing normalization for {dataset_name}...')
            visualize_normalization_check(data_dir, args.output_dir, dataset_name)
            print('='*80 + '\n')

        # Predict
        prediction_map, confidence_map = predict_image(
            model,
            data_dir,
            device,
            use_patches=args.use_patches,
            patch_size=args.patch_size,
            num_bands=args.num_bands,
            batch_size=args.batch_size,
            norm_method=args.norm_method
        )

        # Create output directory
        output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save raw predictions
        np.save(os.path.join(output_dir, 'predictions.npy'), prediction_map)
        np.save(os.path.join(output_dir, 'confidence.npy'), confidence_map)

        # Create and save visualization
        vis_path = os.path.join(output_dir, 'prediction_visualization.png')
        create_visualization(prediction_map, confidence_map, vis_path)

        # Calculate and save statistics
        stats = calculate_statistics(prediction_map, confidence_map)
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=4)

        print(f'Results saved to: {output_dir}')
        print(f'Mean confidence: {stats["mean_confidence"]:.4f}')
        print('\nClass distribution:')
        for class_name, class_stats in stats['class_distribution'].items():
            if class_stats['percentage'] > 0:
                print(f'  {class_name}: {class_stats["percentage"]:.2f}% ({class_stats["count"]} pixels, '
                      f'conf: {class_stats["mean_confidence"]:.4f})')

    print('\nInference complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for hyperspectral material classifier')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='spectral_cnn_2d',
                        choices=['spectral_cnn_1d', 'spectral_cnn_2d', 'hybrid', 'resnet'],
                        help='Model architecture')
    parser.add_argument('--num_bands', type=int, default=26,
                        help='Number of spectral bands')
    parser.add_argument('--num_classes', type=int, default=11,
                        help='Number of material classes')
    parser.add_argument('--use_patches', action='store_true',
                        help='Use spatial patches (required for 2D models)')
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Spatial patch size')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to inference data directory (single dataset or parent directory)')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Output directory for predictions')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for inference')
    parser.add_argument('--norm_method', type=str, default='percentile',
                        choices=['percentile', 'standard'],
                        help='Normalization method (must match training, default: percentile)')

    args = parser.parse_args()

    # Validate arguments
    if args.model != 'spectral_cnn_1d' and not args.use_patches:
        print('Warning: 2D models require --use_patches flag. Setting it automatically.')
        args.use_patches = True

    main(args)
