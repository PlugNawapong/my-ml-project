import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_bands(data_dir, num_bands=26):
    """Load all spectral bands"""
    bands_dir = os.path.join(data_dir, 'bands')
    band_files = sorted([f for f in os.listdir(bands_dir) if f.endswith('.png')])

    bands_data = []
    for band_file in band_files[:num_bands]:
        band_path = os.path.join(bands_dir, band_file)
        band = np.array(Image.open(band_path).convert('L'), dtype=np.float32)
        bands_data.append(band)

    return np.array(bands_data)  # Shape: (num_bands, height, width)

def normalize_band_percentile(band, lower=2, upper=98):
    """Apply percentile normalization to a single band"""
    p_low = np.percentile(band, lower)
    p_high = np.percentile(band, upper)
    if p_high > p_low:
        band_clipped = np.clip(band, p_low, p_high)
        return ((band_clipped - p_low) / (p_high - p_low) * 255.0).astype(np.float32)
    return band

def visualize_all_bands(train_bands, inf1_bands, inf2_bands, output_dir='normalization_check'):
    """Visualize all bands before and after normalization"""
    os.makedirs(output_dir, exist_ok=True)

    num_bands = train_bands.shape[0]
    wavelengths = np.arange(450, 450 + num_bands * 10, 10)

    # Create visualization for training data
    print('\nCreating visualization for Training data...')
    fig = plt.figure(figsize=(28, 18))
    fig.suptitle('Training Data: All Bands Before and After Percentile Normalization', fontsize=20, fontweight='bold')

    for i in range(num_bands):
        # Before normalization
        ax = plt.subplot(4, num_bands, i + 1)
        ax.imshow(train_bands[i], cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Raw', fontsize=10, fontweight='bold')
        ax.set_title(f'{wavelengths[i]}nm', fontsize=8)
        ax.axis('off')

        # After normalization
        ax = plt.subplot(4, num_bands, i + 1 + num_bands)
        norm_band = normalize_band_percentile(train_bands[i])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Normalized', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Statistics - before
        ax = plt.subplot(4, num_bands, i + 1 + 2*num_bands)
        stats_before = f'min={train_bands[i].min():.0f}\nmax={train_bands[i].max():.0f}\nmean={train_bands[i].mean():.0f}'
        ax.text(0.5, 0.5, stats_before, ha='center', va='center', fontsize=7, transform=ax.transAxes)
        if i == 0:
            ax.set_ylabel('Raw Stats', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Statistics - after
        ax = plt.subplot(4, num_bands, i + 1 + 3*num_bands)
        stats_after = f'min={norm_band.min():.0f}\nmax={norm_band.max():.0f}\nmean={norm_band.mean():.0f}'
        ax.text(0.5, 0.5, stats_after, ha='center', va='center', fontsize=7, transform=ax.transAxes)
        if i == 0:
            ax.set_ylabel('Norm Stats', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_all_bands.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved training_all_bands.png')

    # Create visualization for inference dataset 1
    print('\nCreating visualization for Inference Dataset 1...')
    fig = plt.figure(figsize=(28, 18))
    fig.suptitle('Inference Dataset 1: All Bands Before and After Percentile Normalization', fontsize=20, fontweight='bold')

    for i in range(num_bands):
        # Before normalization
        ax = plt.subplot(4, num_bands, i + 1)
        ax.imshow(inf1_bands[i], cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Raw', fontsize=10, fontweight='bold')
        ax.set_title(f'{wavelengths[i]}nm', fontsize=8)
        ax.axis('off')

        # After normalization
        ax = plt.subplot(4, num_bands, i + 1 + num_bands)
        norm_band = normalize_band_percentile(inf1_bands[i])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Normalized', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Statistics - before
        ax = plt.subplot(4, num_bands, i + 1 + 2*num_bands)
        stats_before = f'min={inf1_bands[i].min():.0f}\nmax={inf1_bands[i].max():.0f}\nmean={inf1_bands[i].mean():.0f}'
        ax.text(0.5, 0.5, stats_before, ha='center', va='center', fontsize=7, transform=ax.transAxes)
        if i == 0:
            ax.set_ylabel('Raw Stats', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Statistics - after
        ax = plt.subplot(4, num_bands, i + 1 + 3*num_bands)
        stats_after = f'min={norm_band.min():.0f}\nmax={norm_band.max():.0f}\nmean={norm_band.mean():.0f}'
        ax.text(0.5, 0.5, stats_after, ha='center', va='center', fontsize=7, transform=ax.transAxes)
        if i == 0:
            ax.set_ylabel('Norm Stats', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference1_all_bands.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved inference1_all_bands.png')

    # Create visualization for inference dataset 2
    print('\nCreating visualization for Inference Dataset 2...')
    fig = plt.figure(figsize=(28, 18))
    fig.suptitle('Inference Dataset 2: All Bands Before and After Percentile Normalization', fontsize=20, fontweight='bold')

    for i in range(num_bands):
        # Before normalization
        ax = plt.subplot(4, num_bands, i + 1)
        ax.imshow(inf2_bands[i], cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Raw', fontsize=10, fontweight='bold')
        ax.set_title(f'{wavelengths[i]}nm', fontsize=8)
        ax.axis('off')

        # After normalization
        ax = plt.subplot(4, num_bands, i + 1 + num_bands)
        norm_band = normalize_band_percentile(inf2_bands[i])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Normalized', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Statistics - before
        ax = plt.subplot(4, num_bands, i + 1 + 2*num_bands)
        stats_before = f'min={inf2_bands[i].min():.0f}\nmax={inf2_bands[i].max():.0f}\nmean={inf2_bands[i].mean():.0f}'
        ax.text(0.5, 0.5, stats_before, ha='center', va='center', fontsize=7, transform=ax.transAxes)
        if i == 0:
            ax.set_ylabel('Raw Stats', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Statistics - after
        ax = plt.subplot(4, num_bands, i + 1 + 3*num_bands)
        stats_after = f'min={norm_band.min():.0f}\nmax={norm_band.max():.0f}\nmean={norm_band.mean():.0f}'
        ax.text(0.5, 0.5, stats_after, ha='center', va='center', fontsize=7, transform=ax.transAxes)
        if i == 0:
            ax.set_ylabel('Norm Stats', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference2_all_bands.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved inference2_all_bands.png')

    # Create comparison across datasets
    print('\nCreating side-by-side comparison...')
    fig = plt.figure(figsize=(28, 24))
    fig.suptitle('Comparison: All Datasets and Bands (Normalized)', fontsize=20, fontweight='bold')

    for i in range(num_bands):
        # Training - normalized
        ax = plt.subplot(3, num_bands, i + 1)
        norm_band = normalize_band_percentile(train_bands[i])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Training\n(Normalized)', fontsize=10, fontweight='bold')
        ax.set_title(f'{wavelengths[i]}nm', fontsize=8)
        ax.axis('off')

        # Inference 1 - normalized
        ax = plt.subplot(3, num_bands, i + 1 + num_bands)
        norm_band = normalize_band_percentile(inf1_bands[i])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Inference 1\n(Normalized)', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Inference 2 - normalized
        ax = plt.subplot(3, num_bands, i + 1 + 2*num_bands)
        norm_band = normalize_band_percentile(inf2_bands[i])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Inference 2\n(Normalized)', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_all_datasets.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Saved comparison_all_datasets.png')

    # Print summary statistics
    print('\n' + '='*80)
    print('SUMMARY STATISTICS')
    print('='*80)

    print('\nRAW DATA (Before Normalization):')
    print(f'Training:    min={train_bands.min():.1f}, max={train_bands.max():.1f}, mean={train_bands.mean():.1f}')
    print(f'Inference 1: min={inf1_bands.min():.1f}, max={inf1_bands.max():.1f}, mean={inf1_bands.mean():.1f}')
    print(f'Inference 2: min={inf2_bands.min():.1f}, max={inf2_bands.max():.1f}, mean={inf2_bands.mean():.1f}')

    print('\nNORMALIZED DATA (After Percentile Normalization):')
    train_norm = np.array([normalize_band_percentile(train_bands[i]) for i in range(num_bands)])
    inf1_norm = np.array([normalize_band_percentile(inf1_bands[i]) for i in range(num_bands)])
    inf2_norm = np.array([normalize_band_percentile(inf2_bands[i]) for i in range(num_bands)])

    print(f'Training:    min={train_norm.min():.1f}, max={train_norm.max():.1f}, mean={train_norm.mean():.1f}')
    print(f'Inference 1: min={inf1_norm.min():.1f}, max={inf1_norm.max():.1f}, mean={inf1_norm.mean():.1f}')
    print(f'Inference 2: min={inf2_norm.min():.1f}, max={inf2_norm.max():.1f}, mean={inf2_norm.mean():.1f}')

def main():
    # Paths
    training_dir = 'data'
    inference1_dir = 'inference_data_set1'
    inference2_dir = 'inference_data_set2'
    output_dir = 'normalization_check'

    print('='*80)
    print('ALL BANDS NORMALIZATION VISUALIZATION')
    print('='*80)

    # Load data
    print('\nLoading training data...')
    train_bands = load_bands(training_dir)

    print('Loading inference dataset 1...')
    inf1_bands = load_bands(inference1_dir)

    print('Loading inference dataset 2...')
    inf2_bands = load_bands(inference2_dir)

    print(f'\nTraining shape: {train_bands.shape}')
    print(f'Inference 1 shape: {inf1_bands.shape}')
    print(f'Inference 2 shape: {inf2_bands.shape}')

    # Create visualizations
    visualize_all_bands(train_bands, inf1_bands, inf2_bands, output_dir)

    print('\n' + '='*80)
    print('VISUALIZATION COMPLETE!')
    print('='*80)
    print(f'\nResults saved to: {output_dir}/')
    print('\nGenerated files:')
    print('  1. training_all_bands.png     : Training data - all 26 bands before/after')
    print('  2. inference1_all_bands.png   : Inference 1 - all 26 bands before/after')
    print('  3. inference2_all_bands.png   : Inference 2 - all 26 bands before/after')
    print('  4. comparison_all_datasets.png: Side-by-side comparison (normalized only)')
    print('\nTotal: 4 visualization files')

    print('\n' + '='*80)
    print('WHAT TO CHECK:')
    print('='*80)
    print('\n1. Raw images should show intensity difference (Inference 2 darker)')
    print('2. Normalized images should have similar brightness across all datasets')
    print('3. Check that features/patterns are preserved after normalization')
    print('4. Verify no artifacts or distortions introduced by normalization')

if __name__ == '__main__':
    main()
