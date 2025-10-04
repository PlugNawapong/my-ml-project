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

def normalize_band_to_max(band):
    """Normalize a single band by scaling to maximum value (255)"""
    max_val = band.max()
    if max_val > 0:
        return (band / max_val * 255.0).astype(np.float32)
    return band

def normalize_band_minmax(band):
    """Normalize a single band using min-max scaling to [0, 255]"""
    min_val = band.min()
    max_val = band.max()
    if max_val > min_val:
        return ((band - min_val) / (max_val - min_val) * 255.0).astype(np.float32)
    return band

def normalize_band_percentile(band, lower=2, upper=98):
    """Normalize using percentile clipping (robust to outliers)"""
    p_low = np.percentile(band, lower)
    p_high = np.percentile(band, upper)
    if p_high > p_low:
        band_clipped = np.clip(band, p_low, p_high)
        return ((band_clipped - p_low) / (p_high - p_low) * 255.0).astype(np.float32)
    return band

def visualize_band_normalization(train_bands, inf1_bands, inf2_bands, output_dir='band_normalization_comparison'):
    """Visualize band-wise normalization methods"""
    os.makedirs(output_dir, exist_ok=True)

    # Key wavelengths to visualize (indices: 0, 5, 10, 15, 20, 25 for 450, 500, 550, 600, 650, 700nm)
    key_wavelengths = [0, 5, 10, 15, 20, 25]
    wavelength_nm = [450, 500, 550, 600, 650, 700]

    print('\n' + '='*80)
    print('RAW BAND STATISTICS (Before Normalization)')
    print('='*80)

    for idx, wl in zip(key_wavelengths, wavelength_nm):
        print(f'\nBand {idx+1} ({wl}nm):')
        print(f'  Training:    min={train_bands[idx].min():.1f}, max={train_bands[idx].max():.1f}, mean={train_bands[idx].mean():.1f}')
        print(f'  Inference 1: min={inf1_bands[idx].min():.1f}, max={inf1_bands[idx].max():.1f}, mean={inf1_bands[idx].mean():.1f}')
        print(f'  Inference 2: min={inf2_bands[idx].min():.1f}, max={inf2_bands[idx].max():.1f}, mean={inf2_bands[idx].mean():.1f}')

    # Method 1: Scale to Maximum (255)
    print('\n\nCreating visualization for: Scale to Maximum (255)...')
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Band Normalization: Scale to Maximum (255)', fontsize=18, fontweight='bold')

    for col_idx, (band_idx, wl) in enumerate(zip(key_wavelengths, wavelength_nm)):
        # Row 1: Raw Training
        ax = plt.subplot(7, 6, col_idx + 1)
        ax.imshow(train_bands[band_idx], cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Raw\nTraining', fontsize=10, fontweight='bold')
        ax.set_title(f'{wl}nm\nmax={train_bands[band_idx].max():.0f}', fontsize=9)
        ax.axis('off')

        # Row 2: Raw Inference 1
        ax = plt.subplot(7, 6, col_idx + 7)
        ax.imshow(inf1_bands[band_idx], cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Raw\nInference 1', fontsize=10, fontweight='bold')
        ax.set_title(f'max={inf1_bands[band_idx].max():.0f}', fontsize=9)
        ax.axis('off')

        # Row 3: Raw Inference 2
        ax = plt.subplot(7, 6, col_idx + 13)
        ax.imshow(inf2_bands[band_idx], cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Raw\nInference 2', fontsize=10, fontweight='bold')
        ax.set_title(f'max={inf2_bands[band_idx].max():.0f}', fontsize=9)
        ax.axis('off')

        # Row 4: Spacer
        ax = plt.subplot(7, 6, col_idx + 19)
        ax.axis('off')
        if col_idx == 0:
            ax.text(0.5, 0.5, '↓ NORMALIZED ↓', fontsize=10, fontweight='bold',
                   ha='center', va='center', transform=ax.transAxes)

        # Row 5: Normalized Training
        ax = plt.subplot(7, 6, col_idx + 25)
        norm_band = normalize_band_to_max(train_bands[band_idx])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Normalized\nTraining', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Row 6: Normalized Inference 1
        ax = plt.subplot(7, 6, col_idx + 31)
        norm_band = normalize_band_to_max(inf1_bands[band_idx])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Normalized\nInference 1', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Row 7: Normalized Inference 2
        ax = plt.subplot(7, 6, col_idx + 37)
        norm_band = normalize_band_to_max(inf2_bands[band_idx])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Normalized\nInference 2', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_scale_to_max.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ Saved method_scale_to_max.png')

    # Method 2: Min-Max Normalization [0, 255]
    print('\nCreating visualization for: Min-Max Normalization [0, 255]...')
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Band Normalization: Min-Max [0, 255]', fontsize=18, fontweight='bold')

    for col_idx, (band_idx, wl) in enumerate(zip(key_wavelengths, wavelength_nm)):
        # Row 1: Raw Training
        ax = plt.subplot(7, 6, col_idx + 1)
        ax.imshow(train_bands[band_idx], cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Raw\nTraining', fontsize=10, fontweight='bold')
        ax.set_title(f'{wl}nm\nrange=[{train_bands[band_idx].min():.0f},{train_bands[band_idx].max():.0f}]', fontsize=8)
        ax.axis('off')

        # Row 2: Raw Inference 1
        ax = plt.subplot(7, 6, col_idx + 7)
        ax.imshow(inf1_bands[band_idx], cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Raw\nInference 1', fontsize=10, fontweight='bold')
        ax.set_title(f'range=[{inf1_bands[band_idx].min():.0f},{inf1_bands[band_idx].max():.0f}]', fontsize=8)
        ax.axis('off')

        # Row 3: Raw Inference 2
        ax = plt.subplot(7, 6, col_idx + 13)
        ax.imshow(inf2_bands[band_idx], cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Raw\nInference 2', fontsize=10, fontweight='bold')
        ax.set_title(f'range=[{inf2_bands[band_idx].min():.0f},{inf2_bands[band_idx].max():.0f}]', fontsize=8)
        ax.axis('off')

        # Row 4: Spacer
        ax = plt.subplot(7, 6, col_idx + 19)
        ax.axis('off')
        if col_idx == 0:
            ax.text(0.5, 0.5, '↓ NORMALIZED ↓', fontsize=10, fontweight='bold',
                   ha='center', va='center', transform=ax.transAxes)

        # Row 5: Normalized Training
        ax = plt.subplot(7, 6, col_idx + 25)
        norm_band = normalize_band_minmax(train_bands[band_idx])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Normalized\nTraining', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Row 6: Normalized Inference 1
        ax = plt.subplot(7, 6, col_idx + 31)
        norm_band = normalize_band_minmax(inf1_bands[band_idx])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Normalized\nInference 1', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Row 7: Normalized Inference 2
        ax = plt.subplot(7, 6, col_idx + 37)
        norm_band = normalize_band_minmax(inf2_bands[band_idx])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Normalized\nInference 2', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_minmax.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ Saved method_minmax.png')

    # Method 3: Percentile Normalization (2-98%)
    print('\nCreating visualization for: Percentile Normalization (2-98%)...')
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Band Normalization: Percentile (2-98%)', fontsize=18, fontweight='bold')

    for col_idx, (band_idx, wl) in enumerate(zip(key_wavelengths, wavelength_nm)):
        # Row 1: Raw Training
        ax = plt.subplot(7, 6, col_idx + 1)
        ax.imshow(train_bands[band_idx], cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Raw\nTraining', fontsize=10, fontweight='bold')
        ax.set_title(f'{wl}nm', fontsize=10)
        ax.axis('off')

        # Row 2: Raw Inference 1
        ax = plt.subplot(7, 6, col_idx + 7)
        ax.imshow(inf1_bands[band_idx], cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Raw\nInference 1', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Row 3: Raw Inference 2
        ax = plt.subplot(7, 6, col_idx + 13)
        ax.imshow(inf2_bands[band_idx], cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Raw\nInference 2', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Row 4: Spacer
        ax = plt.subplot(7, 6, col_idx + 19)
        ax.axis('off')
        if col_idx == 0:
            ax.text(0.5, 0.5, '↓ NORMALIZED ↓', fontsize=10, fontweight='bold',
                   ha='center', va='center', transform=ax.transAxes)

        # Row 5: Normalized Training
        ax = plt.subplot(7, 6, col_idx + 25)
        norm_band = normalize_band_percentile(train_bands[band_idx])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Normalized\nTraining', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Row 6: Normalized Inference 1
        ax = plt.subplot(7, 6, col_idx + 31)
        norm_band = normalize_band_percentile(inf1_bands[band_idx])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Normalized\nInference 1', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Row 7: Normalized Inference 2
        ax = plt.subplot(7, 6, col_idx + 37)
        norm_band = normalize_band_percentile(inf2_bands[band_idx])
        ax.imshow(norm_band, cmap='gray', vmin=0, vmax=255)
        if col_idx == 0:
            ax.set_ylabel('Normalized\nInference 2', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_percentile.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ Saved method_percentile.png')

def main():
    # Paths
    training_dir = 'data'
    inference1_dir = 'inference_data_set1'
    inference2_dir = 'inference_data_set2'
    output_dir = 'band_normalization_comparison'

    print('='*80)
    print('BAND-WISE NORMALIZATION COMPARISON')
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
    visualize_band_normalization(train_bands, inf1_bands, inf2_bands, output_dir)

    print('\n' + '='*80)
    print('COMPARISON COMPLETE!')
    print('='*80)
    print(f'\nResults saved to: {output_dir}/')
    print('\nGenerated files:')
    print('  - method_scale_to_max.png : Scale each band to max value (255)')
    print('  - method_minmax.png : Min-max normalization per band [0, 255]')
    print('  - method_percentile.png : Percentile-based (2-98%) normalization')
    print('\nTotal: 3 comparison plots')

    print('\n' + '='*80)
    print('INTERPRETATION GUIDE')
    print('='*80)
    print('\n1. Scale to Maximum (255):')
    print('   - Brightens each band until the brightest pixel = 255')
    print('   - Preserves relative brightness differences within each band')
    print('   - Good for visual comparison')
    print('\n2. Min-Max [0, 255]:')
    print('   - Stretches each band to full range [0, 255]')
    print('   - Dark pixel becomes 0, bright pixel becomes 255')
    print('   - Maximizes contrast per band')
    print('\n3. Percentile (2-98%):')
    print('   - Clips outliers, then stretches to [0, 255]')
    print('   - Robust to bright/dark outlier pixels')
    print('   - Good for noisy data')

if __name__ == '__main__':
    main()
