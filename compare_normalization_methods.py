import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from robust_normalization import get_normalization

# All available normalization methods
NORM_METHODS = ['snv', 'msc', 'minmax', 'robust', 'vector', 'area', 'max', 'snv+minmax', 'robust+snv']

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

def sample_spectra(bands_data, num_samples=1000):
    """Sample random pixels from the image"""
    num_bands, height, width = bands_data.shape

    # Random sampling
    y_coords = np.random.randint(0, height, num_samples)
    x_coords = np.random.randint(0, width, num_samples)

    spectra = bands_data[:, y_coords, x_coords].T  # Shape: (num_samples, num_bands)
    return spectra

def compare_methods_per_band(training_spectra, inference1_spectra, inference2_spectra, output_dir='normalization_comparison'):
    """Compare normalization methods for each band"""
    os.makedirs(output_dir, exist_ok=True)

    num_bands = training_spectra.shape[1]
    wavelengths = np.arange(450, 450 + num_bands * 10, 10)  # 450-700nm in 10nm steps

    # Create comparison for each band
    for band_idx in range(num_bands):
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Band {band_idx + 1} ({wavelengths[band_idx]}nm) - Normalization Methods Comparison',
                     fontsize=16, fontweight='bold')

        for idx, method in enumerate(NORM_METHODS):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            # Get normalizer
            normalizer = get_normalization(method)

            # Normalize training data
            train_band = training_spectra[:, band_idx].copy()
            train_normalized = np.array([normalizer(np.array([val]))[0] for val in train_band])

            # Normalize inference1 data
            inf1_band = inference1_spectra[:, band_idx].copy()
            inf1_normalized = np.array([normalizer(np.array([val]))[0] for val in inf1_band])

            # Normalize inference2 data
            inf2_band = inference2_spectra[:, band_idx].copy()
            inf2_normalized = np.array([normalizer(np.array([val]))[0] for val in inf2_band])

            # Plot distributions
            ax.hist(train_normalized, bins=50, alpha=0.5, label='Training', color='blue', density=True)
            ax.hist(inf1_normalized, bins=50, alpha=0.5, label='Inference 1', color='green', density=True)
            ax.hist(inf2_normalized, bins=50, alpha=0.5, label='Inference 2', color='red', density=True)

            ax.set_title(f'{method}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Normalized Value')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Add statistics
            train_mean, train_std = train_normalized.mean(), train_normalized.std()
            inf2_mean, inf2_std = inf2_normalized.mean(), inf2_normalized.std()

            textstr = f'Train: μ={train_mean:.2f}, σ={train_std:.2f}\nInf2: μ={inf2_mean:.2f}, σ={inf2_std:.2f}'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'band_{band_idx+1:02d}_{wavelengths[band_idx]}nm.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Saved comparison for band {band_idx + 1} ({wavelengths[band_idx]}nm)')

def compare_methods_all_bands(training_spectra, inference1_spectra, inference2_spectra, output_dir='normalization_comparison'):
    """Compare normalization methods across all bands (full spectrum)"""
    os.makedirs(output_dir, exist_ok=True)

    num_bands = training_spectra.shape[1]
    wavelengths = np.arange(450, 450 + num_bands * 10, 10)

    # Create one plot per normalization method showing all wavelengths
    for method in NORM_METHODS:
        fig, ax = plt.subplots(figsize=(14, 8))

        normalizer = get_normalization(method)

        # Sample a few spectra to plot
        num_plot_samples = 100

        # Normalize and plot training spectra
        for i in range(min(num_plot_samples, len(training_spectra))):
            spectrum = training_spectra[i].copy()
            normalized = normalizer(spectrum)
            ax.plot(wavelengths, normalized, color='blue', alpha=0.05, linewidth=0.5)

        # Normalize and plot inference1 spectra
        for i in range(min(num_plot_samples, len(inference1_spectra))):
            spectrum = inference1_spectra[i].copy()
            normalized = normalizer(spectrum)
            ax.plot(wavelengths, normalized, color='green', alpha=0.05, linewidth=0.5)

        # Normalize and plot inference2 spectra
        for i in range(min(num_plot_samples, len(inference2_spectra))):
            spectrum = inference2_spectra[i].copy()
            normalized = normalizer(spectrum)
            ax.plot(wavelengths, normalized, color='red', alpha=0.05, linewidth=0.5)

        # Plot mean spectra with thicker lines
        train_mean = np.array([normalizer(training_spectra[i]) for i in range(len(training_spectra))]).mean(axis=0)
        inf1_mean = np.array([normalizer(inference1_spectra[i]) for i in range(len(inference1_spectra))]).mean(axis=0)
        inf2_mean = np.array([normalizer(inference2_spectra[i]) for i in range(len(inference2_spectra))]).mean(axis=0)

        ax.plot(wavelengths, train_mean, color='blue', linewidth=3, label='Training (mean)')
        ax.plot(wavelengths, inf1_mean, color='green', linewidth=3, label='Inference 1 (mean)')
        ax.plot(wavelengths, inf2_mean, color='red', linewidth=3, label='Inference 2 (mean)')

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Normalized Intensity', fontsize=12)
        ax.set_title(f'Normalization Method: {method.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spectrum_{method}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Saved full spectrum comparison for {method}')

def visualize_images_per_method(train_bands, inf1_bands, inf2_bands, output_dir='normalization_comparison'):
    """Visualize actual images before and after normalization for key wavelengths"""
    os.makedirs(output_dir, exist_ok=True)

    # Key wavelengths to visualize (indices: 0, 5, 10, 15, 20, 25 for 450, 500, 550, 600, 650, 700nm)
    key_wavelengths = [0, 5, 10, 15, 20, 25]
    wavelength_nm = [450, 500, 550, 600, 650, 700]

    # Pre-normalize all spectra for each method (much faster)
    print(f'  Pre-computing normalized spectra for all methods...')
    normalized_data = {}

    for method in NORM_METHODS:
        print(f'    - {method}...')
        normalizer = get_normalization(method)

        # Normalize each dataset separately (they may have different sizes)
        train_norm = np.zeros_like(train_bands)
        inf1_norm = np.zeros_like(inf1_bands)
        inf2_norm = np.zeros_like(inf2_bands)

        # Normalize training data
        train_height, train_width = train_bands.shape[1], train_bands.shape[2]
        for y in range(train_height):
            for x in range(train_width):
                train_spectrum = train_bands[:, y, x]
                train_norm[:, y, x] = normalizer(train_spectrum)

        # Normalize inference1 data
        inf1_height, inf1_width = inf1_bands.shape[1], inf1_bands.shape[2]
        for y in range(inf1_height):
            for x in range(inf1_width):
                inf1_spectrum = inf1_bands[:, y, x]
                inf1_norm[:, y, x] = normalizer(inf1_spectrum)

        # Normalize inference2 data
        inf2_height, inf2_width = inf2_bands.shape[1], inf2_bands.shape[2]
        for y in range(inf2_height):
            for x in range(inf2_width):
                inf2_spectrum = inf2_bands[:, y, x]
                inf2_norm[:, y, x] = normalizer(inf2_spectrum)

        normalized_data[method] = {
            'train': train_norm,
            'inf1': inf1_norm,
            'inf2': inf2_norm
        }

    # Now create visualizations
    for method in NORM_METHODS:
        print(f'  Creating image visualization for {method}...')

        fig = plt.figure(figsize=(24, 20))
        fig.suptitle(f'Image Visualization: {method.upper()} Normalization', fontsize=18, fontweight='bold')

        # Create grid: 7 rows (3 raw + 3 normalized + 1 spacer) x 6 cols (wavelengths)
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

            # Row 4: Spacer (empty)
            ax = plt.subplot(7, 6, col_idx + 19)
            ax.axis('off')
            if col_idx == 0:
                ax.text(0.5, 0.5, '↓ NORMALIZED ↓', fontsize=10, fontweight='bold',
                       ha='center', va='center', transform=ax.transAxes)

            # Row 5: Normalized Training
            ax = plt.subplot(7, 6, col_idx + 25)
            ax.imshow(normalized_data[method]['train'][band_idx], cmap='gray')
            if col_idx == 0:
                ax.set_ylabel('Normalized\nTraining', fontsize=10, fontweight='bold')
            ax.axis('off')

            # Row 6: Normalized Inference 1
            ax = plt.subplot(7, 6, col_idx + 31)
            ax.imshow(normalized_data[method]['inf1'][band_idx], cmap='gray')
            if col_idx == 0:
                ax.set_ylabel('Normalized\nInference 1', fontsize=10, fontweight='bold')
            ax.axis('off')

            # Row 7: Normalized Inference 2
            ax = plt.subplot(7, 6, col_idx + 37)
            ax.imshow(normalized_data[method]['inf2'][band_idx], cmap='gray')
            if col_idx == 0:
                ax.set_ylabel('Normalized\nInference 2', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'images_{method}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  ✓ Saved image visualization for {method}')

def create_summary_comparison(training_spectra, inference1_spectra, inference2_spectra, output_dir='normalization_comparison'):
    """Create a summary showing all methods in one plot"""
    os.makedirs(output_dir, exist_ok=True)

    num_bands = training_spectra.shape[1]
    wavelengths = np.arange(450, 450 + num_bands * 10, 10)

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Summary: All Normalization Methods (Mean Spectra)', fontsize=18, fontweight='bold')

    for idx, method in enumerate(NORM_METHODS):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        normalizer = get_normalization(method)

        # Calculate mean spectra
        train_mean = np.array([normalizer(training_spectra[i]) for i in range(len(training_spectra))]).mean(axis=0)
        inf1_mean = np.array([normalizer(inference1_spectra[i]) for i in range(len(inference1_spectra))]).mean(axis=0)
        inf2_mean = np.array([normalizer(inference2_spectra[i]) for i in range(len(inference2_spectra))]).mean(axis=0)

        ax.plot(wavelengths, train_mean, color='blue', linewidth=2, label='Training', marker='o', markersize=3)
        ax.plot(wavelengths, inf1_mean, color='green', linewidth=2, label='Inference 1', marker='s', markersize=3)
        ax.plot(wavelengths, inf2_mean, color='red', linewidth=2, label='Inference 2', marker='^', markersize=3)

        ax.set_title(f'{method.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Wavelength (nm)', fontsize=10)
        ax.set_ylabel('Normalized Intensity', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_all_methods.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved summary comparison')

def main():
    # Paths
    training_dir = 'data'
    inference1_dir = 'inference_data_set1'
    inference2_dir = 'inference_data_set2'
    output_dir = 'normalization_comparison'

    print('='*80)
    print('NORMALIZATION METHODS COMPARISON')
    print('='*80)

    # Load data
    print('\nLoading training data...')
    train_bands = load_bands(training_dir)

    print('Loading inference dataset 1...')
    inf1_bands = load_bands(inference1_dir)

    print('Loading inference dataset 2...')
    inf2_bands = load_bands(inference2_dir)

    # Sample spectra
    print('\nSampling spectra...')
    train_spectra = sample_spectra(train_bands, num_samples=5000)
    inf1_spectra = sample_spectra(inf1_bands, num_samples=5000)
    inf2_spectra = sample_spectra(inf2_bands, num_samples=5000)

    print(f'Training spectra shape: {train_spectra.shape}')
    print(f'Inference 1 spectra shape: {inf1_spectra.shape}')
    print(f'Inference 2 spectra shape: {inf2_spectra.shape}')

    # Show raw intensity statistics
    print('\n' + '='*80)
    print('RAW INTENSITY STATISTICS (Before Normalization)')
    print('='*80)
    print(f'Training:    Mean={train_spectra.mean():.2f}, Std={train_spectra.std():.2f}, '
          f'Min={train_spectra.min():.2f}, Max={train_spectra.max():.2f}')
    print(f'Inference 1: Mean={inf1_spectra.mean():.2f}, Std={inf1_spectra.std():.2f}, '
          f'Min={inf1_spectra.min():.2f}, Max={inf1_spectra.max():.2f}')
    print(f'Inference 2: Mean={inf2_spectra.mean():.2f}, Std={inf2_spectra.std():.2f}, '
          f'Min={inf2_spectra.min():.2f}, Max={inf2_spectra.max():.2f}')

    # Create comparisons
    print('\n' + '='*80)
    print('GENERATING COMPARISON PLOTS')
    print('='*80)

    print('\n1. Creating summary comparison (all methods in one plot)...')
    create_summary_comparison(train_spectra, inf1_spectra, inf2_spectra, output_dir)

    print('\n2. Creating image visualizations (before/after normalization)...')
    visualize_images_per_method(train_bands, inf1_bands, inf2_bands, output_dir)

    print('\n3. Creating per-method full spectrum plots...')
    compare_methods_all_bands(train_spectra, inf1_spectra, inf2_spectra, output_dir)

    print('\n4. Creating per-band distribution comparisons...')
    compare_methods_per_band(train_spectra, inf1_spectra, inf2_spectra, output_dir)

    print('\n' + '='*80)
    print('COMPARISON COMPLETE!')
    print('='*80)
    print(f'\nResults saved to: {output_dir}/')
    print('\nGenerated files:')
    print(f'  - summary_all_methods.png : Overview of all methods')
    print(f'  - images_<method>.png : Before/after images for each method (9 files)')
    print(f'  - spectrum_<method>.png : Full spectrum for each method (9 files)')
    print(f'  - band_XX_XXXnm.png : Per-band distributions (26 files)')
    print(f'\nTotal: {1 + 9 + 9 + 26} = 45 comparison plots')

    print('\n' + '='*80)
    print('RECOMMENDATION')
    print('='*80)
    print('\nLook for normalization methods where:')
    print('  1. Training and Inference 2 distributions overlap well')
    print('  2. Mean spectra curves align closely across all wavelengths')
    print('  3. Variance is similar between datasets')
    print('\nRecommended methods to try (in order):')
    print('  1. snv+minmax : SNV removes intensity variations, minmax scales to [0,1]')
    print('  2. robust+snv : Robust scaling + SNV for outlier resistance')
    print('  3. msc : Multiplicative Scatter Correction, standard for spectroscopy')
    print('  4. robust : Percentile-based scaling, handles outliers well')

if __name__ == '__main__':
    main()
