import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_bands(data_dir):
    """Load all spectral bands from a directory"""
    bands_dir = os.path.join(data_dir, 'bands')
    band_files = sorted([f for f in os.listdir(bands_dir) if f.endswith('.png')])

    bands = []
    wavelengths = []
    for band_file in band_files:
        band_path = os.path.join(bands_dir, band_file)
        band_img = np.array(Image.open(band_path), dtype=np.float32) / 255.0
        bands.append(band_img)

        # Extract wavelength
        wl = band_file.split('_')[-1].replace('nm.png', '')
        wavelengths.append(int(wl))

    return np.stack(bands, axis=0), np.array(wavelengths)

def compare_datasets(dir1, dir2, name1="Dataset 1", name2="Dataset 2"):
    """Compare two datasets"""
    print(f"\n{'='*80}")
    print(f"COMPARING: {name1} vs {name2}")
    print(f"{'='*80}\n")

    # Load both datasets
    bands1, wavelengths = load_bands(dir1)
    bands2, _ = load_bands(dir2)

    print(f"{name1} shape: {bands1.shape}")
    print(f"{name2} shape: {bands2.shape}")

    # Sample random pixels from each dataset
    n_samples = 10000
    h1, w1 = bands1.shape[1], bands1.shape[2]
    h2, w2 = bands2.shape[1], bands2.shape[2]

    # Random sampling
    idx1_y = np.random.randint(0, h1, n_samples)
    idx1_x = np.random.randint(0, w1, n_samples)
    spectra1 = bands1[:, idx1_y, idx1_x].T  # (n_samples, n_bands)

    idx2_y = np.random.randint(0, h2, n_samples)
    idx2_x = np.random.randint(0, w2, n_samples)
    spectra2 = bands2[:, idx2_y, idx2_x].T  # (n_samples, n_bands)

    # Statistics
    print(f"\n=== Statistics Comparison ===")
    print(f"\n{name1}:")
    print(f"  Mean: {bands1.mean():.6f}")
    print(f"  Std:  {bands1.std():.6f}")
    print(f"  Min:  {bands1.min():.6f}")
    print(f"  Max:  {bands1.max():.6f}")

    print(f"\n{name2}:")
    print(f"  Mean: {bands2.mean():.6f}")
    print(f"  Std:  {bands2.std():.6f}")
    print(f"  Min:  {bands2.min():.6f}")
    print(f"  Max:  {bands2.max():.6f}")

    # Per-band statistics
    print(f"\n=== Per-Band Mean Comparison ===")
    mean1_per_band = bands1.mean(axis=(1, 2))
    mean2_per_band = bands2.mean(axis=(1, 2))

    print(f"{'Wavelength':<12} {name1:<15} {name2:<15} {'Difference':<15}")
    print("-" * 60)
    for i, wl in enumerate(wavelengths):
        diff = mean2_per_band[i] - mean1_per_band[i]
        print(f"{wl}nm{'':<8} {mean1_per_band[i]:.6f}{'':<8} {mean2_per_band[i]:.6f}{'':<8} {diff:+.6f}")

    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Mean spectral signatures
    ax = axes[0, 0]
    ax.plot(wavelengths, mean1_per_band, 'b-', linewidth=2, label=name1)
    ax.plot(wavelengths, mean2_per_band, 'r-', linewidth=2, label=name2)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Mean Reflectance')
    ax.set_title('Mean Spectral Signatures')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Difference plot
    ax = axes[0, 1]
    diff_per_band = mean2_per_band - mean1_per_band
    ax.plot(wavelengths, diff_per_band, 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Difference (Dataset2 - Dataset1)')
    ax.set_title('Spectral Difference')
    ax.grid(True, alpha=0.3)

    # 3. Distribution histograms
    ax = axes[0, 2]
    ax.hist(spectra1.flatten(), bins=50, alpha=0.5, label=name1, density=True)
    ax.hist(spectra2.flatten(), bins=50, alpha=0.5, label=name2, density=True)
    ax.set_xlabel('Reflectance Value')
    ax.set_ylabel('Density')
    ax.set_title('Reflectance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Standard deviation per band
    ax = axes[1, 0]
    std1_per_band = bands1.std(axis=(1, 2))
    std2_per_band = bands2.std(axis=(1, 2))
    ax.plot(wavelengths, std1_per_band, 'b-', linewidth=2, label=name1)
    ax.plot(wavelengths, std2_per_band, 'r-', linewidth=2, label=name2)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Spectral Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Sample spectra overlay (normalized)
    ax = axes[1, 1]
    for i in range(min(50, n_samples)):
        spec1 = spectra1[i]
        spec2 = spectra2[i]
        # Normalize each spectrum
        spec1_norm = (spec1 - spec1.mean()) / (spec1.std() + 1e-6)
        spec2_norm = (spec2 - spec2.mean()) / (spec2.std() + 1e-6)
        ax.plot(wavelengths, spec1_norm, 'b-', alpha=0.1)
        ax.plot(wavelengths, spec2_norm, 'r-', alpha=0.1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized Reflectance')
    ax.set_title('Normalized Spectral Samples (50 each)')
    ax.grid(True, alpha=0.3)

    # 6. Band correlation
    ax = axes[1, 2]
    # Compute correlation between corresponding bands
    correlations = []
    for i in range(len(wavelengths)):
        band1_flat = bands1[i].flatten()
        band2_flat = bands2[i].flatten()
        # Subsample for faster computation
        subsample = np.random.choice(len(band1_flat), min(10000, len(band1_flat)), replace=False)
        corr = np.corrcoef(band1_flat[subsample], band2_flat[subsample])[0, 1]
        correlations.append(corr)

    ax.plot(wavelengths, correlations, 'purple', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Inter-Dataset Band Correlation')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: dataset_comparison.png")
    plt.close()

    # Assess domain shift severity
    print(f"\n=== Domain Shift Assessment ===")
    mean_diff = np.abs(diff_per_band).mean()
    std_diff = np.abs(std2_per_band - std1_per_band).mean()
    mean_corr = np.mean(correlations)

    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Std deviation difference: {std_diff:.6f}")
    print(f"Mean band correlation: {mean_corr:.4f}")

    if mean_diff > 0.1:
        print("\n⚠ WARNING: Large mean difference detected!")
        print("   → Datasets have significantly different brightness/exposure")
        print("   → Recommendation: Use stronger normalization or data augmentation")

    if std_diff > 0.05:
        print("\n⚠ WARNING: Large variance difference detected!")
        print("   → Dataset variability differs significantly")
        print("   → Recommendation: Add noise augmentation during training")

    if mean_corr < 0.7:
        print("\n⚠ WARNING: Low correlation between datasets!")
        print("   → Datasets may have different acquisition conditions")
        print("   → Recommendation: Consider domain adaptation techniques")

    if mean_diff < 0.05 and std_diff < 0.02 and mean_corr > 0.9:
        print("\n✓ Datasets are similar - model should generalize well")
        print("   → If performance is still poor, consider:")
        print("     1. Training longer")
        print("     2. Using more training samples")
        print("     3. Reducing model complexity")

    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'mean_corr': mean_corr,
        'spectra1': spectra1,
        'spectra2': spectra2,
        'wavelengths': wavelengths
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare two hyperspectral datasets')
    parser.add_argument('--dir1', type=str, default='data',
                        help='First dataset directory')
    parser.add_argument('--dir2', type=str, default='inference_data_set2',
                        help='Second dataset directory')
    parser.add_argument('--name1', type=str, default='Training Data',
                        help='Name for first dataset')
    parser.add_argument('--name2', type=str, default='Inference Data Set 2',
                        help='Name for second dataset')

    args = parser.parse_args()

    if not os.path.exists(args.dir1):
        print(f"Error: {args.dir1} not found")
        exit(1)

    if not os.path.exists(args.dir2):
        print(f"Error: {args.dir2} not found")
        exit(1)

    results = compare_datasets(args.dir1, args.dir2, args.name1, args.name2)

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}\n")
