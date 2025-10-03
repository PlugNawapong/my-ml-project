import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

from dataset import HyperspectralDataset, HyperspectralPatchDataset, CLASS_NAMES, CLASS_MAPPING


def visualize_bands(data_dir, num_bands_to_show=6):
    """Visualize sample spectral bands"""
    bands_dir = os.path.join(data_dir, 'bands')
    band_files = sorted([f for f in os.listdir(bands_dir) if f.endswith('.png')])

    print(f"\n=== Spectral Bands ===")
    print(f"Total bands: {len(band_files)}")
    print(f"Band files: {band_files[:3]}...{band_files[-1]}")

    # Load first band to get dimensions
    first_band = Image.open(os.path.join(bands_dir, band_files[0]))
    height, width = np.array(first_band).shape
    print(f"Image dimensions: {height} x {width} pixels")
    print(f"Total pixels: {height * width:,}")

    # Visualize subset of bands
    indices = np.linspace(0, len(band_files)-1, num_bands_to_show, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        band_path = os.path.join(bands_dir, band_files[idx])
        band_img = np.array(Image.open(band_path))

        # Extract wavelength from filename
        wavelength = band_files[idx].split('_')[-1].replace('.png', '')

        im = ax.imshow(band_img, cmap='gray')
        ax.set_title(f'Band {idx}: {wavelength}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig('data_inspection_bands.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved band visualization to: data_inspection_bands.png")
    plt.close()


def visualize_labels(data_dir):
    """Visualize labels and class distribution"""
    labels_dir = os.path.join(data_dir, 'labels')
    label_path = os.path.join(labels_dir, 'labels.png')

    if not os.path.exists(label_path):
        print(f"\n⚠ No labels found at {label_path}")
        return None, None

    print(f"\n=== Labels ===")
    label_img = np.array(Image.open(label_path))
    print(f"Label image shape: {label_img.shape}")

    # Convert RGB to class indices
    height, width = label_img.shape[:2]
    label_map = np.zeros((height, width), dtype=np.int64)

    for rgb, class_idx in CLASS_MAPPING.items():
        mask = np.all(label_img == rgb, axis=-1)
        label_map[mask] = class_idx

    # Count pixels per class
    unique, counts = np.unique(label_map, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    print(f"\nClass Distribution:")
    print(f"{'Class':<15} {'Name':<15} {'Pixel Count':<15} {'Percentage':<10}")
    print("-" * 60)

    total_pixels = height * width
    for class_idx in sorted(class_distribution.keys()):
        count = class_distribution[class_idx]
        percentage = 100 * count / total_pixels
        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else 'Unknown'
        print(f"{class_idx:<15} {class_name:<15} {count:<15,} {percentage:<10.2f}%")

    # Visualize labels
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Original RGB labels
    axes[0].imshow(label_img)
    axes[0].set_title('RGB Labels (Original)')
    axes[0].axis('off')

    # Class map
    im = axes[1].imshow(label_map, cmap='tab20')
    axes[1].set_title('Class Map (0-10)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig('data_inspection_labels.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved label visualization to: data_inspection_labels.png")
    plt.close()

    return label_map, class_distribution


def visualize_spectral_signatures(data_dir, label_map, num_samples_per_class=100, show_normalized=False):
    """Visualize spectral signatures for each class"""
    if label_map is None:
        print("\n⚠ Skipping spectral signature visualization (no labels)")
        return

    print(f"\n=== Spectral Signatures ===")
    if show_normalized:
        print("Note: Showing NORMALIZED spectra (as used in training)")
    else:
        print("Note: Showing RAW spectra (before normalization)")

    # Load all bands
    bands_dir = os.path.join(data_dir, 'bands')
    band_files = sorted([f for f in os.listdir(bands_dir) if f.endswith('.png')])

    bands_data = []
    wavelengths = []
    for band_file in band_files:
        band_path = os.path.join(bands_dir, band_file)
        band_img = np.array(Image.open(band_path), dtype=np.float32) / 255.0
        bands_data.append(band_img)

        # Extract wavelength
        wl = band_file.split('_')[-1].replace('nm.png', '')
        wavelengths.append(int(wl))

    bands_data = np.stack(bands_data, axis=0)  # Shape: (num_bands, height, width)
    wavelengths = np.array(wavelengths)

    # Sample spectra for each class
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for class_idx in range(len(CLASS_NAMES)):
        ax = axes[class_idx]

        # Find pixels of this class
        class_mask = label_map == class_idx
        class_pixels = np.argwhere(class_mask)

        if len(class_pixels) == 0:
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{CLASS_NAMES[class_idx]} (n=0)')
            continue

        # Sample random pixels
        n_samples = min(num_samples_per_class, len(class_pixels))
        sample_indices = np.random.choice(len(class_pixels), n_samples, replace=False)

        # Extract spectra
        spectra = []
        for idx in sample_indices:
            y, x = class_pixels[idx]
            spectrum = bands_data[:, y, x]

            # Apply normalization if requested
            if show_normalized:
                mean = spectrum.mean()
                std = spectrum.std()
                if std > 1e-6:
                    spectrum = (spectrum - mean) / std

            spectra.append(spectrum)

        spectra = np.array(spectra)

        # Plot individual spectra (transparent)
        for spectrum in spectra:
            ax.plot(wavelengths, spectrum, alpha=0.1, color='blue')

        # Plot mean spectrum
        mean_spectrum = spectra.mean(axis=0)
        std_spectrum = spectra.std(axis=0)
        ax.plot(wavelengths, mean_spectrum, color='red', linewidth=2, label='Mean')
        ax.fill_between(wavelengths,
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.3, color='red')

        ax.set_title(f'{CLASS_NAMES[class_idx]} (n={len(class_pixels):,})')
        ax.set_xlabel('Wavelength (nm)')
        ylabel = 'Normalized Value' if show_normalized else 'Reflectance'
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    suffix = '_normalized' if show_normalized else '_raw'
    filename = f'data_inspection_spectra{suffix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved spectral signature visualization to: {filename}")
    plt.close()


def check_dataset_loading(data_dir, bin_factor=1, max_samples_per_class=None):
    """Test dataset loading"""
    print(f"\n=== Dataset Loading Test ===")
    print(f"Parameters: bin_factor={bin_factor}, max_samples_per_class={max_samples_per_class}")

    try:
        dataset = HyperspectralDataset(
            data_dir,
            num_bands=26,
            is_training=True,
            bin_factor=bin_factor,
            max_samples_per_class=max_samples_per_class,
            normalize=True
        )

        print(f"✓ Dataset loaded successfully")
        print(f"  - Total samples: {len(dataset):,}")
        print(f"  - Image dimensions: {dataset.height} x {dataset.width}")
        print(f"  - Number of bands: {dataset.num_bands}")

        # Test loading a sample
        sample, label = dataset[0]
        print(f"  - Sample shape: {sample.shape}")
        print(f"  - Label: {label.item()} ({CLASS_NAMES[label.item()]})")
        print(f"  - Sample range: [{sample.min():.3f}, {sample.max():.3f}]")

        # Check class distribution in dataset
        print(f"\nSampling {min(10000, len(dataset))} samples to check class balance...")
        sample_size = min(10000, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)

        labels = []
        for idx in sample_indices:
            _, label = dataset[idx]
            labels.append(label.item())

        label_counts = Counter(labels)
        print(f"\nClass balance in sampled data:")
        for class_idx in sorted(label_counts.keys()):
            count = label_counts[class_idx]
            percentage = 100 * count / sample_size
            print(f"  {CLASS_NAMES[class_idx]:<15}: {count:>6} ({percentage:>5.2f}%)")

        return True

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(args):
    print("=" * 80)
    print("DATA INSPECTION TOOL")
    print("=" * 80)
    print(f"\nData directory: {args.data_dir}")

    # Check if directory exists
    if not os.path.exists(args.data_dir):
        print(f"✗ Error: Directory {args.data_dir} does not exist")
        return

    # Visualize bands
    visualize_bands(args.data_dir, num_bands_to_show=6)

    # Visualize labels
    label_map, class_distribution = visualize_labels(args.data_dir)

    # Visualize spectral signatures (both raw and normalized)
    if label_map is not None:
        print("\nGenerating spectral signature plots...")
        visualize_spectral_signatures(args.data_dir, label_map, num_samples_per_class=100, show_normalized=False)
        visualize_spectral_signatures(args.data_dir, label_map, num_samples_per_class=100, show_normalized=True)

    # Test dataset loading
    check_dataset_loading(
        args.data_dir,
        bin_factor=args.bin_factor,
        max_samples_per_class=args.max_samples_per_class
    )

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - data_inspection_bands.png (raw band images)")
    print("  - data_inspection_labels.png (label visualization)")
    print("  - data_inspection_spectra_raw.png (BEFORE normalization)")
    print("  - data_inspection_spectra_normalized.png (AFTER normalization - as used in training)")
    print("\nYou can now proceed with training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect hyperspectral training data')

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--bin_factor', type=int, default=1,
                        help='Pixel binning factor to test')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                        help='Max samples per class to test')

    args = parser.parse_args()
    main(args)
