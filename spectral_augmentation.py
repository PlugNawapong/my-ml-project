import numpy as np
import torch


class SpectralAugmentation:
    """Augmentation techniques for 1D spectral signatures"""

    def __init__(self,
                 noise_std=0.01,
                 brightness_range=(-0.1, 0.1),
                 scale_range=(0.9, 1.1),
                 shift_range=(-2, 2),
                 spectral_dropout_prob=0.1,
                 smooth_prob=0.2,
                 p=0.5):
        """
        Args:
            noise_std: Standard deviation for Gaussian noise
            brightness_range: Range for brightness adjustment
            scale_range: Range for multiplicative scaling
            shift_range: Range for spectral band shifting
            spectral_dropout_prob: Probability of dropping spectral bands
            smooth_prob: Probability of applying spectral smoothing
            p: Overall probability of applying augmentations
        """
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.spectral_dropout_prob = spectral_dropout_prob
        self.smooth_prob = smooth_prob
        self.p = p

    def __call__(self, spectrum):
        """Apply random augmentations to spectral signature"""
        if np.random.rand() > self.p:
            return spectrum

        spectrum = spectrum.copy()

        # 1. Additive Gaussian noise
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, self.noise_std, spectrum.shape)
            spectrum = spectrum + noise

        # 2. Brightness adjustment (additive)
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(*self.brightness_range)
            spectrum = spectrum + brightness

        # 3. Scale adjustment (multiplicative)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(*self.scale_range)
            spectrum = spectrum * scale

        # 4. Spectral band dropout (simulate missing bands)
        if np.random.rand() < self.spectral_dropout_prob:
            n_bands = len(spectrum)
            n_drop = np.random.randint(1, max(2, n_bands // 5))  # Drop up to 20% of bands
            drop_indices = np.random.choice(n_bands, n_drop, replace=False)
            # Interpolate dropped bands from neighbors
            for idx in drop_indices:
                if idx == 0:
                    spectrum[idx] = spectrum[idx + 1]
                elif idx == n_bands - 1:
                    spectrum[idx] = spectrum[idx - 1]
                else:
                    spectrum[idx] = (spectrum[idx - 1] + spectrum[idx + 1]) / 2

        # 5. Spectral smoothing (simulate sensor noise reduction)
        if np.random.rand() < self.smooth_prob:
            kernel_size = 3
            kernel = np.ones(kernel_size) / kernel_size
            spectrum = np.convolve(spectrum, kernel, mode='same')

        # 6. Spectral shift (simulate wavelength calibration error)
        if np.random.rand() < 0.3:
            shift = np.random.randint(*self.shift_range)
            if shift != 0:
                spectrum = np.roll(spectrum, shift)

        return spectrum


class MixUp:
    """MixUp augmentation for spectral data"""

    def __init__(self, alpha=0.2, p=0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying mixup
        """
        self.alpha = alpha
        self.p = p

    def __call__(self, spectrum1, spectrum2, label1, label2):
        """Mix two spectra"""
        if np.random.rand() > self.p:
            return spectrum1, label1

        lam = np.random.beta(self.alpha, self.alpha)
        mixed_spectrum = lam * spectrum1 + (1 - lam) * spectrum2

        return mixed_spectrum, label1  # Return original label for simplicity


class SpectralCutout:
    """Cutout augmentation for spectral data (mask out spectral regions)"""

    def __init__(self, n_holes=1, length_ratio=0.2, p=0.3):
        """
        Args:
            n_holes: Number of holes to cut out
            length_ratio: Ratio of spectrum length to cut out
            p: Probability of applying cutout
        """
        self.n_holes = n_holes
        self.length_ratio = length_ratio
        self.p = p

    def __call__(self, spectrum):
        """Apply spectral cutout"""
        if np.random.rand() > self.p:
            return spectrum

        spectrum = spectrum.copy()
        n_bands = len(spectrum)
        length = int(n_bands * self.length_ratio)

        for _ in range(self.n_holes):
            # Random position
            start = np.random.randint(0, n_bands - length)
            end = start + length

            # Fill with mean value
            spectrum[start:end] = spectrum.mean()

        return spectrum


class ContrastAdjustment:
    """Adjust spectral contrast"""

    def __init__(self, contrast_range=(0.8, 1.2), p=0.5):
        """
        Args:
            contrast_range: Range for contrast adjustment
            p: Probability of applying contrast adjustment
        """
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, spectrum):
        """Adjust contrast"""
        if np.random.rand() > self.p:
            return spectrum

        spectrum = spectrum.copy()
        mean = spectrum.mean()

        # Adjust contrast around mean
        contrast = np.random.uniform(*self.contrast_range)
        spectrum = mean + (spectrum - mean) * contrast

        return spectrum


class ComposeSpectral:
    """Compose multiple spectral augmentations"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, spectrum):
        for transform in self.transforms:
            spectrum = transform(spectrum)
        return spectrum


def get_spectral_augmentation(mode='light'):
    """
    Get spectral augmentation pipeline

    Args:
        mode: 'light', 'medium', or 'heavy'
    """
    if mode == 'light':
        return ComposeSpectral([
            SpectralAugmentation(
                noise_std=0.01,
                brightness_range=(-0.05, 0.05),
                scale_range=(0.95, 1.05),
                shift_range=(-1, 1),
                spectral_dropout_prob=0.05,
                smooth_prob=0.1,
                p=0.5
            )
        ])

    elif mode == 'medium':
        return ComposeSpectral([
            SpectralAugmentation(
                noise_std=0.02,
                brightness_range=(-0.1, 0.1),
                scale_range=(0.9, 1.1),
                shift_range=(-2, 2),
                spectral_dropout_prob=0.1,
                smooth_prob=0.2,
                p=0.7
            ),
            ContrastAdjustment(contrast_range=(0.9, 1.1), p=0.3)
        ])

    elif mode == 'heavy':
        return ComposeSpectral([
            SpectralAugmentation(
                noise_std=0.03,
                brightness_range=(-0.15, 0.15),
                scale_range=(0.85, 1.15),
                shift_range=(-3, 3),
                spectral_dropout_prob=0.15,
                smooth_prob=0.3,
                p=0.8
            ),
            ContrastAdjustment(contrast_range=(0.8, 1.2), p=0.5),
            SpectralCutout(n_holes=1, length_ratio=0.15, p=0.3)
        ])

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'light', 'medium', 'heavy'")
