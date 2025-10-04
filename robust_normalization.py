import numpy as np


class RobustSpectralNormalization:
    """Robust normalization techniques for spectral data with intensity differences"""

    @staticmethod
    def min_max_normalize(spectrum, global_min=0.0, global_max=1.0):
        """Min-max normalization to [0, 1] range"""
        spec_min = spectrum.min()
        spec_max = spectrum.max()
        if spec_max - spec_min > 1e-6:
            return (spectrum - spec_min) / (spec_max - spec_min)
        return spectrum

    @staticmethod
    def z_score_normalize(spectrum):
        """Z-score normalization (mean=0, std=1)"""
        mean = spectrum.mean()
        std = spectrum.std()
        if std > 1e-6:
            return (spectrum - mean) / std
        return spectrum

    @staticmethod
    def robust_scale_normalize(spectrum, percentile_range=(5, 95)):
        """Normalize using percentile range (robust to outliers)"""
        p_low, p_high = np.percentile(spectrum, percentile_range)
        if p_high - p_low > 1e-6:
            spectrum_clipped = np.clip(spectrum, p_low, p_high)
            return (spectrum_clipped - p_low) / (p_high - p_low)
        return spectrum

    @staticmethod
    def snv_normalize(spectrum):
        """Standard Normal Variate (SNV) normalization - common in spectroscopy"""
        mean = spectrum.mean()
        std = spectrum.std()
        if std > 1e-6:
            return (spectrum - mean) / std
        return spectrum

    @staticmethod
    def msc_normalize(spectrum, reference=None):
        """Multiplicative Scatter Correction (MSC) - corrects for scattering effects"""
        if reference is None:
            reference = spectrum.mean()

        # Fit a line: spectrum = a + b * reference
        mean_spec = spectrum.mean()
        mean_ref = reference if isinstance(reference, (int, float)) else reference.mean()

        # Simple scaling version
        if mean_ref > 1e-6:
            return spectrum / mean_ref
        return spectrum

    @staticmethod
    def vector_normalize(spectrum):
        """L2 normalization (unit vector)"""
        norm = np.linalg.norm(spectrum)
        if norm > 1e-6:
            return spectrum / norm
        return spectrum

    @staticmethod
    def area_normalize(spectrum):
        """Normalize by area under the curve"""
        area = np.trapz(spectrum)
        if abs(area) > 1e-6:
            return spectrum / area
        return spectrum

    @staticmethod
    def max_normalize(spectrum):
        """Normalize by maximum value"""
        max_val = np.max(np.abs(spectrum))
        if max_val > 1e-6:
            return spectrum / max_val
        return spectrum

    @staticmethod
    def first_derivative(spectrum):
        """First derivative (Savitzky-Golay-like)"""
        return np.gradient(spectrum)

    @staticmethod
    def second_derivative(spectrum):
        """Second derivative"""
        return np.gradient(np.gradient(spectrum))


class CombinedNormalization:
    """Combine multiple normalization techniques for maximum robustness"""

    def __init__(self, method='snv+minmax'):
        """
        Args:
            method: Normalization method
                - 'snv': Standard Normal Variate
                - 'snv+minmax': SNV followed by min-max
                - 'robust': Robust scaling
                - 'msc': Multiplicative Scatter Correction
                - 'vector': L2 normalization
                - 'area': Area normalization
                - 'max': Max normalization
        """
        self.method = method
        self.normalizer = RobustSpectralNormalization()

    def __call__(self, spectrum):
        """Apply normalization"""
        spectrum = spectrum.copy()

        if self.method == 'snv':
            return self.normalizer.snv_normalize(spectrum)

        elif self.method == 'snv+minmax':
            # SNV to handle intensity differences, then min-max for consistent range
            spectrum = self.normalizer.snv_normalize(spectrum)
            return self.normalizer.min_max_normalize(spectrum)

        elif self.method == 'robust':
            return self.normalizer.robust_scale_normalize(spectrum)

        elif self.method == 'msc':
            return self.normalizer.msc_normalize(spectrum)

        elif self.method == 'vector':
            return self.normalizer.vector_normalize(spectrum)

        elif self.method == 'area':
            return self.normalizer.area_normalize(spectrum)

        elif self.method == 'max':
            return self.normalizer.max_normalize(spectrum)

        elif self.method == 'minmax':
            return self.normalizer.min_max_normalize(spectrum)

        elif self.method == 'robust+snv':
            # Robust scaling followed by SNV
            spectrum = self.normalizer.robust_scale_normalize(spectrum)
            return self.normalizer.snv_normalize(spectrum)

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


def get_normalization(method='snv+minmax'):
    """
    Get normalization function

    Recommended methods for handling intensity differences:
    - 'snv+minmax': Best for large intensity differences (RECOMMENDED)
    - 'robust': Good for outliers
    - 'msc': Good for scattering effects
    - 'vector': Good for shape-based classification
    - 'max': Simple but effective
    """
    return CombinedNormalization(method=method)
