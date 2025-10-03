import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Class mappings from labels.json
CLASS_MAPPING = {
    (0,0,0): 0,      # Background
    (255,0,0): 1,    # 95PU
    (0,0,255): 2,    # HIPS
    (255,125,125): 3, # HVDF-HFP
    (255,255,0): 4,  # GPSS
    (0,125,125): 5,  # PU
    (0,200,255): 6,  # 75PU
    (255,0,255): 7,  # 85PU
    (0,255,0): 8,    # PETE
    (255,125,0): 9,  # PET
    (255,0,100): 10  # PMMA
}

CLASS_NAMES = ['Background', '95PU', 'HIPS', 'HVDF-HFP', 'GPSS', 'PU', '75PU', '85PU', 'PETE', 'PET', 'PMMA']

class HyperspectralDataset(Dataset):
    """Dataset for hyperspectral material classification"""

    def __init__(self, data_dir, num_bands=26, transform=None, is_training=True, max_samples_per_class=None, bin_factor=1, normalize=True, spectral_augment=None):
        self.data_dir = data_dir
        self.bands_dir = os.path.join(data_dir, 'bands')
        self.labels_dir = os.path.join(data_dir, 'labels')
        self.num_bands = num_bands
        self.transform = transform
        self.is_training = is_training
        self.max_samples_per_class = max_samples_per_class
        self.bin_factor = bin_factor
        self.normalize = normalize
        self.spectral_augment = spectral_augment

        # Get band files
        self.band_files = sorted([f for f in os.listdir(self.bands_dir) if f.endswith('.png')])

        # Load first band to get image dimensions
        first_band = Image.open(os.path.join(self.bands_dir, self.band_files[0]))
        orig_height, orig_width = np.array(first_band).shape

        # Apply binning to dimensions
        self.height = orig_height // bin_factor
        self.width = orig_width // bin_factor

        # Load and cache all bands
        self.bands_data = self._load_all_bands()

        # Load labels if training
        if self.is_training:
            label_path = os.path.join(self.labels_dir, 'labels.png')
            self.labels = self._load_labels(label_path)

            # Extract all pixel indices (including background for training)
            self.valid_indices = np.argwhere(np.ones((self.height, self.width)))

            # Apply class-balanced sampling if requested
            if self.max_samples_per_class is not None:
                self.valid_indices = self._balance_classes(self.valid_indices)
        else:
            # For inference, use all pixels
            self.valid_indices = np.argwhere(np.ones((self.height, self.width)))

    def _bin_image(self, img, bin_factor):
        """Apply pixel binning by averaging blocks"""
        h, w = img.shape
        new_h, new_w = h // bin_factor, w // bin_factor
        return img[:new_h * bin_factor, :new_w * bin_factor].reshape(
            new_h, bin_factor, new_w, bin_factor
        ).mean(axis=(1, 3))

    def _load_all_bands(self):
        """Load all spectral bands into memory with optional binning"""
        bands = []
        for band_file in self.band_files:
            band_path = os.path.join(self.bands_dir, band_file)
            band_img = np.array(Image.open(band_path), dtype=np.float32) / 255.0

            # Apply pixel binning if bin_factor > 1
            if self.bin_factor > 1:
                band_img = self._bin_image(band_img, self.bin_factor)

            bands.append(band_img)
        return np.stack(bands, axis=0)  # Shape: (num_bands, height, width)

    def _bin_label_image(self, label_img, bin_factor):
        """Bin label image using mode (most common value in each block)"""
        h, w = label_img.shape[:2]
        new_h, new_w = h // bin_factor, w // bin_factor
        binned = np.zeros((new_h, new_w, 3), dtype=label_img.dtype)

        for i in range(new_h):
            for j in range(new_w):
                block = label_img[i*bin_factor:(i+1)*bin_factor, j*bin_factor:(j+1)*bin_factor]
                # Get most common RGB value in block
                block_flat = block.reshape(-1, 3)
                unique_colors, counts = np.unique(block_flat, axis=0, return_counts=True)
                binned[i, j] = unique_colors[np.argmax(counts)]

        return binned

    def _load_labels(self, label_path):
        """Convert RGB labels to class indices with optional binning"""
        label_img = np.array(Image.open(label_path))

        # Apply binning to labels if needed
        if self.bin_factor > 1:
            label_img = self._bin_label_image(label_img, self.bin_factor)

        label_map = np.zeros((self.height, self.width), dtype=np.int64)

        for rgb, class_idx in CLASS_MAPPING.items():
            mask = np.all(label_img == rgb, axis=-1)
            label_map[mask] = class_idx

        return label_map

    def _balance_classes(self, indices):
        """Balance classes by sampling up to max_samples_per_class from each class"""
        balanced_indices = []

        for class_idx in range(len(CLASS_NAMES)):  # Include background (class 0)
            class_mask = self.labels[indices[:, 0], indices[:, 1]] == class_idx
            class_indices = indices[class_mask]

            if len(class_indices) > 0:
                n_samples = min(len(class_indices), self.max_samples_per_class)
                sampled = class_indices[np.random.choice(len(class_indices), n_samples, replace=False)]
                balanced_indices.append(sampled)

        if balanced_indices:
            return np.vstack(balanced_indices)
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        y, x = self.valid_indices[idx]

        # Extract spectral signature for this pixel
        spectral_signature = self.bands_data[:, y, x]  # Shape: (num_bands,)

        if self.is_training:
            label = self.labels[y, x]

            # Apply spectral augmentation BEFORE normalization
            if self.spectral_augment is not None:
                spectral_signature = self.spectral_augment(spectral_signature)

            # Apply spectral normalization for better generalization
            if self.normalize:
                mean = spectral_signature.mean()
                std = spectral_signature.std()
                if std > 1e-6:  # Avoid division by zero
                    spectral_signature = (spectral_signature - mean) / std

            return torch.tensor(spectral_signature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        else:
            # Apply normalization for inference too
            if self.normalize:
                mean = spectral_signature.mean()
                std = spectral_signature.std()
                if std > 1e-6:  # Avoid division by zero
                    spectral_signature = (spectral_signature - mean) / std

            return torch.tensor(spectral_signature, dtype=torch.float32), (y, x)


class HyperspectralPatchDataset(Dataset):
    """Dataset that extracts spatial patches from hyperspectral images"""

    def __init__(self, data_dir, patch_size=3, num_bands=26, transform=None, is_training=True, max_samples_per_class=None, bin_factor=1, normalize=True):
        self.data_dir = data_dir
        self.bands_dir = os.path.join(data_dir, 'bands')
        self.labels_dir = os.path.join(data_dir, 'labels')
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.transform = transform
        self.is_training = is_training
        self.max_samples_per_class = max_samples_per_class
        self.bin_factor = bin_factor
        self.normalize = normalize
        self.padding = patch_size // 2

        # Get band files
        self.band_files = sorted([f for f in os.listdir(self.bands_dir) if f.endswith('.png')])

        # Load first band to get image dimensions
        first_band = Image.open(os.path.join(self.bands_dir, self.band_files[0]))
        orig_height, orig_width = np.array(first_band).shape

        # Apply binning to dimensions
        self.height = orig_height // bin_factor
        self.width = orig_width // bin_factor

        # Load and cache all bands
        self.bands_data = self._load_all_bands()

        # Pad the bands for patch extraction
        self.bands_data = np.pad(
            self.bands_data,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='reflect'
        )

        # Load labels if training
        if self.is_training:
            label_path = os.path.join(self.labels_dir, 'labels.png')
            self.labels = self._load_labels(label_path)

            # Extract all pixel indices (including background for training)
            self.valid_indices = np.argwhere(np.ones((self.height, self.width)))

            # Apply class-balanced sampling if requested
            if self.max_samples_per_class is not None:
                self.valid_indices = self._balance_classes(self.valid_indices)
        else:
            # For inference, use all pixels
            self.valid_indices = np.argwhere(np.ones((self.height, self.width)))

    def _bin_image(self, img, bin_factor):
        """Apply pixel binning by averaging blocks"""
        h, w = img.shape
        new_h, new_w = h // bin_factor, w // bin_factor
        return img[:new_h * bin_factor, :new_w * bin_factor].reshape(
            new_h, bin_factor, new_w, bin_factor
        ).mean(axis=(1, 3))

    def _load_all_bands(self):
        """Load all spectral bands into memory with optional binning"""
        bands = []
        for band_file in self.band_files:
            band_path = os.path.join(self.bands_dir, band_file)
            band_img = np.array(Image.open(band_path), dtype=np.float32) / 255.0

            # Apply pixel binning if bin_factor > 1
            if self.bin_factor > 1:
                band_img = self._bin_image(band_img, self.bin_factor)

            bands.append(band_img)
        return np.stack(bands, axis=0)  # Shape: (num_bands, height, width)

    def _bin_label_image(self, label_img, bin_factor):
        """Bin label image using mode (most common value in each block)"""
        h, w = label_img.shape[:2]
        new_h, new_w = h // bin_factor, w // bin_factor
        binned = np.zeros((new_h, new_w, 3), dtype=label_img.dtype)

        for i in range(new_h):
            for j in range(new_w):
                block = label_img[i*bin_factor:(i+1)*bin_factor, j*bin_factor:(j+1)*bin_factor]
                # Get most common RGB value in block
                block_flat = block.reshape(-1, 3)
                unique_colors, counts = np.unique(block_flat, axis=0, return_counts=True)
                binned[i, j] = unique_colors[np.argmax(counts)]

        return binned

    def _load_labels(self, label_path):
        """Convert RGB labels to class indices with optional binning"""
        label_img = np.array(Image.open(label_path))

        # Apply binning to labels if needed
        if self.bin_factor > 1:
            label_img = self._bin_label_image(label_img, self.bin_factor)

        label_map = np.zeros((self.height, self.width), dtype=np.int64)

        for rgb, class_idx in CLASS_MAPPING.items():
            mask = np.all(label_img == rgb, axis=-1)
            label_map[mask] = class_idx

        return label_map

    def _balance_classes(self, indices):
        """Balance classes by sampling up to max_samples_per_class from each class"""
        balanced_indices = []

        for class_idx in range(len(CLASS_NAMES)):  # Include background (class 0)
            class_mask = self.labels[indices[:, 0], indices[:, 1]] == class_idx
            class_indices = indices[class_mask]

            if len(class_indices) > 0:
                n_samples = min(len(class_indices), self.max_samples_per_class)
                sampled = class_indices[np.random.choice(len(class_indices), n_samples, replace=False)]
                balanced_indices.append(sampled)

        if balanced_indices:
            return np.vstack(balanced_indices)
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        y, x = self.valid_indices[idx]

        # Extract patch (centered at y, x) - accounting for padding
        patch = self.bands_data[
            :,
            y:y + 2*self.padding + 1,
            x:x + 2*self.padding + 1
        ]  # Shape: (num_bands, patch_size, patch_size)

        # Apply spectral normalization for better generalization
        if self.normalize:
            for band_idx in range(patch.shape[0]):
                band = patch[band_idx]
                mean = band.mean()
                std = band.std()
                if std > 1e-6:  # Avoid division by zero
                    patch[band_idx] = (band - mean) / std

        if self.is_training:
            label = self.labels[y, x]

            if self.transform:
                # Transform expects (H, W, C) format
                patch_hwc = np.transpose(patch, (1, 2, 0))
                augmented = self.transform(image=patch_hwc)
                patch = np.transpose(augmented['image'], (2, 0, 1))

            return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        else:
            return torch.tensor(patch, dtype=torch.float32), (y, x)


def get_train_transforms():
    """Augmentation for training"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # GaussNoise removed - can add MultiplicativeNoise or other augmentations if needed
    ])


def get_val_transforms():
    """No augmentation for validation"""
    return None
