import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralCNN1D(nn.Module):
    """1D CNN for pixel-wise spectral classification"""

    def __init__(self, num_bands=26, num_classes=11, dropout=0.3):
        super(SpectralCNN1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

        # Calculate size after pooling
        self.fc_input_size = 256 * (num_bands // 2 // 2 // 2)

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (batch, num_bands)
        x = x.unsqueeze(1)  # (batch, 1, num_bands)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class SpectralCNN2D(nn.Module):
    """2D CNN for spatial-spectral classification using patches"""

    def __init__(self, num_bands=26, num_classes=11, patch_size=3, dropout=0.3):
        super(SpectralCNN2D, self).__init__()

        # 2D convolutions treating bands as channels
        self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (batch, num_bands, patch_size, patch_size)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class HybridSpectralNet(nn.Module):
    """Hybrid network combining spectral and spatial features"""

    def __init__(self, num_bands=26, num_classes=11, patch_size=3, dropout=0.3):
        super(HybridSpectralNet, self).__init__()

        # Spectral branch (1D convolutions)
        self.spectral_conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.spectral_bn1 = nn.BatchNorm1d(32)
        self.spectral_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.spectral_bn2 = nn.BatchNorm1d(64)

        # Spatial branch (2D convolutions)
        self.spatial_conv1 = nn.Conv2d(num_bands, 32, kernel_size=3, padding=1)
        self.spatial_bn1 = nn.BatchNorm2d(32)
        self.spatial_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.spatial_bn2 = nn.BatchNorm2d(64)

        self.pool1d = nn.MaxPool1d(2)
        self.pool2d = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        # Fusion and classification
        spectral_features = 64 * (num_bands // 2 // 2)
        spatial_features = 64

        self.fc1 = nn.Linear(spectral_features + spatial_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, num_bands, patch_size, patch_size)

        # Extract center pixel spectrum for spectral branch
        center = x.shape[2] // 2
        spectral_input = x[:, :, center, center]  # (batch, num_bands)
        spectral_input = spectral_input.unsqueeze(1)  # (batch, 1, num_bands)

        # Spectral branch
        s = self.pool1d(F.relu(self.spectral_bn1(self.spectral_conv1(spectral_input))))
        s = self.pool1d(F.relu(self.spectral_bn2(self.spectral_conv2(s))))
        s = s.view(s.size(0), -1)

        # Spatial branch
        sp = F.relu(self.spatial_bn1(self.spatial_conv1(x)))
        sp = F.relu(self.spatial_bn2(self.spatial_conv2(sp)))
        sp = self.global_pool(sp)
        sp = sp.view(sp.size(0), -1)

        # Fusion
        combined = torch.cat([s, sp], dim=1)

        # Classification
        out = self.dropout(F.relu(self.fc1(combined)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)

        return out


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)

        return out


class SpectralResNet(nn.Module):
    """ResNet-based architecture for hyperspectral classification"""

    def __init__(self, num_bands=26, num_classes=11, dropout=0.3):
        super(SpectralResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256)
        self.layer4 = ResidualBlock(256, 512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: (batch, num_bands, patch_size, patch_size)

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def get_model(model_name='spectral_cnn_2d', num_bands=26, num_classes=11, patch_size=3, dropout=0.3):
    """Factory function to create models"""

    models = {
        'spectral_cnn_1d': SpectralCNN1D(num_bands, num_classes, dropout),
        'spectral_cnn_2d': SpectralCNN2D(num_bands, num_classes, patch_size, dropout),
        'hybrid': HybridSpectralNet(num_bands, num_classes, patch_size, dropout),
        'resnet': SpectralResNet(num_bands, num_classes, dropout),
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")

    return models[model_name]
