import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json
from datetime import datetime

from dataset import HyperspectralDataset, HyperspectralPatchDataset, get_train_transforms, get_val_transforms, CLASS_NAMES
from model import get_model
from spectral_augmentation import get_spectral_augmentation


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Per-class accuracy
    class_correct = [0] * len(CLASS_NAMES)
    class_total = [0] * len(CLASS_NAMES)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    # Calculate per-class accuracy
    class_acc = {}
    for i in range(len(CLASS_NAMES)):
        if class_total[i] > 0:
            class_acc[CLASS_NAMES[i]] = 100 * class_correct[i] / class_total[i]
        else:
            class_acc[CLASS_NAMES[i]] = 0.0

    return epoch_loss, epoch_acc, class_acc


def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device - prioritize MPS for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.model}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Dataset and DataLoader
    print(f'Loading dataset from {args.data_dir}...')

    # Setup spectral augmentation for 1D model
    spectral_aug = None
    if args.spectral_augment and not args.use_patches:
        spectral_aug = get_spectral_augmentation(mode=args.spectral_augment)
        print(f'Using spectral augmentation: {args.spectral_augment}')

    if args.use_patches:
        full_dataset = HyperspectralPatchDataset(
            args.data_dir,
            patch_size=args.patch_size,
            num_bands=args.num_bands,
            transform=get_train_transforms() if args.augment else None,
            is_training=True,
            max_samples_per_class=args.max_samples_per_class,
            bin_factor=args.bin_factor,
            norm_method=args.norm_method
        )
    else:
        full_dataset = HyperspectralDataset(
            args.data_dir,
            num_bands=args.num_bands,
            transform=None,
            is_training=True,
            max_samples_per_class=args.max_samples_per_class,
            bin_factor=args.bin_factor,
            spectral_augment=spectral_aug,
            norm_method=args.norm_method
        )

    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Model
    print(f'Creating model: {args.model}')
    model = get_model(
        args.model,
        num_bands=args.num_bands,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        dropout=args.dropout
    )
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {num_params:,}')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f'\nStarting training for {args.epochs} epochs...\n')

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

        # Validate
        val_loss, val_acc, class_acc = validate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Print per-class accuracy
        print('Per-class accuracy:')
        for class_name, acc in class_acc.items():
            print(f'  {class_name}: {acc:.2f}%')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f'Best model saved with validation accuracy: {val_acc:.2f}%')

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        print()

    # Save final model and history
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print(f'\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%')
    print(f'Models saved to: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hyperspectral material classifier')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to training data directory')
    parser.add_argument('--num_bands', type=int, default=26,
                        help='Number of spectral bands')
    parser.add_argument('--num_classes', type=int, default=11,
                        help='Number of material classes')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Training set split ratio')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                        help='Max samples per class for faster training (e.g., 5000). None = use all samples')
    parser.add_argument('--bin_factor', type=int, default=1,
                        help='Pixel binning factor for faster training (e.g., 2, 4). 1 = no binning')

    # Model parameters
    parser.add_argument('--model', type=str, default='spectral_cnn_2d',
                        choices=['spectral_cnn_1d', 'spectral_cnn_2d', 'hybrid', 'resnet'],
                        help='Model architecture to use')
    parser.add_argument('--use_patches', action='store_true',
                        help='Use spatial patches (required for 2D models)')
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Spatial patch size')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5 for better generalization)')
    parser.add_argument('--spectral_augment', type=str, default=None,
                        choices=[None, 'light', 'medium', 'heavy'],
                        help='Spectral augmentation mode for 1D models (None, light, medium, heavy)')
    parser.add_argument('--norm_method', type=str, default='percentile',
                        choices=['percentile', 'standard'],
                        help='Normalization method: percentile (2-98%% band-wise) or standard (default: percentile)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')

    # Other parameters
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for models and logs')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Validate arguments
    if args.model != 'spectral_cnn_1d' and not args.use_patches:
        print('Warning: 2D models require --use_patches flag. Setting it automatically.')
        args.use_patches = True

    main(args)
