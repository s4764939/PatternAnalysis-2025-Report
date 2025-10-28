import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import csv
import random

# Import our custom modules
from dataset import AlzheimersDataset, AddRegularization
# --- MODIFICATION ---
# Import the new custom model creator instead of the old one
from modules import create_convnext_model as create_custom_convnext_model
# --- END MODIFICATION ---

def train(args):
    # 1. SETUP
    # ============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Finalized data augmentation pipeline
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.15), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddRegularization(probability=0.5, noise_factor=args.noise_factor, cutout_size=args.cutout_size),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
    }

    # 2. DATA LOADING
    # ============================================================================
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'test')

    # Create original dataset
    original_dataset = AlzheimersDataset(root_dir=train_dir, transform=data_transforms['val'])

    # Create augmented dataset
    augmented_dataset_full = AlzheimersDataset(root_dir=train_dir, is_augmented=True, augmentation_transform=data_transforms['train'])
    
    # Take half of the augmented samples
    num_augmented_samples = len(augmented_dataset_full) // 2
    augmented_indices = torch.randperm(len(augmented_dataset_full))[:num_augmented_samples].tolist()
    augmented_subset = Subset(augmented_dataset_full, augmented_indices)

    # Concatenate original and augmented datasets
    train_dataset = ConcatDataset([original_dataset, augmented_subset])

    image_datasets = {
        'train': train_dataset,
        'val': AlzheimersDataset(root_dir=val_dir, transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=0)
    }
    
    print(f"Training set size: {len(image_datasets['train'])}")
    print(f"Validation set size: {len(image_datasets['val'])}")

    # Calculate positive weight for BCE loss from the original dataset
    train_labels = [label for _, label in original_dataset.samples]
    num_positives = np.sum(train_labels)
    num_negatives = len(train_labels) - num_positives
    pos_weight = torch.tensor(num_negatives / num_positives, device=device)
    print(f"Positive weight for BCE loss: {pos_weight:.2f}")

    # 3. MODEL, LOSS, OPTIMIZER, SCHEDULER
    # ============================================================================
    
    # --- MODIFICATION ---
    # Call the new custom model creator.
    # We now pass no params to get the default (ConvNeXt-S like) arch
    print("Creating custom ConvNeXt model from scratch...")
    model = create_custom_convnext_model(
        num_classes=1, 
        in_chans=1
        # We use the default depths/dims which match the old 'convnext_small'
    ).to(device)
    # --- END MODIFICATION ---

    criterion = nn.BCEWithLogitsLoss()
    criterion_smooth = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. TRAINING LOOP
    # ============================================================================
    best_val_f1 = 0.0
    patience_counter = 0
    best_epoch = 0
    log_file = 'training_log.csv'

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'lr'])

        for epoch in range(args.epochs):
            print(f'\nEpoch {epoch+1}/{args.epochs}')
            print('-' * 10)

            # Train phase
            model.train()
            running_loss = 0.0
            
            progress_bar = tqdm(dataloaders['train'], desc="Train Phase")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    labels_smooth = labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                    loss = criterion_smooth(outputs, labels_smooth)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(image_datasets['train'])
            print(f'Train Loss: {epoch_loss:.4f}')

            # Validation phase
            model.eval()
            val_outputs = []
            val_labels = []
            running_val_loss = 0.0

            progress_bar_val = tqdm(dataloaders['val'], desc="Val Phase")
            for inputs, labels in progress_bar_val:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                val_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

            epoch_val_loss = running_val_loss / len(image_datasets['val'])
            
            best_f1 = 0
            best_threshold = 0
            for threshold in np.arange(0.1, 1.0, 0.1):
                val_preds = (np.array(val_outputs) > threshold).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            print(f"Best threshold: {best_threshold:.2f}")
            val_preds = (np.array(val_outputs) > best_threshold).astype(int)
            val_accuracy = accuracy_score(val_labels, val_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=0)

            print(f'Validation Predictions Distribution: {np.bincount(np.array(val_preds).flatten())}')

            print(f'Val Loss: {epoch_val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            print(f'Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

            # Save model if F1 score improved
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), args.model_save_path)
                print(f"Validation F1-score improved. Saving model to {args.model_save_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Log results
            writer.writerow([epoch + 1, epoch_loss, epoch_val_loss, val_accuracy, precision, recall, f1, optimizer.param_groups[0]['lr']])

            # Scheduler step
            scheduler.step()

            # Early stopping
            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {args.early_stopping_patience} epochs with no improvement.")
                print(f"Best F1 score of {best_val_f1:.4f} was achieved at epoch {best_epoch}.")
                break
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a *custom* ConvNeXt model for Alzheimer\'s classification.')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(base_dir, 'ADNI', 'AD_NC')

    parser.add_argument('--data-dir', type=str, default=default_data_dir, help='Path to the root data directory')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-2, help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--model-save-path', type=str, default='alzheimers_convnext_v3.pth', help='Path to save the model')
    parser.add_argument('--label-smoothing', type=float, default=0.2, help='Label smoothing factor')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--early-stopping-patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--noise-factor', type=float, default=0.1, help='Factor for Gaussian noise augmentation')
    parser.add_argument('--cutout-size', type=float, default=0.4, help='Size of the cutout augmentation as a fraction of image size')

    args = parser.parse_args()
    train(args)

