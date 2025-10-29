import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import csv
from dataset import AlzheimersDataset
from dataset import AddRegularization
from torch.optim.lr_scheduler import CosineAnnealingLR
from modules import create_convnext_model

# Import our custom modules
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def find_best_threshold(labels, outputs):
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (outputs > threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def train(args):
    # 1. SETUP
    # ============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # This helps PyTorch find the best algorithms for your hardware.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Enhanced data augmentation for MRI
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Intensity jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)), # Less aggressive erasing
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
    original_dataset = AlzheimersDataset(root_dir=train_dir, transform=data_transforms['train'])

    # Create augmented dataset
    augmented_dataset_full = AlzheimersDataset(root_dir=train_dir, is_augmented=True, augmentation_transform=data_transforms['train'])

    # Concatenate original and augmented datasets
    train_dataset = ConcatDataset([original_dataset, augmented_dataset_full])

    image_datasets = {
        'train': train_dataset,
        'val': AlzheimersDataset(root_dir=val_dir, transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, # Use multiple processes for data loading
            pin_memory=True # Speeds up CPU-to-GPU data transfer
        ),
        'val': DataLoader(
            image_datasets['val'], 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, pin_memory=True)
    }
    
    print(f"Training set size: {len(image_datasets['train'])}")
    print(f"Validation set size: {len(image_datasets['val'])}")

    # Calculate class weights for unbalanced dataset
    train_labels = [label for _, label in original_dataset.samples]
    num_positives = np.sum(train_labels)
    num_negatives = len(train_labels) - num_positives
    pos_weight = torch.tensor(num_negatives / num_positives, device=device)
    print(f"Positive weight for BCE loss: {pos_weight:.2f}")

    # 3. MODEL, LOSS, OPTIMIZER, SCHEDULER
    # ============================================================================
    # Define ConvNeXt-Small architecture parameters
    convnext_s_depths = [3, 3, 9, 3]
    convnext_s_dims = [96, 192, 384, 768] # ConvNeXt-Small dims
    model = create_convnext_model(
        num_classes=1, 
        in_chans=1,
        depths=convnext_s_depths, 
        dims=convnext_s_dims, 
        drop_path_rate=args.drop_path_rate
    ).to(device)

    print("Compiling the model... (This may take a moment on the first run)")
    # If you encounter TritonMissing error on Windows, comment out the line below.
    # model = torch.compile(model, backend="inductor", mode="reduce-overhead")


    ema = EMA(model, decay=args.ema_decay)
    ema.register()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Using the newer torch.amp.GradScaler API to avoid deprecation warnings.
    scaler = torch.amp.GradScaler()
    
    # Use CosineAnnealingLR scheduler with warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. TRAINING LOOP
    # ============================================================================
    best_val_f1 = 0.0
    patience_counter = 0
    best_epoch = 0
    log_file = f'training_log_lr{args.learning_rate}_wd{args.weight_decay}_bs{args.batch_size}.csv'

    # Open the log file
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'lr'])

        for epoch in range(args.epochs):
            print(f'\nEpoch {epoch+1}/{args.epochs}')
            print('-' * 10)

            # --- Warmup Phase ---
            if epoch < args.warmup_epochs:
                # Linearly increase the learning rate
                lr_scale = (epoch + 1) / args.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate * lr_scale

            # --- Train Phase ---
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(dataloaders['train'], desc="Train Phase")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                # Mixup
                if args.mixup_alpha > 0:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.mixup_alpha, use_cuda=torch.cuda.is_available())
                else:
                    targets_a, targets_b, lam = labels, labels, 1.0

                # Label smoothing
                if args.label_smoothing > 0:
                    targets_a = targets_a * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                    targets_b = targets_b * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing

                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                ema.update()

                running_loss += loss.item() * inputs.size(0)

            train_loss = running_loss / len(image_datasets['train'])
            print(f'Train Loss: {train_loss:.4f}')

            # --- Validation Phase ---
            model.eval()
            ema.apply_shadow() # Use EMA weights for validation

            val_outputs = []
            val_labels = []
            running_val_loss = 0.0

            progress_bar_val = tqdm(dataloaders['val'], desc="Val Phase")
            for inputs, labels in progress_bar_val:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                val_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

            ema.restore() # Restore original weights

            val_loss = running_val_loss / len(image_datasets['val'])
            
            best_threshold = find_best_threshold(val_labels, np.array(val_outputs))
            val_preds = (np.array(val_outputs) > best_threshold).astype(int)
            val_accuracy = accuracy_score(val_labels, val_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=0)

            print(f'Val Loss: {val_loss:.4f}')
            print(f'Best threshold: {best_threshold:.2f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            print(f'Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

            # Log metrics to CSV
            writer.writerow([epoch + 1, train_loss, val_loss, val_accuracy, precision, recall, f1, optimizer.param_groups[0]['lr']])
            f.flush()

            # Early stopping & Model saving
            if f1 > best_val_f1:
                best_val_f1 = f1
                patience_counter = 0
                best_epoch = epoch + 1
                # Save the EMA model state
                ema.apply_shadow()
                torch.save(model.state_dict(), args.model_save_path)
                ema.restore()
                print(f"Validation F1-score improved. Saving EMA model to {args.model_save_path}")
            else:
                patience_counter += 1
                print(f"Validation F1-score did not improve. Patience: {patience_counter}/{args.patience}")

            # Scheduler step (after warmup)
            if epoch >= args.warmup_epochs:
                scheduler.step()
            
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print("\nTraining complete.")
    print(f"Best model from epoch {best_epoch} with validation F1-score {best_val_f1:.4f} saved to {args.model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ConvNeXt model for Alzheimer\'s classification.')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(base_dir, 'ADNI', 'AD_NC')

    parser.add_argument('--data-dir', type=str, default=default_data_dir, help='Path to the root data directory (containing train and test folders)')
    parser.add_argument('--learning-rate', type=float, default=4e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay for the AdamW optimizer')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train for')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of worker processes for data loading')
    parser.add_argument('--model-save-path', type=str, default='alzheimers_convnext_v2.pth', help='Path to save the trained model')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--mixup-alpha', type=float, default=0.8, help='Alpha parameter for Mixup augmentation')
    parser.add_argument('--drop-path-rate', type=float, default=0.2, help='Stochastic depth rate for ConvNeXt-S')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='Decay rate for Exponential Moving Average')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Number of epochs for learning rate warmup')


    args = parser.parse_args()
    train(args)