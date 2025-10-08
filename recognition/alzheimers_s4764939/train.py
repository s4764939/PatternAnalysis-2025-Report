import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Import our custom modules
from dataset import AlzheimersDataset
from modules import create_convnext_model

def train(args):
    # 1. SETUP
    # ============================================================================
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define data transformations
    # For training, we add data augmentation. For validation, we only resize and normalize.
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # 2. DATA LOADING
    # ============================================================================
    # Create datasets and dataloaders for training and validation
    # NOTE: We are using the 'test' set provided in the data as our validation set here.
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'test')

    image_datasets = {
        'train': AlzheimersDataset(root_dir=train_dir, transform=data_transforms['train']),
        'val': AlzheimersDataset(root_dir=val_dir, transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=4)
    }
    
    print(f"Training set size: {len(image_datasets['train'])}")
    print(f"Validation set size: {len(image_datasets['val'])}")

    # 3. MODEL, LOSS, OPTIMIZER
    # ============================================================================
    # Create the model and move it to the selected device
    model = create_convnext_model(num_classes=1).to(device)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 4. TRAINING LOOP
    # ============================================================================
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            all_labels = []
            all_preds = []

            # Iterate over data using tqdm for a progress bar
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Phase', ascii=True, ncols=100):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1) # Ensure labels are correct shape and type

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                
                if phase == 'val':
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(image_datasets[phase])
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')

            # Calculate and print validation metrics
            if phase == 'val':
                val_accuracy = accuracy_score(all_labels, all_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
                print(f'Validation Accuracy: {val_accuracy:.4f}')
                print(f'Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

                # Save the model if it has the best validation accuracy so far
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    torch.save(model.state_dict(), args.model_save_path)
                    print(f"New best model saved to {args.model_save_path} with accuracy: {best_val_acc:.4f}")

    print("\nTraining complete.")
    print(f"Best validation accuracy achieved: {best_val_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ConvNeXt model for Alzheimer\'s classification.')
    
    # Get the parent directory of the current script to find the 'ADNI' folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(base_dir, 'ADNI', 'AD_NC')

    parser.add_argument('--data-dir', type=str, default=default_data_dir, help='Path to the root data directory (containing train and test folders)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--model-save-path', type=str, default='alzheimers_convnext.pth', help='Path to save the trained model')

    args = parser.parse_args()
    train(args)