import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import random
import numpy as np

IMAGE_SIZE = (240, 240)

class AddRegularization(object):
    def __init__(self, probability=0.5, noise_factor=0.05, cutout_size=0.4):
        self.probability = probability
        self.noise_factor = noise_factor
        self.cutout_size = cutout_size

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            raise TypeError("Image must be a torch.Tensor. Apply ToTensor() before this transform.")

        if random.random() > self.probability:
            return img

        if random.random() < 0.5:
            # Add Gaussian noise and clip
            noise = torch.randn_like(img) * self.noise_factor
            noisy_img = img + noise
            return torch.clamp(noisy_img, 0., 1.)
        else:
            # Apply cutout
            h, w = img.size(1), img.size(2)
            cutout_h, cutout_w = int(h * self.cutout_size), int(w * self.cutout_size)
            
            corner = random.randint(0, 3)
            if corner == 0: x1, y1 = 0, 0
            elif corner == 1: x1, y1 = w - cutout_w, 0
            elif corner == 2: x1, y1 = 0, h - cutout_h
            else: x1, y1 = w - cutout_w, h - cutout_h
            
            img[:, y1:y1 + cutout_h, x1:x1 + cutout_w] = 0
            return img

class AlzheimersDataset(Dataset):
    """
    Custom Dataset for loading Alzheimer's MRI scans.
    It expects a directory structure like:
    root_dir/
    ├── AD/
    │   ├── image1.jpeg
    │   └── ...
    └── NC/
        ├── image1.jpeg
        └── ...
    """
    def __init__(self, root_dir, transform=None, is_augmented=False, augmentation_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_augmented (bool): Flag to determine if augmentation should be applied.
            augmentation_transform (callable, optional): Augmentation transform.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = {'NC': 0, 'AD': 1}
        self.is_augmented = is_augmented
        self.augmentation_transform = augmentation_transform

        for class_name in self.classes.keys():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, file_name)
                    self.samples.append((image_path, self.classes[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label = self.samples[idx]
        
        # Load image and convert to greyscale
        image = Image.open(img_path).convert("L")

        if self.is_augmented and self.augmentation_transform:
            image = self.augmentation_transform(image)
        elif self.transform:
            image = self.transform(image)
            
        return image, label

# This block allows you to test the dataset script directly
if __name__ == '__main__':
    # Define the transformations for a pre-trained model
    # Using standard ImageNet normalization values
    data_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # --- IMPORTANT ---
    # Change this path to your actual test dataset directory
    test_data_dir = os.path.join(os.path.dirname(__file__), 'ADNI', 'AD_NC', 'test')
    
    print(f"Attempting to load data from: {test_data_dir}")

    # Create an instance of the dataset
    alz_dataset = AlzheimersDataset(root_dir=test_data_dir, transform=data_transform)

    # Check if the dataset was loaded
    if len(alz_dataset) > 0:
        print(f"Successfully loaded {len(alz_dataset)} images.")

        # Create a DataLoader
        dataloader = DataLoader(alz_dataset, batch_size=4, shuffle=True, num_workers=0)

        # Get one batch of training images
        try:
            images, labels = next(iter(dataloader))

            print(f"Batch of images has shape: {images.shape}")
            print(f"Batch of labels has shape: {labels.shape}")
            print(f"Labels in the batch: {labels}")
        except Exception as e:
            print(f"Error creating or iterating through DataLoader: {e}")
            print("This might happen if the dataset path is incorrect or the directory is empty.")

    else:
        print("Failed to load dataset. Please check the 'test_data_dir' path in this script.")
        print("Expected to find 'AD' and 'NC' subdirectories in that path.")
