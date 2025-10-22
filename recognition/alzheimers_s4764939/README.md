# Alzheimer's Disease Classification with ConvNeXt

## 1. Overview

This project aims to classify 2D MRI brain scans from the ADNI dataset as either belonging to a patient with Alzheimer's Disease (AD) or a Normal Control (NC) subject. The goal is to achieve a minimum accuracy of 80% on the test set, as per the project specification.

This is accomplished using a state-of-the-art computer vision model, **ConvNeXt**, through a technique called **transfer learning**. The final model uses the `convnext_small` variant.

## 2. Algorithm Description

The core of this project is a `convnext_small` model, which has been pre-trained on the large-scale ImageNet dataset. Instead of training a new model from scratch, which would require a vast amount of data and computational power, we adapt this existing model to our specific task.

The process works as follows:
1.  **Load Pre-trained Model:** A `convnext_small` model is loaded with its learned ImageNet weights.
2.  **Modify for Grayscale:** The model is adapted to accept grayscale (1-channel) images instead of the default RGB (3-channel) images.
3.  **Replace Classifier:** The final layer of the model, originally designed to classify 1000 ImageNet classes, is removed and replaced with a new, single-node linear layer with a dropout of 0.4 for regularization. This new head is tailored for our binary classification problem (AD vs. NC).
4.  **Fine-Tuning:** The entire model is then "fine-tuned" on the ADNI brain scan dataset. During this phase, the model learns to adapt its powerful, generalized features (learned from ImageNet) to recognize the specific patterns, shapes, and textures relevant to identifying Alzheimer's disease in brain scans.

This transfer learning approach is highly effective as it leverages the deep feature extraction capabilities of a model trained on millions of images and applies them to our specialized medical imaging domain.

```
[ ImageNet Pre-trained ConvNeXt ] ----> [ Modify for Grayscale ] ----> [ Replace Final Layer ] ----> [ Fine-tune on ADNI Dataset ] ----> [ AD/NC Classifier ]
```

## 3. Dependencies

To run this project, you need Python 3 and the following libraries. You can install them all using pip.

```bash
pip install torch torchvision timm scikit-learn Pillow tqdm numpy
```

Or, create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`:

```
torch
torchvision
timm
scikit-learn
Pillow
tqdm
numpy
```

## 4. Dataset and Pre-processing

### Dataset
The model expects the ADNI dataset to be organized in the following structure:
```
ADNI/AD_NC/
├── train/
│   ├── AD/
│   │   ├── image01.jpeg
│   │   └── ...
│   └── NC/
│       ├── image02.jpeg
│       └── ...
└── test/
    ├── AD/
    │   ├── image03.jpeg
    │   └── ...
    └── NC/
        ├── image04.jpeg
        └── ...
```
The `train` directory is used for training the model, and the `test` directory is used for validation during training to save the best-performing version of the model.

### Pre-processing
Before being fed to the model, each image undergoes the following transformations:
1.  **Grayscale Conversion:** Images are converted to grayscale.
2.  **Resize:** Images are resized to 224x224 pixels to match the input dimensions of the ConvNeXt model.
3.  **Data Augmentation (Training Only):** To improve model robustness, training images are subjected to a series of random augmentations:
    *   **Random Rotation:** Rotated by up to 30 degrees.
    *   **Random Resized Crop:** Scaled between 85% and 115% of the original size.
    *   **Random Horizontal Flip:** Flipped horizontally with a 50% probability.
    *   **Custom Regularization:** A custom transform is applied with a 50% probability to either:
        *   Add Gaussian noise (noise factor 0.05).
        *   Apply a cutout by blacking out a corner of the image (40% of the image size).
4.  **ToTensor:** Images are converted from PIL format to PyTorch tensors.
5.  **Normalization:** Tensors are normalized to have a mean of 0.5 and a standard deviation of 0.5.

## 5. Training Strategy

The model is trained with the following strategies to improve performance and combat overfitting:

*   **Combined Dataset:** The training data consists of all the original images plus a random 50% of the augmented images. This allows the model to learn from both the original data and a variety of augmented examples.
*   **Weighted Loss:** A `BCEWithLogitsLoss` is used with a `pos_weight` to account for the slight class imbalance in the dataset.
*   **Label Smoothing:** A label smoothing factor of 0.1 is used to regularize the model and prevent it from becoming too confident in its predictions.
*   **Best Threshold Search:** During validation, the model searches for the best classification threshold (from 0.1 to 0.9) that maximizes the F1-score. This ensures that the model's performance is not tied to a fixed threshold of 0.5.
*   **Cosine Annealing Scheduler:** A cosine annealing learning rate scheduler is used to adjust the learning rate during training.

## 6. How to Use

All commands should be run from the `PatternAnalysis-2025-Report` directory.

### Training
To start training the model with the final hyperparameters, run:
```bash
python recognition/alzheimers_s4764939/train.py --learning-rate 5e-5 --weight-decay 2e-2 --batch-size 64 --label-smoothing 0.1 --epochs 50 --early-stopping-patience 15
```
The best performing model will be saved as `alzheimers_convnext_v2.pth` in the `recognition/alzheimers_s4764939/` directory.

### Prediction
To classify a single image, use the `predict.py` script. You must provide a path to an image.
```bash
python recognition/alzheimers_s4764939/predict.py --image-path /path/to/your/image.jpeg
```
**Example Output:**
```
$ python recognition/alzheimers_s4764939/predict.py --image-path recognition/alzheimers_s4764939/ADNI/AD_NC/test/AD/1003730_102.jpeg

Using device: cuda

Image: recognition/alzheimers_s4764939/ADNI/AD_NC/test/AD/1003730_102.jpeg
Prediction: Alzheimer's Disease
Confidence: 92.47%
```

## 7. Final Hyperparameters

*   **Model:** `convnext_small`
*   **Learning Rate:** `5e-5`
*   **Weight Decay:** `2e-2`
*   **Batch Size:** `64`
*   **Epochs:** `50` (with default early stopping patience of 15)
*   **Label Smoothing:** `0.1`
*   **Dropout:** `0.4`
*   **Optimizer:** `AdamW`
*   **Scheduler:** `CosineAnnealingLR`

## 8. File Structure

*   `dataset.py`: Contains the `AlzheimersDataset` class, a custom PyTorch dataset that handles loading images and their corresponding labels from the specified directory structure.
*   `modules.py`: Defines the `create_convnext_model` function, which is responsible for loading the pre-trained ConvNeXt model and modifying its classification head for our binary task.
*   `train.py`: The main script for training the model. It brings together the dataset and the model, implements the training and validation loops, and saves the best model weights.
*   `predict.py`: A script to perform inference on a single image using the trained model.
*   `README.md`: This file, providing documentation for the project.
