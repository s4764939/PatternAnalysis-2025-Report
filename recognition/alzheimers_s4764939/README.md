# Alzheimer's Disease Classification with ConvNeXt

## 1. Overview

This is accomplished by building a **ConvNeXt** model from scratch. The architecture is inspired by the "A ConvNet for the 2020s" paper and is tailored to the specifics of this classification task. Instead of relying on a pre-trained model, this project implements the ConvNeXt blocks layer-by-layer. The training process also adopts several modern techniques discussed in the paper to maximize performance.

An interesting side-effect of the model architecture is if you feed it an image that isn't a brain mri, it will have high confidence that it is a normal control image. This shows that our model hasn't learned to detect alzheimers by looking for features of a normal brain, but, looks for alzheimers by detecting features only in an alzheimers brain. This model requires .jpeg images that are imaged looking from the side of the brain, not front on or from above. 
 
## 2. Algorithm Description

The core of this project is a custom implementation of the ConvNeXt architecture, specifically mirroring the configuration of the `ConvNeXt-Small` variant. The model is not pre-trained but is built from the ground up, inspired by the design principles outlined in the "A ConvNet for the 2020s" paper.

The process works as follows:
1.  **Stem Layer:** The model begins with a "patchify" stem layer, consisting of a `4x4` convolution with a stride of 4. This aggressively downsamples the input image, similar to the patch embedding in Vision Transformers.
2.  **ConvNeXt Stages:** The model consists of four sequential stages, each containing multiple `ConvNeXtBlock` modules. The number of blocks per stage is `[3, 3, 9, 3]`, matching the ConvNeXt-S design.
3.  **ConvNeXt Block:** Each block contains a 7x7 depthwise convolution, followed by `LayerNorm`, and an inverted bottleneck block (a 1x1 convolution that expands channel dimensions by 4x, a `GELU` activation, and another 1x1 convolution to project it back). This design separates spatial and channel mixing.
4.  **Downsampling:** Between stages, separate downsampling layers (a `LayerNorm` followed by a `2x2` convolution with stride 2) are used to reduce the feature map resolution and increase the channel count.
5.  **Classification Head:** After the final stage, a global average pooling operation is performed, followed by a final `LayerNorm` and a single linear layer that outputs a logit for our binary (AD vs. NC) classification task.


## 3. Training Strategy

The model is trained using several modern techniques to improve performance and combat overfitting:

*   **Optimizer:** `AdamW` is used, which improves upon the standard Adam optimizer by decoupling weight decay from the gradient update.
*   **Scheduler:** A `CosineAnnealingLR` scheduler is used, which starts with a high learning rate and slowly anneals it down to a minimum. This is combined with a linear **Warmup** phase for the first few epochs to stabilize training.
*   **Loss Function:** A `BCEWithLogitsLoss` is used with a `pos_weight` to account for the class imbalance in the dataset.
*   **Regularization:**
    *   **Mixup:** A portion of training batches are created by linearly interpolating two different images and their labels, preventing the model from becoming overconfident.
    *   **Label Smoothing:** Labels are "smoothed" (e.g., 0.9 instead of 1.0) to further reduce overconfidence.
    *   **Stochastic Depth:** During training, entire `ConvNeXtBlock`s are randomly dropped (bypassed), forcing the network to learn redundant representations.
*   **Model Checkpointing:** An **Exponential Moving Average (EMA)** of the model's weights is maintained. This averaged model is used for validation and saved as the final best model, as it often provides better generalization.
*   **Mixed Precision:** `torch.amp` (Automatic Mixed Precision) is used to perform computations in float16, significantly speeding up training on compatible GPUs.

## 4. Training Performance & Analysis

### Hardware & Speed
*   **GPU:** NVIDIA RTX 5060 Ti (16GB VRAM)
*   **CPU:** Ryzen 7 7700 (8-core)
*   **Performance:** Training averaged approximately **2 minutes per epoch**.

### Results Analysis
The model was trained for an extended number of epochs, and the following behavior was observed from the validation metrics:

*   **Overfitting:** The training loss showed a consistent exponential decay throughout the run. However, the validation loss reached its minimum of ~0.52 at approximately epoch 50 and then began to increase, indicating the onset of overfitting.
*   **Precision/Recall Trade-off:** The best balance between precision and recall (both at approximately 0.8) was achieved between epochs 50 and 90. After this point, recall started to drop off while precision continued to increase, with the F1-score remaining relatively stable.
*   **Accuracy:** The validation accuracy followed a logarithmic curve, steadily improving throughout training and finally surpassing the 80% (0.8) threshold after epoch 120.

This analysis suggests that the best model, balancing all metrics, is likely found in the 50-90 epoch range, which aligns with the early stopping mechanism based on the F1-score. This is despite the fact that it doesn't acheive the exact 0.8+ threshold, and instead is only very close at approximately 0.77 to 0.78. 

## 5. Dependencies

To run this project, you need Python 3 and the following libraries. You can install them all using pip.

```bash
pip install torch torchvision scikit-learn Pillow tqdm numpy
```

Or, create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`:

```
torch
torchvision
scikit-learn
Pillow
tqdm
numpy
```

## 6. Dataset and Pre-processing

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
├── test/
│   ├── AD/
│   │   ├── image03.jpeg
│   │   └── ...
│   └── NC/
│       ├── image04.jpeg
│       └── ...
├─── train.py
├─── modules.py
└─── ...
```
The `train` directory is used for training the model, and the `test` directory is used for validation during training to save the best-performing version of the model.

### Pre-processing & Augmentation
Before being fed to the model, each image undergoes the following transformations:

1.  **Grayscale Conversion:** Images are loaded as grayscale (`L` mode).
2.  **Resize:** Images are resized to 224x224 pixels.
3.  **Data Augmentation (Training Only):** To improve model robustness, training images are subjected to a series of random augmentations:
    *   **Random Horizontal Flip:** Flipped horizontally with a 50% probability.
    *   **Random Affine:** Randomly rotated (±10 degrees), translated (±10%), and scaled (±10%).
    *   **Color Jitter:** Randomly adjusts brightness and contrast.
    *   **Random Erasing:** Randomly zeroes out a rectangular region of the image.
4.  **ToTensor:** Images are converted from PIL format to PyTorch tensors.
5.  **Normalization:** Tensors are normalized to have a mean of 0.5 and a standard deviation of 0.5.

## 7. How to Use

All commands should be run from the root directory containing the scripts.

### Training
To start training the model with the default hyperparameters, simply run:
```bash
python recognition/alzheimers_s4764939/train.py --data-dir /path/to/ADNI/AD_NC
```

You can tune the training run by passing different arguments. For example:
```bash
python recognition/alzheimers_s4764939/train.py --data-dir /path/to/ADNI/AD_NC --learning-rate 5e-4 --batch-size 32
```
The best performing model (based on validation F1-score) will be saved as `alzheimers_convnext_v2.pth`.

### Prediction
To classify a single image, use the `predict.py` script. You must provide a path to an image.
```bash
python recognition/alzheimers_s4764939/predict.py --image-path /path/to/your/image.jpeg
```
**Example Output:**
```
Using device: cuda
Creating custom ConvNeXt model structure...

Image: /path/to/your/image.jpeg
Prediction: Alzheimer's Disease
Confidence: 92.47%
```

## 8. Final Hyperparameters (Defaults)

*   **Model:** Custom ConvNeXt-Small (Depth: `[3,3,9,3]`, Dims: `[96,192,384,768]`)
*   **Learning Rate:** `4e-4`
*   **Weight Decay:** `0.05`
*   **Batch Size:** `64`
*   **Epochs:** `80` (with early stopping patience of 10)
*   **Label Smoothing:** `0.1`
*   **Mixup Alpha:** `0.8`
*   **Drop Path Rate:** `0.2`
*   **EMA Decay:** `0.9999`
*   **Warmup Epochs:** `3`
*   **Optimizer:** `AdamW`
*   **Scheduler:** `CosineAnnealingLR` (with warmup)

## 9. File Structure

*   `dataset.py`: Contains the `AlzheimersDataset` class, a custom PyTorch dataset that handles loading images and their corresponding labels from the specified directory structure.
*   `modules.py`: Defines the custom ConvNeXt model architecture, including the `ConvNeXtBlock` and `LayerNorm2d` helper classes.
*   `train.py`: The main script for training the model. It brings together the dataset and model, and implements the complete training loop with augmentation, mixed precision, EMA, Mixup, and validation.
*   `predict.py`: A script to perform inference on a single image using the trained model.
*   `README.md`: This file, providing documentation for the project.