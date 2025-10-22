import torch
import torch.nn as nn
import timm

# You must have the 'timm' library installed for this to work.
# You can install it via pip: pip install timm

def create_convnext_model(num_classes=1, pretrained=True, model_name='convnext_tiny', in_chans=3):
    """
    Creates a ConvNeXt model for transfer learning.

    Args:
        num_classes (int): The number of output classes. For binary classification
                           with BCEWithLogitsLoss, this should be 1.
        pretrained (bool): Whether to load weights pre-trained on ImageNet.
        model_name (str): The name of the ConvNeXt model to create (e.g., 'convnext_tiny').
        in_chans (int): The number of input channels (3 for RGB, 1 for greyscale).

    Returns:
        A PyTorch model.
    """
    # Load a pre-trained ConvNeXt model.
    model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans)

    # Get the number of input features for the classifier
    num_ftrs = model.head.fc.in_features

    # Sequential layer containing dropout for regularization.
    model.head.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, num_classes)
    )

    return model


if __name__ == '__main__':
    # Create the model
    model = create_convnext_model(num_classes=1, model_name='convnext_small', in_chans=1)
    print("Model created successfully with 1 input channel.")
    # print(model)
