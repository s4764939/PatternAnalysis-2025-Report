import torch
import torch.nn as nn
import timm

# You must have the 'timm' library installed for this to work.
# You can install it via pip: pip install timm

def create_convnext_model(num_classes=1, pretrained=True):
    """
    Creates a ConvNeXt model for transfer learning.

    Args:
        num_classes (int): The number of output classes. For binary classification
                           with BCEWithLogitsLoss, this should be 1.
        pretrained (bool): Whether to load weights pre-trained on ImageNet.

    Returns:
        A PyTorch model.
    """
    # Load a pre-trained ConvNeXt model. 'convnext_tiny' is a good starting point.
    model = timm.create_model('convnext_tiny', pretrained=pretrained)

    # Get the number of input features for the classifier
    num_ftrs = model.head.fc.in_features

    # Replace the model's final classification layer (the "head")
    # with a new linear layer for our specific number of classes.
    model.head.fc = nn.Linear(num_ftrs, num_classes)

    return model

# This block allows you to test the module script directly
if __name__ == '__main__':
    # Create the model
    model = create_convnext_model(num_classes=1)
    
    # Print the model architecture to verify the change
    # print(model)

    # Create a dummy input tensor to simulate a single image
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("Successfully created ConvNeXt model.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Pass the dummy input through the model
    output = model(dummy_input)
    
    print(f"\nShape of dummy input: {dummy_input.shape}")
    print(f"Shape of model output: {output.shape}")
    print("\nThis confirms the model is working correctly.")
    print("The output shape is [1, 1], which is expected for a batch of 1 image and 1 output class.")