import argparse
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# --- MODIFICATION ---
# Import the new custom model creator instead of the old one
from modules import create_convnext_model as create_custom_convnext_model
# --- END MODIFICATION ---

def predict(args):
    # 1. SETUP
    # ============================================================================
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The class names
    class_names = ['Normal Control', 'Alzheimer\'s Disease']

    # 2. MODEL LOADING
    # ============================================================================
    
    # --- MODIFICATION ---
    # Create a new *custom* model instance
    # We use the default arch parameters which match the old 'convnext_small'
    print("Creating custom ConvNeXt model structure...")
    model = create_custom_convnext_model(
        num_classes=1, 
        in_chans=1
    )
    # --- END MODIFICATION ---
    
    # Load the saved state dictionary
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please make sure you have trained the model and the .pth file is in the correct directory.")
        return
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This may be because the saved model's architecture does not match the custom model architecture.")
        return

    # Move model to the device and set to evaluation mode
    model.to(device)
    model.eval()

    # 3. IMAGE PROCESSING
    # ============================================================================
    # Define the same transformations as used for validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load and transform the image
    try:
        image = Image.open(args.image_path).convert("L")
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        return
        
    # Apply transformations and add a batch dimension (B, C, H, W)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 4. PREDICTION
    # ============================================================================
    with torch.no_grad():
        output = model(image_tensor)
        # Apply sigmoid to the output logit to get a probability
        probability = torch.sigmoid(output).item()

    # Determine the predicted class and confidence
    if probability > 0.5:
        predicted_class = 1
        confidence = probability
    else:
        predicted_class = 0
        confidence = 1 - probability

    # 5. DISPLAY RESULTS
    # ============================================================================
    print(f"\nImage: {args.image_path}")
    print(f"Prediction: {class_names[predicted_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Alzheimer\'s from a single brain scan image.')
    
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--model-path', type=str, default='alzheimers_convnext_v2.pth', help='Path to the saved model weights (.pth file).')

    args = parser.parse_args()
    predict(args)

