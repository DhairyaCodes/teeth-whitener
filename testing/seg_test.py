import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from collections import OrderedDict


# --- 1. Define Model Architecture ---
# This must be the EXACT same architecture you used for training.
# Make sure to move the model to the correct device (CPU or GPU).

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,  # We are loading our own weights, so we don't need pre-trained ones.
    in_channels=3,
    classes=1,
)

# --- 2. Load Your Trained Weights ---
# Make sure the .pth file is in the same directory or provide the correct path.
MODEL_PATH = "../model-weights/best_teeth_segmentation_model.pth"
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Create a new dictionary without the "module." prefix
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

# Load the corrected weights
model.load_state_dict(new_state_dict)

model.to(DEVICE)
model.eval()  # Set the model to evaluation mode (very important!)

# --- 3. Create Preprocessing Transforms ---
# These must match the validation transforms from your training notebook.
IMG_SIZE = 256
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- 4. Create the Prediction Function ---
def get_teeth_mask(image_path):
    """
    Takes an image file path and returns the binary segmentation mask.
    """
    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original dimensions for resizing the mask later
    original_height, original_width = image.shape[:2]

    # Apply transformations
    augmented = val_transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE) # Add batch dimension

    # Get model prediction
    with torch.no_grad():
        prediction = model(image_tensor)
        
    # Post-process the prediction
    # Apply sigmoid to get probabilities, then threshold to get a binary mask
    prediction_prob = torch.sigmoid(prediction)
    binary_mask = (prediction_prob > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # Resize mask back to original image size
    resized_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    return resized_mask

# --- Example Usage ---
if __name__ == '__main__':
    # Replace 'path/to/your/image.jpg' with an actual image file
    input_image_path = '../images/c16.jpg'
    
    # Get the mask
    mask = get_teeth_mask(input_image_path)
    
    # To visualize the result (optional)
    import matplotlib.pyplot as plt
    
    original_image = cv2.imread(input_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Teeth Mask")
    
    plt.show()