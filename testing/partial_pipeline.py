import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

from predictor import get_teeth_mask
from lab_whitening import LABTeethWhitener

def run_whitening_pipeline(image_path: str):
    """
    Executes the full teeth segmentation and whitening pipeline for a single image.

    Args:
        image_path: The path to the input image.
    """
    print(f"\nProcessing image: {image_path}")

    # 1. Load the original image
    try:
        original_image_bgr = cv2.imread(image_path)
        if original_image_bgr is None:
            raise FileNotFoundError(f"Could not read the image file at {image_path}")
        original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Get the teeth mask from the segmentation model
    print("Generating teeth mask...")
    try:
        mask = get_teeth_mask(image_path)
        # Ensure mask is a binary (0 or 255) uint8 image
        mask = (mask * 255).astype(np.uint8)
    except Exception as e:
        print(f"Error during mask generation: {e}")
        return
        
    # Check if any teeth were detected
    if np.sum(mask) == 0:
        print("No teeth detected in the image. Skipping whitening.")
        # Display only the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(original_image_rgb)
        plt.title(f"Original Image (No Teeth Detected)\n{os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()
        return

    # 3. Apply teeth whitening using the simplified lab_whitening script
    print("Applying adaptive teeth whitening...")
    whitener = LABTeethWhitener()
    # Using the corrected, simplified function name
    whitened_image_bgr = whitener.apply_adaptive_whitening(original_image_bgr, mask)
    whitened_image_rgb = cv2.cvtColor(whitened_image_bgr, cv2.COLOR_BGR2RGB)

    # 4. Display the results for the current image
    print("Displaying results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Results for: {os.path.basename(image_path)}", fontsize=16)

    axes[0].imshow(original_image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Predicted Teeth Mask")
    axes[1].axis('off')

    axes[2].imshow(whitened_image_rgb)
    axes[2].set_title("Whitened Image")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- How to run this script ---
    # In your terminal, from your project's root directory, use:
    # python src/run_pipeline.py --images "path/to/image1.jpg" "path/to/image2.jpg"
    
    parser = argparse.ArgumentParser(description="Run the teeth segmentation and whitening pipeline on one or more images.")
    parser.add_argument(
        "--images",
        type=str,
        nargs='+',  # This allows for multiple arguments
        required=True,
        help="Paths to one or more input image files (up to 5)."
    )
    args = parser.parse_args()
    
    for image_path in args.images:
        run_whitening_pipeline(image_path)

