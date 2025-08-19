import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import tempfile # Used for temporarily saving cropped images

# Add the project's source directory to the Python path
# This allows us to import modules from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all necessary classes and functions
from face_detector import MouthDetector
from predictor import get_teeth_mask
from lab_whitening import LABTeethWhitener

def run_full_pipeline(image_path: str, detector: MouthDetector, whitener: LABTeethWhitener):
    """
    Executes the full face detection, segmentation, and whitening pipeline.

    Args:
        image_path: The path to the input image.
        detector: An initialized MouthDetector instance.
        whitener: An initialized LABTeethWhitener instance.
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

    # Create a copy to place the final whitened results on
    final_image_bgr = original_image_bgr.copy()
    
    # 2. Detect and crop mouth regions
    print("Detecting mouth regions...")
    mouth_regions = detector.detect_mouth_regions(original_image_bgr, padding_factor=0.4)

    if not mouth_regions:
        print("No faces or mouths detected. Skipping.")
        return

    # Create a combined mask for visualization
    combined_mask = np.zeros(original_image_bgr.shape[:2], dtype=np.uint8)

    # 3. Process each detected mouth
    for i, region in enumerate(mouth_regions):
        print(f"  -> Processing mouth {i+1}/{len(mouth_regions)}...")
        cropped_mouth_bgr = region['cropped_image']
        x_min, y_min, x_max, y_max = region['bbox']

        # To use get_teeth_mask (which expects a file path), we save the crop temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, cropped_mouth_bgr)
            temp_image_path = tmp.name

        try:
            # 4. Get the teeth mask for the CROPPED mouth
            mask = get_teeth_mask(temp_image_path)
            mask = (mask * 255).astype(np.uint8)

            # Check if teeth were detected in this crop
            if np.sum(mask) == 0:
                continue

            # 5. Apply adaptive whitening to the CROPPED mouth
            whitened_crop_bgr = whitener.apply_adaptive_whitening(cropped_mouth_bgr, mask)

            # 6. Place the whitened crop back into the final image
            final_image_bgr[y_min:y_max, x_min:x_max] = whitened_crop_bgr
            
            # Add the individual mask to the combined mask for visualization
            combined_mask[y_min:y_max, x_min:x_max] = mask

        except Exception as e:
            print(f"     Error processing mouth crop: {e}")
        finally:
            # Clean up the temporary file
            os.remove(temp_image_path)

    if np.array_equal(final_image_bgr, original_image_bgr):
        print("No teeth detected in the image. Skipping whitening.")
        # Display only the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(original_image_rgb)
        plt.title(f"Original Image (No Teeth Detected)\n{os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()

        return

    # 7. Display the final results
    print("Displaying final results...")
    final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Final Results for: {os.path.basename(image_path)}", fontsize=16)

    axes[0].imshow(original_image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(combined_mask, cmap='gray')
    axes[1].set_title("Combined Teeth Mask")
    axes[1].axis('off')

    axes[2].imshow(final_image_rgb)
    axes[2].set_title("Final Whitened Image")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- How to run this script ---
    # In your terminal, from your project's root directory, use:
    # python src/run_pipeline.py --images "path/to/image1.jpg" "path/to/image2.jpg"
    
    parser = argparse.ArgumentParser(description="Run the full teeth whitening pipeline.")
    parser.add_argument(
        "--images",
        type=str,
        nargs='+',
        required=True,
        help="Paths to one or more input image files."
    )
    args = parser.parse_args()

    # Initialize detectors once to save time
    print("Initializing detectors... (This might take a moment)")
    try:
        mouth_detector = MouthDetector()
        teeth_whitener = LABTeethWhitener()
    except Exception as e:
        print(f"Failed to initialize a required module: {e}")
        sys.exit(1)

    # Loop through each provided image path and run the pipeline
    for image_path in args.images:
        run_full_pipeline(image_path, mouth_detector, teeth_whitener)
