import cv2
import matplotlib.pyplot as plt
from face_detector import MouthDetector

# Load an image
image = cv2.imread("images/i16.jpg")

# Initialize detector
detector = MouthDetector()

# Run detection
mouth_regions = detector.detect_mouth_regions(image, padding_factor=0.5)

# Create a copy of the image to draw on, preserving the original
annotated_image = image.copy()

# Draw bounding boxes on the copy
for region in mouth_regions:
    x_min, y_min, x_max, y_max = region['bbox']
    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# --- Display results using Matplotlib ---

# Create a figure with a controlled size
# The number of rows will adjust based on how many mouths are found
num_mouths = len(mouth_regions)
fig, axes = plt.subplots(1, 1 + num_mouths, figsize=(15, 5))

# 1. Display the main annotated image
# Convert from BGR (OpenCV) to RGB (Matplotlib) for correct colors
axes[0].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Detected Mouths")
axes[0].axis('off')

# 2. Display each cropped mouth image
for i, region in enumerate(mouth_regions):
    ax = axes[i + 1]
    cropped_rgb = cv2.cvtColor(region['cropped_image'], cv2.COLOR_BGR2RGB)
    ax.imshow(cropped_rgb)
    ax.set_title(f"Mouth {region['face_id']}")
    ax.axis('off')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()