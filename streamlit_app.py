import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image

# Import all necessary classes and functions from your project files
from teeth_whitener.detection.face_detector import MouthDetector
from teeth_whitener.segmentation.predictor import get_teeth_mask
from teeth_whitener.enhancement.lab_whitening import LABTeethWhitener

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Teeth Whitener",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Caching ---
# Use st.cache_resource to load models only once, speeding up the app
@st.cache_resource
def load_models():
    """Loads and returns the initialized models."""
    try:
        detector = MouthDetector()
        whitener = LABTeethWhitener()
        return detector, whitener
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# --- Main Application Logic ---
def main():
    st.title("🦷 AI Teeth Whitening Pipeline")
    st.markdown("Upload a photo to automatically detect teeth and apply a realistic whitening effect.")

    # Load the models
    with st.spinner("Initializing models... This might take a moment on first run."):
        detector, whitener = load_models()

    if not detector or not whitener:
        st.stop()

    # --- Sidebar for Upload and Controls ---
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.info("For best results, use a high resolution, well-lit photo.")

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image_bgr = cv2.imdecode(file_bytes, 1)
        original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)

        st.header("Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(original_image_rgb, caption="Original Image", use_container_width=True)

        with st.spinner("Processing image..."):
            # --- Run the Full Pipeline (adapted for Streamlit) ---
            final_image_bgr = original_image_bgr.copy()
            combined_mask = np.zeros(original_image_bgr.shape[:2], dtype=np.uint8)
            teeth_detected = False

            # 1. Detect mouth regions
            mouth_regions = detector.detect_mouth_regions(original_image_bgr, padding_factor=0.4)

            if not mouth_regions:
                st.warning("No faces or mouths were detected in the image.")
                return

            # 2. Process each detected mouth
            for region in mouth_regions:
                cropped_mouth_bgr = region['cropped_image']
                x_min, y_min, x_max, y_max = region['bbox']

                # Use a temporary file to work with get_teeth_mask
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    cv2.imwrite(tmp.name, cropped_mouth_bgr)
                    temp_image_path = tmp.name
                
                try:
                    # 3. Get the teeth mask for the cropped mouth
                    mask = get_teeth_mask(temp_image_path)
                    mask = (mask * 255).astype(np.uint8)

                    if np.sum(mask) > 0:
                        teeth_detected = True
                        # 4. Apply adaptive whitening
                        whitened_crop_bgr = whitener.apply_adaptive_whitening(cropped_mouth_bgr, mask)
                        # 5. Place the whitened crop back into the final image
                        final_image_bgr[y_min:y_max, x_min:x_max] = whitened_crop_bgr
                        combined_mask[y_min:y_max, x_min:x_max] = mask
                finally:
                    os.remove(temp_image_path)

            # --- Display the results ---
            if not teeth_detected:
                with col2:
                    st.info("No teeth were detected in the detected mouth regions.")
                with col3:
                    st.info("No whitening was applied.")
            else:
                final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)
                with col2:
                    st.image(combined_mask, caption="Predicted Teeth Mask", use_container_width=True)
                with col3:
                    st.image(final_image_rgb, caption="Whitened Teeth Image", use_container_width=True)
                
                st.success("Processing complete!")

if __name__ == '__main__':
    main()