"""
Simplified and adaptive LAB color space teeth whitening.
"""

import cv2
import numpy as np

class LABTeethWhitener:
    """
    Accomplishes natural teeth whitening using an adaptive, pixel-by-pixel
    approach in the LAB color space.
    """

    def apply_adaptive_whitening(self, image: np.ndarray, mask: np.ndarray,
                                 gamma: float = 0.85,
                                 yellowness_reduction_factor: float = 0.3,
                                 blur_kernel: int = 15) -> np.ndarray:
        """
        Applies adaptive whitening to the teeth region defined by the mask.

        This method brightens mid-tones more than highlights (preventing blowouts)
        and reduces yellowness proportionally to how yellow a pixel is.

        Args:
            image: The input image in BGR format.
            mask: The binary mask where teeth are located (255 for teeth).
            gamma: Controls lightness. Lower values mean stronger brightening (0.7-0.9 recommended).
            yellowness_reduction_factor: Percentage to reduce yellowness (0.1-0.4 recommended).
            blur_kernel: Kernel size for smoothing the mask edges for natural blending.

        Returns:
            The whitened image in BGR format.
        """
        # Create a copy of the image to work on
        whitened_image = image.copy()

        # Convert to LAB color space and split channels
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab_image)

        # Convert to float for calculations
        L_float = L.astype(np.float32)
        B_float = B.astype(np.float32)

        # Identify the pixels corresponding to teeth
        teeth_mask = (mask == 255)

        # --- 1. Adaptive Lightness (Gamma Correction) ---
        # This non-linear adjustment impacts mid-tones most, preserving highlights.
        L_teeth = L_float[teeth_mask]
        L_teeth_gamma = 255 * np.power(L_teeth / 255.0, gamma)
        L_float[teeth_mask] = L_teeth_gamma

        # --- 2. Adaptive Yellowness (Proportional Reduction) ---
        # This reduces the "distance" of each pixel from the neutral gray point (128).
        B_teeth = B_float[teeth_mask]
        B_teeth_new = B_teeth - (B_teeth - 128) * yellowness_reduction_factor
        B_float[teeth_mask] = B_teeth_new

        # Clip values to the valid 0-255 range and convert back to uint8
        L_new = np.clip(L_float, 0, 255).astype(np.uint8)
        B_new = np.clip(B_float, 0, 255).astype(np.uint8)

        # Merge the modified channels back
        whitened_lab = cv2.merge([L_new, A, B_new])
        whitened_bgr = cv2.cvtColor(whitened_lab, cv2.COLOR_LAB2BGR)

        # --- 3. Smooth Blending ---
        # Blur the mask to create a smooth transition between whitened and original areas
        smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_kernel, blur_kernel), 0) / 255.0

        # Combine the original and whitened images using the smooth mask
        # The np.newaxis is used to make the 2D mask compatible with the 3D image
        result = (image * (1 - smooth_mask[..., np.newaxis]) +
                  whitened_bgr * smooth_mask[..., np.newaxis]).astype(np.uint8)

        return result
