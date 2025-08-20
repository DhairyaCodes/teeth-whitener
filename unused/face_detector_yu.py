"""
Face and mouth detection using RetinaFace detector.
Detects faces and extracts mouth landmarks for cropping.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import urllib.request
import os


class MouthDetector:
    """
    Face detector that identifies mouth regions using YUNet.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the mouth detector.
        
        Args:
            model_path: Path to YUNet model. If None, downloads automatically.
        """
        self.model_path = model_path
        
        # Verify model file exists and is valid
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Check file size (should be around 1.3MB for YUNet model)
        file_size = os.path.getsize(self.model_path)
        
        try:
            self.detector = cv2.FaceDetectorYN.create(self.model_path, "", (0, 0))
        except cv2.error as e:
            print(f"Error loading YUNet model: {e}")
            print("This might be due to a corrupted model file.")
            print("Try deleting the models/ directory and running again.")
            raise        
    
    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face detections, each containing [x, y, w, h, landmarks...]
        """
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        
        # Detect faces
        _, faces = self.detector.detect(image)
        
        if faces is None:
            return []
            
        return faces.tolist()
    
    def extract_mouth_landmarks(self, face_detection: np.ndarray) -> dict:
        """
        Extract mouth landmarks from face detection.
        
        Args:
            face_detection: Single face detection array
            
        Returns:
            Dictionary containing mouth landmark coordinates
        """
        # YUNet returns 15 values: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        # Where:
        # re = right eye, le = left eye, nt = nose tip
        # rcm = right corner of mouth, lcm = left corner of mouth
        
        if len(face_detection) < 15:
            raise ValueError("Invalid face detection format")
            
        landmarks = {
            'right_eye': (int(face_detection[4]), int(face_detection[5])),
            'left_eye': (int(face_detection[6]), int(face_detection[7])),
            'nose_tip': (int(face_detection[8]), int(face_detection[9])),
            'right_mouth_corner': (int(face_detection[10]), int(face_detection[11])),
            'left_mouth_corner': (int(face_detection[12]), int(face_detection[13])),
            'confidence': face_detection[14]
        }
        
        return landmarks
    
    def get_mouth_bounding_box(self, landmarks: dict, padding_factor: float = 0.3) -> Tuple[int, int, int, int]:
        """
        Calculate mouth bounding box with padding.
        
        Args:
            landmarks: Dictionary of facial landmarks
            padding_factor: Padding factor (0.2-0.5 recommended)
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        left_corner = landmarks['left_mouth_corner']
        right_corner = landmarks['right_mouth_corner']
        
        # Estimate top and bottom of mouth based on mouth corners and nose
        nose_tip = landmarks['nose_tip']
        
        # Calculate mouth dimensions
        mouth_width = abs(right_corner[0] - left_corner[0])
        mouth_height = int(mouth_width * 0.6)  # Approximate mouth height
        
        # Find mouth center
        mouth_center_x = (left_corner[0] + right_corner[0]) // 2
        mouth_center_y = (left_corner[1] + right_corner[1]) // 2
        
        # Adjust center slightly below nose tip for better mouth capture
        mouth_center_y = max(mouth_center_y, nose_tip[1] + int(mouth_height * 0.3))
        
        # Calculate initial bounding box
        x_min = mouth_center_x - mouth_width // 2
        x_max = mouth_center_x + mouth_width // 2
        y_min = mouth_center_y - mouth_height // 2
        y_max = mouth_center_y + mouth_height // 2
        
        # Add padding
        pad_x = int(mouth_width * padding_factor)
        pad_y = int(mouth_height * padding_factor)
        
        final_x_min = x_min - pad_x
        final_x_max = x_max + pad_x
        final_y_min = y_min - pad_y
        final_y_max = y_max + pad_y
        
        return final_x_min, final_y_min, final_x_max, final_y_max
    
    def crop_mouth_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop mouth region from image with bounds checking.
        
        Args:
            image: Input image
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            Tuple of (cropped_image, adjusted_bbox)
        """
        x_min, y_min, x_max, y_max = bbox
        height, width = image.shape[:2]
        
        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
        # Crop the mouth region
        cropped_mouth = image[y_min:y_max, x_min:x_max]
        
        return cropped_mouth, (x_min, y_min, x_max, y_max)
    
    def detect_mouth_regions(self, image: np.ndarray, padding_factor: float = 0.3) -> List[dict]:
        """
        Complete pipeline to detect all mouth regions in an image.
        
        Args:
            image: Input image
            padding_factor: Padding factor for mouth cropping
            
        Returns:
            List of dictionaries containing mouth region info
        """
        faces = self.detect_faces(image)
        mouth_regions = []
        
        for i, face in enumerate(faces):
            try:
                landmarks = self.extract_mouth_landmarks(face)
                bbox = self.get_mouth_bounding_box(landmarks, padding_factor)
                cropped_mouth, adjusted_bbox = self.crop_mouth_region(image, bbox)
                
                mouth_regions.append({
                    'face_id': i,
                    'landmarks': landmarks,
                    'bbox': adjusted_bbox,
                    'cropped_image': cropped_mouth,
                    'confidence': landmarks['confidence']
                })
                
            except Exception as e:
                print(f"Error processing face {i}: {e}")
                continue
                
        return mouth_regions
