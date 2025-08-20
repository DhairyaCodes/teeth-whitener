"""
Face and mouth detection using RetinaFace detector.
Detects faces and extracts mouth landmarks for cropping.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from retinaface import RetinaFace


class MouthDetector:
    """
    Face detector that identifies mouth regions using RetinaFace.
    """
    
    def __init__(self):
        """
        Initialize the mouth detector.
        """
        print("Using RetinaFace for face detection...")
        # RetinaFace will download models automatically on first use
        try:
            # Test RetinaFace with a dummy image to ensure it's working
            dummy_img = np.ones((100, 100, 3), dtype=np.uint8)
            RetinaFace.detect_faces(dummy_img)
            print("RetinaFace initialized successfully!")
        except Exception as e:
            print(f"RetinaFace initialization failed: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face detections with landmarks
        """
        return self._detect_faces_retinaface(image)
    
    def _detect_faces_retinaface(self, image: np.ndarray) -> List[dict]:
        """Detect faces using RetinaFace."""
        try:
            # RetinaFace expects RGB format
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = RetinaFace.detect_faces(rgb_image)
            
            face_list = []
            if isinstance(faces, dict):
                for face_key, face_data in faces.items():
                    if isinstance(face_data, dict) and 'facial_area' in face_data:
                        face_info = {
                            'bbox': face_data['facial_area'],  # [x1, y1, x2, y2]
                            'landmarks': face_data.get('landmarks', {}),
                            'confidence': face_data.get('score', 1.0)
                        }
                        face_list.append(face_info)
            
            return face_list
            
        except Exception as e:
            print(f"RetinaFace detection failed: {e}")
            return []
    
    def extract_mouth_landmarks(self, face_detection: dict) -> dict:
        """
        Extract mouth landmarks from face detection.
        
        Args:
            face_detection: Face detection dictionary
            
        Returns:
            Dictionary containing mouth landmark coordinates
        """
        landmarks = {
            'confidence': face_detection.get('confidence', 1.0)
        }
        
        if 'landmarks' in face_detection:
            # RetinaFace provides detailed landmarks
            rf_landmarks = face_detection['landmarks']
            
            if 'right_eye' in rf_landmarks:
                landmarks['right_eye'] = tuple(map(int, rf_landmarks['right_eye']))
            if 'left_eye' in rf_landmarks:
                landmarks['left_eye'] = tuple(map(int, rf_landmarks['left_eye']))
            if 'nose' in rf_landmarks:
                landmarks['nose_tip'] = tuple(map(int, rf_landmarks['nose']))
            if 'mouth_right' in rf_landmarks:
                landmarks['right_mouth_corner'] = tuple(map(int, rf_landmarks['mouth_right']))
            if 'mouth_left' in rf_landmarks:
                landmarks['left_mouth_corner'] = tuple(map(int, rf_landmarks['mouth_left']))
        
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
                
                # Check if we have the required landmarks for mouth detection
                required_landmarks = ['right_mouth_corner', 'left_mouth_corner', 'nose_tip']
                if not all(landmark in landmarks for landmark in required_landmarks):
                    print(f"Face {i}: Missing required landmarks for mouth detection")
                    continue
                
                bbox = self.get_mouth_bounding_box(landmarks, padding_factor)
                cropped_mouth, adjusted_bbox = self.crop_mouth_region(image, bbox)
                
                # Skip if cropped region is too small
                # if cropped_mouth.shape[0] < 20 or cropped_mouth.shape[1] < 20:
                #     print(f"Face {i}: Cropped mouth region too small")
                #     continue
                
                mouth_regions.append({
                    'face_id': i,
                    'landmarks': landmarks,
                    'bbox': adjusted_bbox,
                    'cropped_image': cropped_mouth,
                    'confidence': landmarks['confidence'],
                    'detection_method': 'retinaface'
                })
                
            except Exception as e:
                print(f"Error processing face {i}: {e}")
                continue
                
        return mouth_regions
