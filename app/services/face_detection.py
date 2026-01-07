import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Tuple, Optional
import urllib.request
import logging

logger = logging.getLogger('faceID.detection')


class FaceDetectionService:
    """Service for detecting faces in images using OpenCV DNN"""

    # URLs for OpenCV's pre-trained face detection model
    PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize face detection service.

        Args:
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.net = None
        self._ensure_model()

    def _ensure_model(self):
        """Download and load the face detection model if not present"""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'models')
        os.makedirs(models_dir, exist_ok=True)

        prototxt_path = os.path.join(models_dir, 'deploy.prototxt')
        model_path = os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

        # Download if not exists
        if not os.path.exists(prototxt_path):
            logger.info("Downloading face detection model config...")
            urllib.request.urlretrieve(self.PROTOTXT_URL, prototxt_path)

        if not os.path.exists(model_path):
            logger.info("Downloading face detection model weights...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)

        # Load the model
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    def detect_faces(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Args:
            image_path: Path to image file

        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        (h, w) = image.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        # Run face detection
        self.net.setInput(blob)
        detections = self.net.forward()

        face_locations = []

        # Process detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure valid coordinates
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Convert to (top, right, bottom, left) format
                face_locations.append((startY, endX, endY, startX))

        return face_locations

    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Get image width and height.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (width, height)
        """
        with Image.open(image_path) as img:
            return img.size

    def create_thumbnail(self, image_path: str, output_path: str, size: Tuple[int, int] = (300, 300)) -> str:
        """
        Create a thumbnail of an image.

        Args:
            image_path: Path to source image
            output_path: Path to save thumbnail
            size: Thumbnail size (width, height)

        Returns:
            Path to created thumbnail
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with Image.open(image_path) as img:
            # Handle EXIF orientation
            try:
                from PIL import ExifTags
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass

            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Create thumbnail maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Save
            img.save(output_path, 'JPEG', quality=85)

        return output_path

    def extract_face_region(self, image_path: str, bbox: Tuple[int, int, int, int],
                            padding: float = 0.2) -> Image.Image:
        """
        Extract a face region from an image with optional padding.

        Args:
            image_path: Path to image file
            bbox: Face bounding box (top, right, bottom, left)
            padding: Padding factor (0.2 = 20% padding on each side)

        Returns:
            PIL Image of the face region
        """
        top, right, bottom, left = bbox

        with Image.open(image_path) as img:
            width, height = img.size

            # Calculate padding
            face_width = right - left
            face_height = bottom - top
            pad_x = int(face_width * padding)
            pad_y = int(face_height * padding)

            # Apply padding with bounds checking
            new_left = max(0, left - pad_x)
            new_top = max(0, top - pad_y)
            new_right = min(width, right + pad_x)
            new_bottom = min(height, bottom + pad_y)

            # Crop
            face_img = img.crop((new_left, new_top, new_right, new_bottom))

            return face_img.copy()
