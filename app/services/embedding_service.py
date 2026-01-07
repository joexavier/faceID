import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import urllib.request
import logging

from app.services.face_detection import load_image_with_exif_rotation

logger = logging.getLogger('faceID.embedding')


class EmbeddingService:
    """Service for extracting face embeddings using OpenCV DNN"""

    # OpenFace model for 128D embeddings (same as dlib/face_recognition)
    MODEL_URL = "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7"

    def __init__(self):
        self.embedding_dim = 128
        self.default_threshold = 0.6
        self.net = None
        self._ensure_model()

    def _ensure_model(self):
        """Download and load the embedding model if not present"""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, 'openface_nn4.small2.v1.t7')

        # Download if not exists
        if not os.path.exists(model_path):
            logger.info("Downloading face embedding model...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
            except Exception as e:
                logger.warning(f"Could not download embedding model: {e}. Using fallback.")
                self.net = None
                return

        # Load the model
        try:
            self.net = cv2.dnn.readNetFromTorch(model_path)
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            self.net = None

    def _extract_openface_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding using OpenFace model"""
        # Resize to 96x96 as required by OpenFace
        face_blob = cv2.dnn.blobFromImage(
            face_image, 1.0 / 255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False
        )

        self.net.setInput(face_blob)
        embedding = self.net.forward()

        return embedding.flatten()

    def _extract_fallback_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract simple histogram-based embedding as fallback"""
        # Resize to standard size
        face_resized = cv2.resize(face_image, (64, 64))

        # Convert to grayscale
        if len(face_resized.shape) == 3:
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_resized

        # Extract LBP-like features
        # Simple approach: compute histogram of gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude and angle
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi

        # Create histogram features from 4x4 grid
        features = []
        cell_size = 16
        for i in range(4):
            for j in range(4):
                cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                cell_ang = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

                # 8-bin histogram
                hist, _ = np.histogram(cell_ang, bins=8, range=(-180, 180), weights=cell_mag)
                features.extend(hist / (np.sum(hist) + 1e-6))

        # Pad to 128 dimensions
        features = np.array(features)
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))

        return features[:128]

    def extract_embedding(self, image_path: str,
                          face_location: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Extract 128D embedding for a single face.

        Automatically handles EXIF orientation for consistent embeddings.

        Args:
            image_path: Path to image file
            face_location: Optional face location (top, right, bottom, left)

        Returns:
            128-dimensional numpy array or None if no face found
        """
        # Load image with EXIF rotation applied
        pil_img = load_image_with_exif_rotation(image_path)

        # Convert to RGB if necessary
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Convert PIL to OpenCV format (BGR)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if image is None:
            return None

        if face_location is not None:
            top, right, bottom, left = face_location
            # Ensure valid coordinates
            h, w = image.shape[:2]
            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)
            face_image = image[top:bottom, left:right]
        else:
            face_image = image

        if face_image.size == 0:
            return None

        # Extract embedding
        if self.net is not None:
            try:
                return self._extract_openface_embedding(face_image)
            except Exception:
                pass

        return self._extract_fallback_embedding(face_image)

    def extract_all_embeddings(self, image_path: str,
                                face_locations: Optional[List[Tuple[int, int, int, int]]] = None) -> List[np.ndarray]:
        """
        Extract embeddings for all faces in an image.

        Args:
            image_path: Path to image file
            face_locations: Optional list of face locations

        Returns:
            List of 128-dimensional numpy arrays
        """
        if not face_locations:
            return []

        embeddings = []
        for loc in face_locations:
            emb = self.extract_embedding(image_path, loc)
            if emb is not None:
                embeddings.append(emb)

        return embeddings

    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(embedding1 - embedding2))

    def compute_centroid(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute centroid (mean) of multiple embeddings.

        Args:
            embeddings: List of embeddings

        Returns:
            Centroid embedding
        """
        return np.mean(embeddings, axis=0)

    def are_same_person(self, embedding1: np.ndarray, embedding2: np.ndarray,
                        threshold: float = None) -> Tuple[bool, float]:
        """
        Check if two embeddings are from the same person.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Distance threshold (default: 0.6)

        Returns:
            Tuple of (is_same_person, distance)
        """
        if threshold is None:
            threshold = self.default_threshold

        distance = self.compute_distance(embedding1, embedding2)
        return distance < threshold, distance

    def find_best_match(self, target_embedding: np.ndarray,
                        candidate_embeddings: List[np.ndarray]) -> Tuple[int, float]:
        """
        Find the best matching embedding from a list of candidates.

        Args:
            target_embedding: Target embedding to match
            candidate_embeddings: List of candidate embeddings

        Returns:
            Tuple of (best_match_index, distance)
        """
        if not candidate_embeddings:
            return -1, float('inf')

        distances = [self.compute_distance(target_embedding, e) for e in candidate_embeddings]
        best_idx = np.argmin(distances)

        return int(best_idx), float(distances[best_idx])
