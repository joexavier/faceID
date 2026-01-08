"""
MobileFaceNet Embedding Service

Uses InsightFace's w600k_mbf model converted to CoreML for face recognition:
- Input: 112x112 RGB, normalized to [-1, 1]
- Output: L2-normalized 512-D embedding
- Matching: Cosine similarity with thresholds (0.4 definitive, 0.3 probable)

Model source: InsightFace buffalo_s pack (w600k_mbf.onnx -> MobileFaceNet.mlpackage)
"""

import cv2
import numpy as np
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger('faceID.mobilefacenet')


class MobileFaceNetService:
    """
    MobileFaceNet embedding service using CoreML (native macOS).

    Key features:
    - 112x112 input with [-1, 1] normalization
    - 512-D L2-normalized embeddings
    - Face alignment preprocessing
    - Cosine similarity matching
    """

    # Matching thresholds for InsightFace model (cosine similarity)
    THRESHOLD_DEFINITIVE = 0.4  # Score >= 0.4: Definitive match
    THRESHOLD_PROBABLE = 0.3   # Score 0.3-0.39: Probable match

    def __init__(self):
        self.embedding_dim = 512
        self.input_size = (112, 112)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the CoreML model using native macOS framework"""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'models')
        mlmodelc_path = os.path.join(models_dir, 'MobileFaceNet.mlmodelc')
        mlpackage_path = os.path.join(models_dir, 'MobileFaceNet.mlpackage')

        # Prefer compiled model, fall back to mlpackage
        if os.path.exists(mlmodelc_path):
            model_path = mlmodelc_path
        elif os.path.exists(mlpackage_path):
            model_path = mlpackage_path
        else:
            logger.error(f"MobileFaceNet model not found in {models_dir}")
            self.model = None
            return

        try:
            # Use native CoreML via PyObjC
            import CoreML
            import Foundation

            url = Foundation.NSURL.fileURLWithPath_(model_path)
            config = CoreML.MLModelConfiguration.alloc().init()

            # Compile mlpackage if needed
            if model_path.endswith('.mlpackage'):
                logger.info("Compiling mlpackage...")
                result = CoreML.MLModel.compileModelAtURL_error_(url, None)
                if isinstance(result, tuple):
                    url, error = result
                    if error:
                        logger.error(f"Model compilation error: {error}")
                        self.model = None
                        return
                else:
                    url = result

            # Load the model
            # PyObjC returns (result, error) tuple for methods with _error_ parameter
            result = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
                url, config, None
            )

            if isinstance(result, tuple):
                self.model, error = result
                if error:
                    logger.error(f"CoreML model load error: {error}")
                    self.model = None
            else:
                self.model = result

            if self.model:
                logger.info("Loaded MobileFaceNet CoreML model (native)")
            else:
                logger.error("Failed to load CoreML model")
                self.model = None

        except ImportError:
            logger.warning("PyObjC not available, trying coremltools...")
            self._load_with_coremltools(mlpackage_path)
        except Exception as e:
            logger.error(f"Could not load MobileFaceNet model: {e}")
            self._load_with_coremltools(mlpackage_path)

    def _load_with_coremltools(self, mlpackage_path: str):
        """Fallback to coremltools if PyObjC fails"""
        try:
            import coremltools as ct
            self.model = ct.models.MLModel(mlpackage_path)
            self._use_coremltools = True
            logger.info("Loaded MobileFaceNet CoreML model (coremltools)")
        except Exception as e:
            logger.error(f"coremltools fallback failed: {e}")
            self.model = None

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for MobileFaceNet.

        - Resize to 112x112
        - Normalize: (pixel - 127.5) / 127.5 → maps [0,255] to [-1,1]
        - Convert to CHW format
        """
        # Resize
        if face_image.shape[:2] != self.input_size:
            face_image = cv2.resize(face_image, self.input_size)

        # Convert BGR to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = face_image

        # Normalize to [-1, 1]
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 127.5

        # Convert to CHW format (channels first)
        face_chw = np.transpose(face_normalized, (2, 0, 1))

        # Add batch dimension
        face_batch = np.expand_dims(face_chw, axis=0)

        return face_batch

    def _l2_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding to unit length"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _predict_native(self, face_input: np.ndarray) -> np.ndarray:
        """Run prediction using native CoreML"""
        import CoreML
        from Foundation import NSNumber, NSArray

        # Create MLMultiArray for input
        shape = NSArray.arrayWithArray_(list(face_input.shape))
        flat_data = face_input.astype(np.float32).flatten()

        input_array, error = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
            shape, CoreML.MLMultiArrayDataTypeFloat32, None
        )

        if error:
            raise RuntimeError(f"Failed to create MLMultiArray: {error}")

        # Set values using indexed subscript
        for i, val in enumerate(flat_data):
            ns_val = NSNumber.numberWithFloat_(float(val))
            input_array.setObject_atIndexedSubscript_(ns_val, i)

        # Create feature value and provider
        feature_value = CoreML.MLFeatureValue.featureValueWithMultiArray_(input_array)
        input_dict = {"input": feature_value}
        provider, error = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            input_dict, None
        )

        if error:
            raise RuntimeError(f"Failed to create feature provider: {error}")

        # Run prediction
        result, error = self.model.predictionFromFeatures_error_(provider, None)

        if error:
            raise RuntimeError(f"Prediction failed: {error}")

        # Get output
        output_names = list(result.featureNames())
        output_name = output_names[0] if output_names else "var_854"
        output_value = result.featureValueForName_(output_name)
        output_array = output_value.multiArrayValue()

        # Convert to numpy using indexed subscript
        count = int(output_array.count())
        embedding = np.array([float(output_array.objectAtIndexedSubscript_(i)) for i in range(count)], dtype=np.float32)

        return embedding

    def _predict_coremltools(self, face_input: np.ndarray) -> np.ndarray:
        """Run prediction using coremltools"""
        prediction = self.model.predict({'input': face_input})
        output_key = list(prediction.keys())[0]
        return prediction[output_key].flatten()

    def extract_embedding(self, image_path: str,
                          face_location: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Extract L2-normalized 512-D embedding.

        Args:
            image_path: Path to image file
            face_location: Face bbox (top, right, bottom, left)

        Returns:
            L2-normalized 512-D embedding or None
        """
        if self.model is None:
            logger.warning("MobileFaceNet model not loaded")
            return None

        from app.services.face_detection import load_image_with_exif_rotation

        # Load image with EXIF rotation
        pil_img = load_image_with_exif_rotation(image_path)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if face_location is not None:
            top, right, bottom, left = face_location
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

        # Use simple alignment (center crop)
        # Note: Eye-based alignment was tested but produced worse results
        # Proper 5-point landmark alignment would require a landmark detector
        face_aligned = self._align_face(face_image)

        # Preprocess
        face_input = self._preprocess(face_aligned)

        # Extract embedding
        try:
            if hasattr(self, '_use_coremltools') and self._use_coremltools:
                embedding = self._predict_coremltools(face_input)
            else:
                embedding = self._predict_native(face_input)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

        # L2 normalize
        embedding = self._l2_normalize(embedding)

        return embedding

    # Standard ArcFace alignment coordinates for 112x112
    ARCFACE_DST = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ], dtype=np.float32)

    def _align_face_with_landmarks(self, image: np.ndarray,
                                    face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Align face using eye detection for InsightFace model.

        Uses Haar cascade eye detection to find eyes, then aligns based on eye positions.
        """
        try:
            top, right, bottom, left = face_location
            h, w = image.shape[:2]

            # Expand face region slightly for better eye detection
            margin_h = int((bottom - top) * 0.2)
            margin_w = int((right - left) * 0.2)
            top_exp = max(0, top - margin_h)
            bottom_exp = min(h, bottom + margin_h)
            left_exp = max(0, left - margin_w)
            right_exp = min(w, right + margin_w)

            face_region = image[top_exp:bottom_exp, left_exp:right_exp]

            # Convert to grayscale for eye detection
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Use Haar cascade for eye detection
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

            if len(eyes) < 2:
                # Try with different parameters
                eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15))

            if len(eyes) >= 2:
                # Sort eyes by x coordinate (left to right)
                eyes = sorted(eyes, key=lambda e: e[0])

                # Get eye centers (relative to expanded face region)
                left_eye_x = eyes[0][0] + eyes[0][2] // 2
                left_eye_y = eyes[0][1] + eyes[0][3] // 2
                right_eye_x = eyes[1][0] + eyes[1][2] // 2
                right_eye_y = eyes[1][1] + eyes[1][3] // 2

                # Convert to full image coordinates
                left_eye = np.array([left_exp + left_eye_x, top_exp + left_eye_y], dtype=np.float32)
                right_eye = np.array([left_exp + right_eye_x, top_exp + right_eye_y], dtype=np.float32)

                # Estimate other points based on eye positions
                eye_center = (left_eye + right_eye) / 2
                eye_dist = np.linalg.norm(right_eye - left_eye)

                # Standard ratios for face geometry
                nose = eye_center + np.array([0, eye_dist * 0.6], dtype=np.float32)
                left_mouth = left_eye + np.array([eye_dist * 0.1, eye_dist * 1.2], dtype=np.float32)
                right_mouth = right_eye + np.array([-eye_dist * 0.1, eye_dist * 1.2], dtype=np.float32)

                src_pts = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)

                # Compute similarity transform
                tform = self._estimate_similarity_transform(src_pts, self.ARCFACE_DST)

                # Apply transform
                aligned = cv2.warpAffine(image, tform, (112, 112), borderValue=0)

                return aligned

            return None

        except Exception as e:
            logger.warning(f"Landmark alignment failed: {e}, falling back to simple crop")
            return None

    def _estimate_similarity_transform(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Estimate similarity transform matrix from src to dst points."""
        num = src.shape[0]
        dim = src.shape[1]

        # Center the points
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        # Compute scale
        src_std = np.sqrt(np.sum(src_centered ** 2) / num)
        dst_std = np.sqrt(np.sum(dst_centered ** 2) / num)
        scale = dst_std / src_std

        # Normalize
        src_norm = src_centered / src_std
        dst_norm = dst_centered / dst_std

        # Compute rotation using SVD
        H = np.dot(src_norm.T, dst_norm)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Build transform matrix
        T = np.eye(3)
        T[:2, :2] = scale * R
        T[:2, 2] = dst_mean - scale * np.dot(R, src_mean)

        return T[:2, :]

    def _align_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Simple face alignment (fallback when landmarks not available).
        Just resize and center-pad to 112x112.
        """
        h, w = face_image.shape[:2]

        # Calculate scale to fit 112x112
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(face_image, (new_w, new_h))

        # Center crop or pad to 112x112
        result = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)

        y_offset = (self.input_size[1] - new_h) // 2
        x_offset = (self.input_size[0] - new_w) // 2

        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return result

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.

        For L2-normalized vectors, cosine similarity = dot product.
        """
        return float(np.dot(emb1, emb2))

    def match_score(self, embedding: np.ndarray, reference: np.ndarray) -> Tuple[str, float]:
        """
        Match embedding against reference using thresholds.

        Returns:
            Tuple of (match_type, score) where match_type is:
            - "definitive": score >= 0.4
            - "probable": score 0.3-0.39
            - "no_match": score < 0.3
        """
        score = self.cosine_similarity(embedding, reference)

        if score >= self.THRESHOLD_DEFINITIVE:
            return ("definitive", score)
        elif score >= self.THRESHOLD_PROBABLE:
            return ("probable", score)
        else:
            return ("no_match", score)
