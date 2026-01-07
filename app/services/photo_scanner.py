import os
from typing import List, Dict, Optional, Generator
from datetime import datetime
from PIL import Image
import logging

from app import db

logger = logging.getLogger('faceID.scanner')
from app.models.database import Photo, DetectedFace, ScanResult, Classifier
from app.services.face_detection import FaceDetectionService
from app.services.embedding_service import EmbeddingService


class PhotoScanner:
    """Service for scanning photos and managing the photo database"""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def __init__(self, photos_dir: str):
        """
        Initialize scanner with photos directory.

        Args:
            photos_dir: Path to photos directory
        """
        self.photos_dir = photos_dir
        self.face_detector = FaceDetectionService()
        self.embedding_service = EmbeddingService()

    def discover_photos(self) -> Generator[Dict, None, None]:
        """
        Discover all photos in the photos directory.

        Yields:
            Dictionary with photo info
        """
        for root, dirs, files in os.walk(self.photos_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, filename)
                    folder_name = os.path.basename(root)

                    # Skip thumbnails folder
                    if folder_name == 'thumbnails':
                        continue

                    yield {
                        'file_path': file_path,
                        'folder_name': folder_name if folder_name != os.path.basename(self.photos_dir) else 'Root',
                        'file_name': filename
                    }

    def get_folders(self) -> List[Dict]:
        """
        Get list of folders with photo counts.

        Returns:
            List of folder dictionaries with name and count
        """
        folders = {}
        for photo_info in self.discover_photos():
            folder = photo_info['folder_name']
            folders[folder] = folders.get(folder, 0) + 1

        return [{'name': name, 'count': count} for name, count in sorted(folders.items())]

    def index_photos(self, folder: Optional[str] = None) -> int:
        """
        Index photos into database (without face detection).

        Args:
            folder: Optional folder to limit indexing to

        Returns:
            Number of photos indexed
        """
        count = 0
        for photo_info in self.discover_photos():
            if folder and photo_info['folder_name'] != folder:
                continue

            # Check if already indexed
            existing = Photo.query.filter_by(file_path=photo_info['file_path']).first()
            if existing:
                continue

            # Get image dimensions
            try:
                width, height = self.face_detector.get_image_dimensions(photo_info['file_path'])
            except Exception:
                width, height = None, None

            photo = Photo(
                file_path=photo_info['file_path'],
                folder_name=photo_info['folder_name'],
                file_name=photo_info['file_name'],
                width=width,
                height=height
            )
            db.session.add(photo)
            count += 1

        db.session.commit()
        return count

    def process_photo(self, photo: Photo, compute_embeddings: bool = True) -> int:
        """
        Detect faces in a photo and optionally compute embeddings.

        Args:
            photo: Photo model instance
            compute_embeddings: Whether to compute embeddings

        Returns:
            Number of faces detected
        """
        # Detect faces
        try:
            face_locations = self.face_detector.detect_faces(photo.file_path)
        except Exception as e:
            logger.error(f"Error detecting faces in {photo.file_path}: {e}")
            return 0

        # Get embeddings if requested
        embeddings = []
        if compute_embeddings and face_locations:
            try:
                embeddings = self.embedding_service.extract_all_embeddings(
                    photo.file_path, face_locations
                )
            except Exception as e:
                logger.error(f"Error extracting embeddings from {photo.file_path}: {e}")

        # Store faces
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face = DetectedFace(
                photo_id=photo.id,
                bbox_top=top,
                bbox_right=right,
                bbox_bottom=bottom,
                bbox_left=left
            )

            if i < len(embeddings):
                face.embedding = embeddings[i]

            db.session.add(face)

        photo.face_count = len(face_locations)
        photo.processed_at = datetime.utcnow()
        db.session.commit()

        return len(face_locations)

    def scan_all_photos(self, classifier: Classifier,
                        folder: Optional[str] = None) -> Generator[Dict, None, None]:
        """
        Scan all photos with a classifier.

        Args:
            classifier: Trained classifier to use
            folder: Optional folder to limit scan to

        Yields:
            Progress updates with photo and match info
        """
        from app.models.classifier import load_classifier

        # Load the classifier model
        clf = load_classifier(classifier)

        # Get photos to scan
        query = Photo.query
        if folder:
            query = query.filter_by(folder_name=folder)

        photos = query.all()
        total = len(photos)

        for i, photo in enumerate(photos):
            # Process photo if not already done
            if photo.processed_at is None:
                self.process_photo(photo)

            # Check each face
            matches = []
            for face in photo.detected_faces:
                if face.embedding is None:
                    continue

                is_match, score = clf.predict(face.embedding)

                # Save result
                result = ScanResult(
                    classifier_id=classifier.id,
                    detected_face_id=face.id,
                    prediction_score=score,
                    is_match=is_match
                )
                db.session.add(result)

                if is_match:
                    matches.append({
                        'face_id': face.id,
                        'score': score,
                        'bbox': face.to_dict()['bbox']
                    })

            db.session.commit()

            yield {
                'progress': i + 1,
                'total': total,
                'photo': photo.to_dict(),
                'matches': matches,
                'has_match': len(matches) > 0
            }

    def get_scan_results(self, classifier_id: int,
                         folder: Optional[str] = None,
                         matches_only: bool = False) -> List[Dict]:
        """
        Get scan results grouped by photo.

        Args:
            classifier_id: ID of classifier used
            folder: Optional folder filter
            matches_only: Only return photos with matches

        Returns:
            List of photo results with match info
        """
        # Get all photos with their results
        query = db.session.query(Photo).join(
            DetectedFace
        ).join(
            ScanResult
        ).filter(
            ScanResult.classifier_id == classifier_id
        )

        if folder:
            query = query.filter(Photo.folder_name == folder)

        photos = query.distinct().all()

        results = []
        for photo in photos:
            photo_results = ScanResult.query.join(
                DetectedFace
            ).filter(
                DetectedFace.photo_id == photo.id,
                ScanResult.classifier_id == classifier_id
            ).all()

            matches = [r for r in photo_results if r.is_match]

            if matches_only and not matches:
                continue

            results.append({
                'photo': photo.to_dict(),
                'matches': [{
                    'result_id': r.id,
                    'face': r.detected_face.to_dict(),
                    'score': r.prediction_score,
                    'verified': r.user_verified
                } for r in matches],
                'total_faces': len(photo_results),
                'match_count': len(matches)
            })

        return results
