from datetime import datetime
from app import db
import numpy as np
import struct


def bytes_to_int(value):
    """Convert a value to int, handling bytes from DB"""
    if value is None:
        return 0
    if isinstance(value, bytes):
        # Unpack as int64 little-endian
        if len(value) == 8:
            return struct.unpack('<q', value)[0]
        elif len(value) == 4:
            return struct.unpack('<i', value)[0]
        else:
            return int.from_bytes(value, 'little')
    return int(value)


class Person(db.Model):
    """Target person for face identification"""
    __tablename__ = 'persons'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    # Relationships
    face_examples = db.relationship('FaceExample', backref='person', lazy='dynamic')
    classifiers = db.relationship('Classifier', backref='person', lazy='dynamic')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_active': self.is_active,
            'example_count': self.face_examples.filter_by(is_test_set=False).count(),
            'test_count': self.face_examples.filter_by(is_test_set=True).count()
        }


class Photo(db.Model):
    """Photo file metadata"""
    __tablename__ = 'photos'

    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.String(500), unique=True, nullable=False)
    folder_name = db.Column(db.String(100), nullable=False)
    file_name = db.Column(db.String(200), nullable=False)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    processed_at = db.Column(db.DateTime)
    face_count = db.Column(db.Integer, default=0)

    # Relationships
    detected_faces = db.relationship('DetectedFace', backref='photo', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self, include_faces=False):
        result = {
            'id': self.id,
            'file_path': self.file_path,
            'folder_name': self.folder_name,
            'file_name': self.file_name,
            'width': self.width,
            'height': self.height,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'face_count': self.face_count
        }
        if include_faces:
            result['faces'] = [f.to_dict() for f in self.detected_faces]
        return result


class DetectedFace(db.Model):
    """A face detected in a photo"""
    __tablename__ = 'detected_faces'

    id = db.Column(db.Integer, primary_key=True)
    photo_id = db.Column(db.Integer, db.ForeignKey('photos.id'), nullable=False)

    # Bounding box (original detection) - format: top, right, bottom, left
    bbox_top = db.Column(db.Integer, nullable=False)
    bbox_right = db.Column(db.Integer, nullable=False)
    bbox_bottom = db.Column(db.Integer, nullable=False)
    bbox_left = db.Column(db.Integer, nullable=False)

    # User-adjusted bounding box (nullable - only if adjusted)
    adjusted_top = db.Column(db.Integer)
    adjusted_right = db.Column(db.Integer)
    adjusted_bottom = db.Column(db.Integer)
    adjusted_left = db.Column(db.Integer)

    # 128-dimensional embedding stored as binary blob
    embedding_blob = db.Column(db.LargeBinary)

    # Metadata
    detection_confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def embedding(self):
        """Deserialize embedding from blob"""
        if self.embedding_blob:
            return np.frombuffer(self.embedding_blob, dtype=np.float64)
        return None

    @embedding.setter
    def embedding(self, value):
        """Serialize embedding to blob"""
        if value is not None:
            self.embedding_blob = np.array(value, dtype=np.float64).tobytes()
        else:
            self.embedding_blob = None

    @property
    def effective_bbox(self):
        """Return adjusted bbox if available, otherwise original"""
        if self.adjusted_top is not None:
            return (bytes_to_int(self.adjusted_top), bytes_to_int(self.adjusted_right),
                    bytes_to_int(self.adjusted_bottom), bytes_to_int(self.adjusted_left))
        return (bytes_to_int(self.bbox_top), bytes_to_int(self.bbox_right),
                bytes_to_int(self.bbox_bottom), bytes_to_int(self.bbox_left))

    def to_dict(self):
        top, right, bottom, left = self.effective_bbox
        # Ensure values are integers (handle bytes from DB)
        top = int(top) if top is not None else 0
        right = int(right) if right is not None else 0
        bottom = int(bottom) if bottom is not None else 0
        left = int(left) if left is not None else 0
        return {
            'id': self.id,
            'photo_id': self.photo_id,
            'bbox': {
                'top': top,
                'right': right,
                'bottom': bottom,
                'left': left,
                'width': right - left,
                'height': bottom - top
            },
            'original_bbox': {
                'top': bytes_to_int(self.bbox_top),
                'right': bytes_to_int(self.bbox_right),
                'bottom': bytes_to_int(self.bbox_bottom),
                'left': bytes_to_int(self.bbox_left)
            },
            'is_adjusted': self.adjusted_top is not None,
            'has_embedding': self.embedding_blob is not None,
            'detection_confidence': self.detection_confidence
        }


class FaceExample(db.Model):
    """A face example selected by user for training"""
    __tablename__ = 'face_examples'

    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('persons.id'), nullable=False)
    detected_face_id = db.Column(db.Integer, db.ForeignKey('detected_faces.id'), nullable=False)

    # Example metadata
    is_test_set = db.Column(db.Boolean, default=False)
    is_positive = db.Column(db.Boolean, default=True)  # True = target person, False = negative example
    quality_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship
    detected_face = db.relationship('DetectedFace')

    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'detected_face_id': self.detected_face_id,
            'is_test_set': self.is_test_set,
            'is_positive': self.is_positive,
            'quality_score': self.quality_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'face': self.detected_face.to_dict() if self.detected_face else None
        }


class Classifier(db.Model):
    """Trained classifier model"""
    __tablename__ = 'classifiers'

    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('persons.id'), nullable=False)

    # Model info
    algorithm = db.Column(db.String(50), nullable=False)  # 'svm', 'centroid', 'knn'
    model_path = db.Column(db.String(500))

    # Training metrics
    num_training_examples = db.Column(db.Integer)
    num_negative_examples = db.Column(db.Integer)
    embedding_variance = db.Column(db.Float)

    # Hyperparameters (JSON string)
    hyperparameters = db.Column(db.Text)

    # Performance metrics (from Phase 2)
    test_accuracy = db.Column(db.Float)
    test_precision = db.Column(db.Float)
    test_recall = db.Column(db.Float)
    test_f1 = db.Column(db.Float)
    optimal_threshold = db.Column(db.Float, default=0.5)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'algorithm': self.algorithm,
            'num_training_examples': self.num_training_examples,
            'num_negative_examples': self.num_negative_examples,
            'test_accuracy': self.test_accuracy,
            'test_precision': self.test_precision,
            'test_recall': self.test_recall,
            'test_f1': self.test_f1,
            'optimal_threshold': self.optimal_threshold,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ScanResult(db.Model):
    """Results from Phase 3 full photo scan"""
    __tablename__ = 'scan_results'

    id = db.Column(db.Integer, primary_key=True)
    classifier_id = db.Column(db.Integer, db.ForeignKey('classifiers.id'), nullable=False)
    detected_face_id = db.Column(db.Integer, db.ForeignKey('detected_faces.id'), nullable=False)

    # Prediction
    prediction_score = db.Column(db.Float, nullable=False)
    is_match = db.Column(db.Boolean, nullable=False)

    # User feedback
    user_verified = db.Column(db.Boolean)  # True=correct, False=incorrect, None=not reviewed
    verified_at = db.Column(db.DateTime)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    classifier = db.relationship('Classifier')
    detected_face = db.relationship('DetectedFace')

    def to_dict(self):
        return {
            'id': self.id,
            'classifier_id': self.classifier_id,
            'detected_face_id': self.detected_face_id,
            'prediction_score': self.prediction_score,
            'is_match': self.is_match,
            'user_verified': self.user_verified,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None,
            'face': self.detected_face.to_dict() if self.detected_face else None
        }
