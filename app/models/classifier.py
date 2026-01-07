import numpy as np
import joblib
import os
from typing import List, Tuple, Dict, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.models.database import Classifier


class BaseClassifier:
    """Base class for face classifiers"""

    def __init__(self):
        self.is_trained = False
        self.threshold = 0.5

    def train(self, positive_embeddings: List[np.ndarray],
              negative_embeddings: List[np.ndarray]) -> Dict:
        raise NotImplementedError

    def predict(self, embedding: np.ndarray) -> Tuple[bool, float]:
        raise NotImplementedError

    def predict_batch(self, embeddings: List[np.ndarray]) -> List[Tuple[bool, float]]:
        return [self.predict(e) for e in embeddings]

    def evaluate(self, embeddings: List[np.ndarray],
                 labels: List[bool]) -> Dict:
        """
        Evaluate classifier on test data.

        Args:
            embeddings: List of test embeddings
            labels: List of ground truth labels (True = target person)

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        scores = []
        for emb in embeddings:
            pred, score = self.predict(emb)
            predictions.append(pred)
            scores.append(score)

        y_true = np.array(labels)
        y_pred = np.array(predictions)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'scores': scores
        }

    def set_threshold(self, value):
        """Update decision threshold"""
        self.threshold = value


class SVMClassifier(BaseClassifier):
    """SVM-based face classifier"""

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.classifier = SVC(
            kernel='linear',
            probability=True,
            C=1.0,
            class_weight='balanced'
        )

    def train(self, positive_embeddings: List[np.ndarray],
              negative_embeddings: List[np.ndarray]) -> Dict:
        """
        Train SVM on positive (target person) and negative (others) examples.

        Returns training metrics.
        """
        if not positive_embeddings:
            raise ValueError("Need at least one positive example")
        if not negative_embeddings:
            raise ValueError("Need at least one negative example")

        X = np.vstack([positive_embeddings, negative_embeddings])
        y = np.array([1] * len(positive_embeddings) + [0] * len(negative_embeddings))

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        self.classifier.fit(X_scaled, y)
        self.is_trained = True

        return {
            'num_positive': len(positive_embeddings),
            'num_negative': len(negative_embeddings),
            'algorithm': 'svm'
        }

    def predict(self, embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (is_match, probability)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained")

        embedding_scaled = self.scaler.transform([embedding])
        proba = self.classifier.predict_proba(embedding_scaled)[0]
        prob_positive = float(proba[1])

        return prob_positive >= self.threshold, prob_positive

    def save(self, path: str):
        """Save model to disk"""
        joblib.dump({
            'scaler': self.scaler,
            'classifier': self.classifier,
            'threshold': self.threshold
        }, path)

    @classmethod
    def load(cls, path: str) -> 'SVMClassifier':
        """Load model from disk"""
        data = joblib.load(path)
        instance = cls()
        instance.scaler = data['scaler']
        instance.classifier = data['classifier']
        instance.threshold = data.get('threshold', 0.5)
        instance.is_trained = True
        return instance


class CentroidClassifier(BaseClassifier):
    """Match faces using distance to centroid of training embeddings"""

    def __init__(self):
        super().__init__()
        self.centroid = None
        self.threshold = 0.6  # Distance threshold

    def train(self, positive_embeddings: List[np.ndarray],
              negative_embeddings: List[np.ndarray] = None) -> Dict:
        """
        Train by computing centroid of positive examples.
        Negative examples are used to tune threshold if provided.
        """
        if not positive_embeddings:
            raise ValueError("Need at least one positive example")

        self.centroid = np.mean(positive_embeddings, axis=0)
        self.is_trained = True

        # Calculate distances from training examples to centroid
        pos_distances = [np.linalg.norm(e - self.centroid) for e in positive_embeddings]
        max_pos_dist = max(pos_distances)
        avg_pos_dist = np.mean(pos_distances)

        # Set threshold to cover training examples + margin for variation
        # Use max distance + 50% margin to account for test variation
        self.threshold = max_pos_dist * 1.5

        # Tune threshold if negatives provided
        if negative_embeddings:
            self._tune_threshold(positive_embeddings, negative_embeddings)

        return {
            'num_positive': len(positive_embeddings),
            'num_negative': len(negative_embeddings) if negative_embeddings else 0,
            'algorithm': 'centroid',
            'threshold': self.threshold
        }

    def _tune_threshold(self, positive_embeddings: List[np.ndarray],
                        negative_embeddings: List[np.ndarray]):
        """Find optimal threshold using validation data"""
        pos_distances = [np.linalg.norm(e - self.centroid) for e in positive_embeddings]
        neg_distances = [np.linalg.norm(e - self.centroid) for e in negative_embeddings]

        # Find threshold that maximizes F1
        best_f1 = 0
        max_dist = max(max(pos_distances), max(neg_distances)) if neg_distances else max(pos_distances)
        best_threshold = max_dist * 1.2

        for threshold in np.arange(0.3, max_dist * 1.5, 0.05):
            pos_correct = sum(1 for d in pos_distances if d < threshold)
            neg_correct = sum(1 for d in neg_distances if d >= threshold)

            tp = pos_correct
            fp = len(neg_distances) - neg_correct
            fn = len(pos_distances) - pos_correct

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = best_threshold

    def predict(self, embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (is_match, distance)
        Lower distance = more likely to be a match
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained")

        distance = float(np.linalg.norm(embedding - self.centroid))
        # Convert distance to a score (1 = closest, 0 = furthest)
        score = max(0, 1 - distance / 1.5)

        return distance < self.threshold, score

    def save(self, path: str):
        """Save model to disk"""
        joblib.dump({
            'centroid': self.centroid,
            'threshold': self.threshold
        }, path)

    @classmethod
    def load(cls, path: str) -> 'CentroidClassifier':
        """Load model from disk"""
        data = joblib.load(path)
        instance = cls()
        instance.centroid = data['centroid']
        instance.threshold = data['threshold']
        instance.is_trained = True
        return instance


class KNNClassifier(BaseClassifier):
    """K-Nearest Neighbors face classifier"""

    def __init__(self, k: int = 3):
        super().__init__()
        self.embeddings = None
        self.k = k
        self.threshold = 0.6  # Distance threshold

    def train(self, positive_embeddings: List[np.ndarray],
              negative_embeddings: List[np.ndarray] = None) -> Dict:
        """
        Train by storing positive examples.
        """
        if not positive_embeddings:
            raise ValueError("Need at least one positive example")

        self.embeddings = np.array(positive_embeddings)
        self.k = min(self.k, len(positive_embeddings))
        self.is_trained = True

        return {
            'num_positive': len(positive_embeddings),
            'num_negative': 0,
            'algorithm': 'knn',
            'k': self.k
        }

    def predict(self, embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (is_match, average_distance_to_k_nearest)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained")

        distances = np.linalg.norm(self.embeddings - embedding, axis=1)
        k_nearest_distances = np.sort(distances)[:self.k]
        avg_distance = float(np.mean(k_nearest_distances))

        # Convert to score
        score = max(0, 1 - avg_distance / 1.5)

        return avg_distance < self.threshold, score

    def save(self, path: str):
        """Save model to disk"""
        joblib.dump({
            'embeddings': self.embeddings,
            'k': self.k,
            'threshold': self.threshold
        }, path)

    @classmethod
    def load(cls, path: str) -> 'KNNClassifier':
        """Load model from disk"""
        data = joblib.load(path)
        instance = cls(k=data['k'])
        instance.embeddings = data['embeddings']
        instance.threshold = data['threshold']
        instance.is_trained = True
        return instance


def create_classifier(algorithm: str) -> BaseClassifier:
    """Factory function to create classifiers"""
    if algorithm == 'svm':
        return SVMClassifier()
    elif algorithm == 'centroid':
        return CentroidClassifier()
    elif algorithm == 'knn':
        return KNNClassifier()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def load_classifier(classifier_model: Classifier) -> BaseClassifier:
    """Load a classifier from database model"""
    if not classifier_model.model_path or not os.path.exists(classifier_model.model_path):
        raise ValueError("Classifier model file not found")

    if classifier_model.algorithm == 'svm':
        return SVMClassifier.load(classifier_model.model_path)
    elif classifier_model.algorithm == 'centroid':
        return CentroidClassifier.load(classifier_model.model_path)
    elif classifier_model.algorithm == 'knn':
        return KNNClassifier.load(classifier_model.model_path)
    else:
        raise ValueError(f"Unknown algorithm: {classifier_model.algorithm}")


def train_and_save_classifier(person_id: int, algorithm: str,
                               positive_embeddings: List[np.ndarray],
                               negative_embeddings: List[np.ndarray],
                               models_dir: str) -> Tuple[BaseClassifier, str]:
    """
    Train a classifier and save it to disk.

    Returns:
        Tuple of (classifier instance, model path)
    """
    clf = create_classifier(algorithm)
    metrics = clf.train(positive_embeddings, negative_embeddings)

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"classifier_person{person_id}_{algorithm}.joblib")
    clf.save(model_path)

    return clf, model_path
