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


class TemplateMatcherClassifier(BaseClassifier):
    """
    Template-based face matcher using cosine similarity.

    Based on iOS on-device face identification spec:
    - Stores individual templates (up to 10)
    - Uses cosine similarity with L2-normalized embeddings
    - Multi-threshold matching: s_max, s_mu, s_top2_mean
    - Adaptive threshold calibration
    - Precision > recall (conservative matching)
    """

    def __init__(self):
        super().__init__()
        self.templates = None  # Individual template embeddings
        self.mean_template = None  # Mean embedding (μ)
        self.t_high = 0.7  # High confidence threshold for s_max
        self.t_mu = 0.6  # Threshold for mean similarity
        self.t_top2 = 0.55  # Threshold for top-2 mean
        self.template_std = 0.0  # Std dev of inter-template similarities

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _cosine_similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        e1_norm = self._normalize(e1)
        e2_norm = self._normalize(e2)
        return float(np.dot(e1_norm, e2_norm))

    def train(self, positive_embeddings: List[np.ndarray],
              negative_embeddings: List[np.ndarray] = None) -> Dict:
        """
        Train by storing templates and calibrating thresholds.

        During enrollment:
        - Store up to 10 templates
        - Compute mean template
        - Calibrate thresholds based on template similarity distribution
        """
        if not positive_embeddings:
            raise ValueError("Need at least one positive example")

        # Limit to 10 templates as per spec
        templates = positive_embeddings[:10]

        # L2 normalize all templates
        self.templates = np.array([self._normalize(e) for e in templates])

        # Compute mean template
        self.mean_template = self._normalize(np.mean(self.templates, axis=0))

        # Calibrate thresholds based on inter-template similarities
        self._calibrate_thresholds()

        self.is_trained = True

        return {
            'num_positive': len(templates),
            'num_negative': len(negative_embeddings) if negative_embeddings else 0,
            'algorithm': 'template',
            't_high': self.t_high,
            't_mu': self.t_mu,
            't_top2': self.t_top2
        }

    def _calibrate_thresholds(self):
        """
        Calibrate thresholds based on template similarity distribution.

        Set T_MU = mean - k*std (clamped) for conservative matching.
        """
        if len(self.templates) < 2:
            # Can't calibrate with single template, use defaults
            return

        # Compute pairwise similarities between templates
        similarities = []
        for i in range(len(self.templates)):
            for j in range(i + 1, len(self.templates)):
                sim = self._cosine_similarity(self.templates[i], self.templates[j])
                similarities.append(sim)

        # Also compute similarity of each template to mean
        mean_similarities = []
        for t in self.templates:
            sim = self._cosine_similarity(t, self.mean_template)
            mean_similarities.append(sim)

        # Statistics
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        self.template_std = std_sim

        min_mean_sim = min(mean_similarities)

        # Calibrate thresholds (conservative - precision > recall)
        # T_HIGH: should be achievable by good matches
        self.t_high = max(0.6, min(0.85, mean_sim - 0.5 * std_sim))

        # T_MU: based on worst template-to-mean similarity with margin
        self.t_mu = max(0.5, min(0.75, min_mean_sim - 1.5 * std_sim))

        # T_TOP2: slightly lower than t_mu
        self.t_top2 = max(0.45, self.t_mu - 0.1)

    def predict(self, embedding: np.ndarray,
                quality_score: Optional[float] = None) -> Tuple[bool, float]:
        """
        Match using multi-threshold logic.

        Compute:
        - s_max = max similarity to any template
        - s_mu = similarity to mean template
        - s_top2_mean = mean of top 2 template similarities

        Accept if:
        (s_max >= T_HIGH) OR (s_mu >= T_MU AND s_top2_mean >= T_TOP2)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained")

        # Normalize candidate embedding
        e_norm = self._normalize(embedding)

        # Compute similarity to all templates
        similarities = [float(np.dot(e_norm, t)) for t in self.templates]

        # s_max: best match to any template
        s_max = max(similarities)

        # s_mu: similarity to mean template
        s_mu = float(np.dot(e_norm, self.mean_template))

        # s_top2_mean: mean of top 2 similarities
        sorted_sims = sorted(similarities, reverse=True)
        s_top2_mean = np.mean(sorted_sims[:min(2, len(sorted_sims))])

        # Multi-threshold matching logic (precision > recall)
        is_match = (
            (s_max >= self.t_high) or
            (s_mu >= self.t_mu and s_top2_mean >= self.t_top2)
        )

        # Return s_max as the primary score
        return is_match, s_max

    def predict_detailed(self, embedding: np.ndarray) -> Dict:
        """
        Return detailed matching information for debugging/display.
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained")

        e_norm = self._normalize(embedding)
        similarities = [float(np.dot(e_norm, t)) for t in self.templates]

        s_max = max(similarities)
        s_mu = float(np.dot(e_norm, self.mean_template))
        sorted_sims = sorted(similarities, reverse=True)
        s_top2_mean = np.mean(sorted_sims[:min(2, len(sorted_sims))])

        is_match = (
            (s_max >= self.t_high) or
            (s_mu >= self.t_mu and s_top2_mean >= self.t_top2)
        )

        return {
            'is_match': is_match,
            's_max': s_max,
            's_mu': s_mu,
            's_top2_mean': s_top2_mean,
            't_high': self.t_high,
            't_mu': self.t_mu,
            't_top2': self.t_top2,
            'match_reason': 'high_confidence' if s_max >= self.t_high else
                           ('multi_threshold' if is_match else 'no_match'),
            'all_similarities': similarities
        }

    def save(self, path: str):
        """Save model to disk"""
        joblib.dump({
            'templates': self.templates,
            'mean_template': self.mean_template,
            't_high': self.t_high,
            't_mu': self.t_mu,
            't_top2': self.t_top2,
            'template_std': self.template_std
        }, path)

    @classmethod
    def load(cls, path: str) -> 'TemplateMatcherClassifier':
        """Load model from disk"""
        data = joblib.load(path)
        instance = cls()
        instance.templates = data['templates']
        instance.mean_template = data['mean_template']
        instance.t_high = data['t_high']
        instance.t_mu = data['t_mu']
        instance.t_top2 = data['t_top2']
        instance.template_std = data.get('template_std', 0.0)
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
    elif algorithm == 'template':
        return TemplateMatcherClassifier()
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
    elif classifier_model.algorithm == 'template':
        return TemplateMatcherClassifier.load(classifier_model.model_path)
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
