import numpy as np
from typing import List, Dict
from sklearn.decomposition import PCA


class DiversityAnalyzer:
    """Analyze diversity of collected face embeddings to guide example collection"""

    # Thresholds
    MIN_EXAMPLES = 5
    OPTIMAL_EXAMPLES = 10
    MAX_USEFUL_EXAMPLES = 20

    # Variance thresholds
    MIN_VARIANCE = 0.03
    OPTIMAL_VARIANCE = 0.08

    def __init__(self, embeddings: List[np.ndarray] = None):
        """
        Initialize analyzer with embeddings.

        Args:
            embeddings: List of 128D face embeddings
        """
        self.embeddings = np.array(embeddings) if embeddings else np.array([])
        self.n_samples = len(self.embeddings) if embeddings else 0

    def set_embeddings(self, embeddings: List[np.ndarray]):
        """Update embeddings for analysis"""
        self.embeddings = np.array(embeddings) if embeddings else np.array([])
        self.n_samples = len(self.embeddings)

    def analyze(self) -> Dict:
        """
        Comprehensive diversity analysis.

        Returns:
            Dictionary with:
            - count: number of examples
            - overall_variance: average variance across dimensions
            - spread_score: normalized measure of embedding spread
            - coverage_estimate: estimated coverage of face variations
            - recommendations: list of suggestions
            - ready_for_training: boolean
            - quality_score: 0-100 score
        """
        if self.n_samples == 0:
            return {
                'count': 0,
                'overall_variance': 0.0,
                'spread_score': 0.0,
                'coverage_estimate': {'estimated_coverage': 0.0},
                'recommendations': ['Start by adding face examples of the target person'],
                'ready_for_training': False,
                'quality_score': 0.0
            }

        if self.n_samples == 1:
            return {
                'count': 1,
                'overall_variance': 0.0,
                'spread_score': 0.0,
                'coverage_estimate': {'estimated_coverage': 0.0},
                'recommendations': [f'Add {self.MIN_EXAMPLES - 1} more examples (minimum {self.MIN_EXAMPLES})'],
                'ready_for_training': False,
                'quality_score': 10.0
            }

        # Basic statistics
        variance_per_dim = np.var(self.embeddings, axis=0)
        overall_variance = float(np.mean(variance_per_dim))

        # Spread: average pairwise distance
        spread_score = self._compute_spread()

        # Coverage estimate using PCA
        coverage = self._estimate_coverage()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_variance, spread_score, coverage
        )

        # Readiness check - more lenient for small datasets
        ready = (
            self.n_samples >= self.MIN_EXAMPLES and
            (overall_variance >= self.MIN_VARIANCE or self.n_samples >= self.OPTIMAL_EXAMPLES)
        )

        quality_score = self._compute_quality_score(overall_variance, spread_score)

        return {
            'count': self.n_samples,
            'overall_variance': overall_variance,
            'spread_score': spread_score,
            'coverage_estimate': coverage,
            'recommendations': recommendations,
            'ready_for_training': ready,
            'quality_score': quality_score,
            'progress': {
                'current': self.n_samples,
                'minimum': self.MIN_EXAMPLES,
                'optimal': self.OPTIMAL_EXAMPLES,
                'percentage': min(100, int(self.n_samples / self.OPTIMAL_EXAMPLES * 100))
            }
        }

    def _compute_spread(self) -> float:
        """Compute normalized spread (average pairwise distance)"""
        if self.n_samples < 2:
            return 0.0

        # Compute pairwise distances
        total_distance = 0.0
        count = 0
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                total_distance += np.linalg.norm(
                    self.embeddings[i] - self.embeddings[j]
                )
                count += 1

        avg_distance = total_distance / count
        # Normalize: dlib threshold is 0.6, max reasonable distance ~1.2
        return min(float(avg_distance / 1.0), 1.0)

    def _estimate_coverage(self) -> Dict:
        """
        Estimate coverage of face variations using PCA.
        High variance in top principal components = good coverage.
        """
        if self.n_samples < 3:
            return {'estimated_coverage': 0.0, 'top_pc_variance': []}

        try:
            n_components = min(10, self.n_samples - 1)
            pca = PCA(n_components=n_components)
            pca.fit(self.embeddings)

            return {
                'estimated_coverage': float(sum(pca.explained_variance_ratio_[:3])),
                'top_pc_variance': [float(v) for v in pca.explained_variance_ratio_[:5]]
            }
        except Exception:
            return {'estimated_coverage': 0.0, 'top_pc_variance': []}

    def _generate_recommendations(self, variance: float,
                                   spread: float, coverage: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recs = []

        if self.n_samples < self.MIN_EXAMPLES:
            remaining = self.MIN_EXAMPLES - self.n_samples
            recs.append(f"Add {remaining} more example{'s' if remaining > 1 else ''} (minimum {self.MIN_EXAMPLES})")
        elif self.n_samples < self.OPTIMAL_EXAMPLES:
            remaining = self.OPTIMAL_EXAMPLES - self.n_samples
            recs.append(f"Consider adding {remaining} more example{'s' if remaining > 1 else ''} for better accuracy")

        if self.n_samples >= 2:
            if variance < self.MIN_VARIANCE:
                recs.append("Try adding photos with different angles or lighting")

            if spread < 0.25:
                recs.append("Examples look similar - try photos from different events/days")

        if self.n_samples >= self.MIN_EXAMPLES:
            if variance >= self.MIN_VARIANCE or self.n_samples >= self.OPTIMAL_EXAMPLES:
                if self.n_samples >= self.OPTIMAL_EXAMPLES:
                    recs.append("Excellent diversity! Ready for training.")
                else:
                    recs.append("Good progress! You can proceed to training or add more examples.")

        if not recs:
            recs.append("Keep adding diverse examples of the target person")

        return recs

    def _compute_quality_score(self, variance: float, spread: float) -> float:
        """
        Compute overall quality score 0-100.
        """
        # Count component (0-50 points)
        count_score = min(self.n_samples / self.OPTIMAL_EXAMPLES, 1.0) * 50

        # Variance component (0-25 points)
        variance_score = min(variance / self.OPTIMAL_VARIANCE, 1.0) * 25

        # Spread component (0-25 points)
        spread_score = min(spread / 0.4, 1.0) * 25

        return round(count_score + variance_score + spread_score, 1)

    def get_most_different_from(self, new_embedding: np.ndarray) -> Dict:
        """
        Analyze how different a new embedding is from existing ones.

        Args:
            new_embedding: The embedding to analyze

        Returns:
            Dictionary with analysis of the new embedding
        """
        if self.n_samples == 0:
            return {
                'min_distance': None,
                'avg_distance': None,
                'adds_diversity': True,
                'message': 'First example - good start!'
            }

        distances = [np.linalg.norm(new_embedding - e) for e in self.embeddings]
        min_dist = float(min(distances))
        avg_dist = float(np.mean(distances))

        # Check if it adds diversity (not too similar to existing)
        adds_diversity = min_dist > 0.3  # At least somewhat different

        if min_dist < 0.2:
            message = "Very similar to an existing example - consider a different photo"
        elif min_dist < 0.3:
            message = "Somewhat similar to existing examples"
        else:
            message = "Good diversity - this adds variety to your examples"

        return {
            'min_distance': min_dist,
            'avg_distance': avg_dist,
            'adds_diversity': adds_diversity,
            'message': message
        }
