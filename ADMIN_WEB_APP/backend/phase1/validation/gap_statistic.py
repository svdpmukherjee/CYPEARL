"""
Gap Statistic for Optimal K Selection

The Gap Statistic compares within-cluster dispersion to a null reference
distribution (uniform random data). This provides a principled, data-driven
method for selecting the optimal number of clusters.

Reference:
    Tibshirani, R., Walther, G., & Hastie, T. (2001).
    Estimating the number of clusters in a data set via the gap statistic.
    Journal of the Royal Statistical Society: Series B, 63(2), 411-423.

Scientific Rationale:
    - Compares log(W_k) to E*[log(W_k)] under null hypothesis of no clusters
    - Gap(k) = E*[log(W_k)] - log(W_k)
    - Optimal K is smallest K such that Gap(K) >= Gap(K+1) - s(K+1)
    - Where s(K+1) is the standard error of the gap
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GapStatisticResult:
    """Results from Gap Statistic analysis."""
    k: int
    gap: float
    gap_std: float
    log_wk: float
    log_wk_ref_mean: float
    log_wk_ref_std: float

    def to_dict(self) -> Dict:
        return {
            'k': self.k,
            'gap': self.gap,
            'gap_std': self.gap_std,
            'log_wk': self.log_wk,
            'log_wk_ref_mean': self.log_wk_ref_mean,
            'log_wk_ref_std': self.log_wk_ref_std
        }


class GapStatisticAnalyzer:
    """
    Compute Gap Statistic for determining optimal number of clusters.

    The Gap Statistic provides a principled method for choosing K that
    doesn't rely on arbitrary thresholds or heuristics like the "elbow method".

    Key Features:
    - Compares actual clustering to null reference distribution
    - Provides statistical justification for K selection
    - Includes standard error for uncertainty quantification
    """

    def __init__(
        self,
        n_references: int = 20,
        random_state: int = 42
    ):
        """
        Initialize Gap Statistic analyzer.

        Args:
            n_references: Number of reference datasets to generate (default: 20)
                         Higher = more stable but slower
            random_state: Random seed for reproducibility
        """
        self.n_references = n_references
        self.random_state = random_state

    def analyze(
        self,
        X: np.ndarray,
        k_min: int = 2,
        k_max: int = 15,
        algorithm: str = 'kmeans'
    ) -> Dict:
        """
        Compute Gap Statistic for range of K values.

        Args:
            X: Feature matrix (n_samples, n_features)
            k_min: Minimum K to test
            k_max: Maximum K to test
            algorithm: Clustering algorithm ('kmeans' supported)

        Returns:
            Dictionary with gap statistics and optimal K recommendation
        """
        results = []

        # Compute gap for each K
        for k in range(k_min, k_max + 1):
            gap_result = self._compute_gap(X, k)
            results.append(gap_result)

        # Find optimal K using gap criterion
        optimal_k, selection_details = self._find_optimal_k(results)

        # Calculate confidence in the selection
        confidence = self._calculate_confidence(results, optimal_k)

        return {
            'optimal_k': optimal_k,
            'confidence': confidence,
            'selection_method': 'gap_statistic',
            'selection_criterion': 'Gap(k) >= Gap(k+1) - s(k+1)',
            'selection_details': selection_details,
            'results_by_k': [r.to_dict() for r in results],
            'recommendation': self._generate_recommendation(optimal_k, confidence, results)
        }

    def _compute_gap(self, X: np.ndarray, k: int) -> GapStatisticResult:
        """Compute Gap Statistic for single K value."""

        # Compute W_k for actual data
        log_wk = self._compute_log_wk(X, k)

        # Compute W_k for reference datasets
        log_wk_refs = []
        np.random.seed(self.random_state)

        for b in range(self.n_references):
            X_ref = self._generate_reference_data(X, seed=self.random_state + b)
            log_wk_ref = self._compute_log_wk(X_ref, k)
            log_wk_refs.append(log_wk_ref)

        log_wk_refs = np.array(log_wk_refs)

        # Gap = E*[log(W_k)] - log(W_k)
        gap = np.mean(log_wk_refs) - log_wk

        # Standard deviation (adjusted for Monte Carlo error)
        # s_k = sd_k * sqrt(1 + 1/B) where B is number of references
        sd_k = np.std(log_wk_refs)
        s_k = sd_k * np.sqrt(1 + 1 / self.n_references)

        return GapStatisticResult(
            k=k,
            gap=float(gap),
            gap_std=float(s_k),
            log_wk=float(log_wk),
            log_wk_ref_mean=float(np.mean(log_wk_refs)),
            log_wk_ref_std=float(sd_k)
        )

    def _compute_log_wk(self, X: np.ndarray, k: int) -> float:
        """
        Compute log(W_k) - the within-cluster dispersion.

        W_k = sum over clusters r of (1/(2*n_r)) * sum of pairwise distances in cluster r
        """
        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        # Compute within-cluster dispersion
        wk = 0.0
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_points = X[cluster_mask]

            if len(cluster_points) > 1:
                # Sum of pairwise squared distances divided by 2*n_r
                # Using Euclidean distance
                distances = pairwise_distances(cluster_points, metric='euclidean')
                n_r = len(cluster_points)
                cluster_dispersion = np.sum(distances ** 2) / (2 * n_r)
                wk += cluster_dispersion
            # If only 1 point, dispersion is 0

        # Return log to avoid numerical issues with small W_k
        return np.log(wk + 1e-10)

    def _generate_reference_data(self, X: np.ndarray, seed: int) -> np.ndarray:
        """
        Generate reference dataset under null hypothesis.

        Two options:
        1. Uniform over bounding box (simple, used here)
        2. Uniform over principal components (more accurate for non-spherical data)

        We use option 1 for simplicity and robustness.
        """
        np.random.seed(seed)
        n_samples, n_features = X.shape

        # Generate uniform random data within the bounding box of X
        X_ref = np.zeros((n_samples, n_features))

        for j in range(n_features):
            min_val = X[:, j].min()
            max_val = X[:, j].max()
            X_ref[:, j] = np.random.uniform(min_val, max_val, n_samples)

        return X_ref

    def _find_optimal_k(
        self,
        results: List[GapStatisticResult]
    ) -> Tuple[int, Dict]:
        """
        Find optimal K using the gap criterion.

        Standard criterion (Tibshirani et al.):
        Smallest K such that Gap(K) >= Gap(K+1) - s(K+1)

        Alternative: Maximum gap method (simpler but less principled)
        """
        details = {
            'method': 'standard_gap_criterion',
            'comparisons': []
        }

        # Standard criterion: Gap(k) >= Gap(k+1) - s(k+1)
        for i in range(len(results) - 1):
            current = results[i]
            next_result = results[i + 1]

            threshold = next_result.gap - next_result.gap_std
            meets_criterion = current.gap >= threshold

            details['comparisons'].append({
                'k': current.k,
                'gap_k': current.gap,
                'gap_k_plus_1_minus_s': threshold,
                'meets_criterion': meets_criterion
            })

            if meets_criterion:
                details['selected_k'] = current.k
                details['reason'] = f'Gap({current.k})={current.gap:.4f} >= Gap({next_result.k})-s={threshold:.4f}'
                return current.k, details

        # If no K meets criterion, return K with maximum gap
        max_gap_result = max(results, key=lambda r: r.gap)
        details['fallback'] = True
        details['reason'] = f'No K met standard criterion. Using maximum gap at K={max_gap_result.k}'
        return max_gap_result.k, details

    def _calculate_confidence(
        self,
        results: List[GapStatisticResult],
        optimal_k: int
    ) -> Dict:
        """
        Calculate confidence in the K selection.

        Confidence is based on:
        1. Margin over the criterion threshold
        2. Stability of gap across neighboring K values
        3. Distinctiveness of the gap peak
        """
        # Find the result for optimal K
        optimal_result = next(r for r in results if r.k == optimal_k)
        optimal_idx = next(i for i, r in enumerate(results) if r.k == optimal_k)

        # 1. Margin over criterion
        if optimal_idx < len(results) - 1:
            next_result = results[optimal_idx + 1]
            threshold = next_result.gap - next_result.gap_std
            margin = optimal_result.gap - threshold
            margin_confidence = min(margin / 0.1, 1.0)  # Normalize by 0.1 target margin
        else:
            margin_confidence = 0.5  # Default if at max K

        # 2. Gap magnitude relative to standard error
        signal_to_noise = optimal_result.gap / (optimal_result.gap_std + 1e-10)
        snr_confidence = min(signal_to_noise / 2.0, 1.0)  # Target SNR of 2

        # 3. Distinctiveness: gap at optimal vs. mean of neighbors
        gaps = [r.gap for r in results]
        if optimal_idx > 0 and optimal_idx < len(results) - 1:
            neighbor_mean = (gaps[optimal_idx - 1] + gaps[optimal_idx + 1]) / 2
            distinctiveness = (gaps[optimal_idx] - neighbor_mean) / (np.std(gaps) + 1e-10)
            distinct_confidence = min(max(distinctiveness, 0) / 0.5, 1.0)
        else:
            distinct_confidence = 0.5

        # Combined confidence
        overall = (0.4 * margin_confidence + 0.3 * snr_confidence + 0.3 * distinct_confidence)

        # Confidence level interpretation
        if overall >= 0.8:
            level = 'high'
        elif overall >= 0.5:
            level = 'moderate'
        else:
            level = 'low'

        return {
            'overall': float(overall),
            'level': level,
            'components': {
                'margin_over_threshold': float(margin_confidence),
                'signal_to_noise': float(snr_confidence),
                'distinctiveness': float(distinct_confidence)
            }
        }

    def _generate_recommendation(
        self,
        optimal_k: int,
        confidence: Dict,
        results: List[GapStatisticResult]
    ) -> str:
        """Generate human-readable recommendation."""

        level = confidence['level']
        overall = confidence['overall']

        if level == 'high':
            return (
                f"Strong recommendation: K={optimal_k} is supported by gap statistic analysis "
                f"(confidence: {overall:.2f}). The gap criterion is clearly satisfied."
            )
        elif level == 'moderate':
            # Find runner-up K values
            sorted_by_gap = sorted(results, key=lambda r: r.gap, reverse=True)
            alternatives = [r.k for r in sorted_by_gap[:3] if r.k != optimal_k][:2]
            alt_str = ', '.join(map(str, alternatives)) if alternatives else 'none'
            return (
                f"Moderate recommendation: K={optimal_k} is suggested but with moderate confidence "
                f"({overall:.2f}). Consider also testing K={alt_str} and comparing behavioral validation."
            )
        else:
            return (
                f"Weak recommendation: K={optimal_k} is suggested but with low confidence ({overall:.2f}). "
                f"The gap statistic does not show a clear optimal K. "
                f"Rely more heavily on behavioral validation (eta-squared) for final selection."
            )

    def compare_to_elbow(
        self,
        X: np.ndarray,
        k_min: int = 2,
        k_max: int = 15
    ) -> Dict:
        """
        Compare Gap Statistic recommendation to traditional elbow method.

        This helps validate the gap statistic and identify cases where they disagree.
        """
        # Compute inertias for elbow method
        inertias = []
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append({
                'k': k,
                'inertia': float(kmeans.inertia_)
            })

        # Simple elbow detection: largest second derivative
        if len(inertias) >= 3:
            second_derivs = []
            for i in range(1, len(inertias) - 1):
                prev_inertia = inertias[i-1]['inertia']
                curr_inertia = inertias[i]['inertia']
                next_inertia = inertias[i+1]['inertia']

                # Second derivative (curvature)
                d2 = prev_inertia - 2 * curr_inertia + next_inertia
                second_derivs.append({
                    'k': inertias[i]['k'],
                    'curvature': d2
                })

            elbow_k = max(second_derivs, key=lambda x: x['curvature'])['k']
        else:
            elbow_k = k_min

        return {
            'elbow_k': elbow_k,
            'inertias': inertias,
            'note': 'Elbow method is heuristic; Gap Statistic provides statistical justification'
        }
