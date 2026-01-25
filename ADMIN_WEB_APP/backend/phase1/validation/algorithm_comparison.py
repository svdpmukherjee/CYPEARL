"""
Statistical Comparison of Clustering Algorithms

When comparing K-Means vs. GMM vs. Hierarchical vs. Spectral, we need
to determine if differences in composite scores are STATISTICALLY SIGNIFICANT
or just random variation.

Scientific Rationale:
- Current: "GMM scored 0.78, K-Means scored 0.76, so GMM wins"
- Problem: Is 0.78 > 0.76 significant, or noise?
- Solution: Bootstrap confidence intervals and statistical tests

Methods:
1. Paired Bootstrap Test: Compare algorithms on same bootstrap samples
2. Confidence Intervals: Provide uncertainty bounds on scores
3. Effect Size: Measure practical significance of differences
4. Multiple Comparison Correction: Bonferroni/Holm for many algorithms
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class AlgorithmScore:
    """Score summary for a single algorithm."""
    algorithm: str
    k: int
    mean_score: float
    std_score: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int

    def to_dict(self) -> Dict:
        return {
            'algorithm': self.algorithm,
            'k': self.k,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'ci_width': self.ci_upper - self.ci_lower,
            'n_bootstrap': self.n_bootstrap
        }


@dataclass
class PairwiseComparison:
    """Statistical comparison between two algorithms."""
    algorithm_a: str
    algorithm_b: str
    mean_difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    effect_size: float
    winner: Optional[str]

    def to_dict(self) -> Dict:
        return {
            'algorithm_a': self.algorithm_a,
            'algorithm_b': self.algorithm_b,
            'mean_difference': self.mean_difference,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'p_value': self.p_value,
            'significant': self.significant,
            'effect_size': self.effect_size,
            'effect_size_interpretation': (
                'large' if abs(self.effect_size) >= 0.8 else
                'medium' if abs(self.effect_size) >= 0.5 else
                'small' if abs(self.effect_size) >= 0.2 else
                'negligible'
            ),
            'winner': self.winner
        }


class AlgorithmComparisonAnalyzer:
    """
    Statistically compare clustering algorithms.

    Provides rigorous statistical testing to determine if
    differences between algorithms are significant.
    """

    ALGORITHMS = {
        'kmeans': KMeans,
        'gmm': GaussianMixture,
        'hierarchical': AgglomerativeClustering,
        'spectral': SpectralClustering
    }

    def __init__(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        random_state: int = 42
    ):
        """
        Initialize analyzer.

        Args:
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level for intervals (default: 0.95)
            random_state: Random seed
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.alpha = 1 - confidence_level

    def compare(
        self,
        X: np.ndarray,
        algorithms: List[str],
        k: int,
        outcome_data: Optional[pd.DataFrame] = None,
        outcome_cols: Optional[List[str]] = None,
        metric: str = 'composite'
    ) -> Dict:
        """
        Compare multiple algorithms with statistical rigor.

        Args:
            X: Feature matrix
            algorithms: List of algorithm names
            k: Number of clusters
            outcome_data: Behavioral outcomes (for composite score)
            outcome_cols: Outcome column names
            metric: 'silhouette', 'eta_squared', or 'composite'

        Returns:
            Comprehensive comparison with statistical tests
        """
        # 1. Bootstrap evaluation for each algorithm
        algorithm_scores = {}
        bootstrap_scores = {}

        for algo in algorithms:
            scores, bootstrap_results = self._bootstrap_evaluate(
                X, algo, k, outcome_data, outcome_cols, metric
            )
            algorithm_scores[algo] = scores
            bootstrap_scores[algo] = bootstrap_results

        # 2. Pairwise comparisons with statistical tests
        pairwise_comparisons = self._pairwise_comparisons(
            algorithms, bootstrap_scores
        )

        # 3. Multiple comparison correction
        corrected_comparisons = self._multiple_comparison_correction(
            pairwise_comparisons
        )

        # 4. Rank algorithms
        ranking = self._rank_algorithms(algorithm_scores)

        # 5. Best algorithm determination
        best_algorithm = self._determine_best(
            algorithm_scores, corrected_comparisons
        )

        return {
            'algorithm_scores': {a: s.to_dict() for a, s in algorithm_scores.items()},
            'pairwise_comparisons': [c.to_dict() for c in corrected_comparisons],
            'ranking': ranking,
            'best_algorithm': best_algorithm,
            'statistical_summary': self._generate_summary(
                algorithm_scores, corrected_comparisons, best_algorithm
            ),
            'recommendation': self._generate_recommendation(
                best_algorithm, corrected_comparisons
            )
        }

    def _bootstrap_evaluate(
        self,
        X: np.ndarray,
        algorithm: str,
        k: int,
        outcome_data: Optional[pd.DataFrame],
        outcome_cols: Optional[List[str]],
        metric: str
    ) -> Tuple[AlgorithmScore, List[float]]:
        """Bootstrap evaluation of a single algorithm."""
        n_samples = X.shape[0]
        scores = []

        np.random.seed(self.random_state)

        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]

            if outcome_data is not None:
                outcomes_boot = outcome_data.iloc[indices]
            else:
                outcomes_boot = None

            # Cluster
            labels = self._cluster(X_boot, algorithm, k, seed=self.random_state + i)

            # Calculate metric
            score = self._calculate_metric(
                X_boot, labels, k, outcomes_boot, outcome_cols, metric
            )

            if score is not None:
                scores.append(score)

        if len(scores) < 10:
            # Not enough successful runs
            return AlgorithmScore(
                algorithm=algorithm,
                k=k,
                mean_score=np.nan,
                std_score=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                n_bootstrap=len(scores)
            ), scores

        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_lower = np.percentile(scores, (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(scores, (1 + self.confidence_level) / 2 * 100)

        return AlgorithmScore(
            algorithm=algorithm,
            k=k,
            mean_score=float(mean_score),
            std_score=float(std_score),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            n_bootstrap=len(scores)
        ), scores

    def _cluster(
        self,
        X: np.ndarray,
        algorithm: str,
        k: int,
        seed: int
    ) -> np.ndarray:
        """Run clustering with specified algorithm."""
        try:
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=k, random_state=seed, n_init=5)
                return model.fit_predict(X)
            elif algorithm == 'gmm':
                model = GaussianMixture(n_components=k, random_state=seed, n_init=5)
                return model.fit_predict(X)
            elif algorithm == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                return model.fit_predict(X)
            elif algorithm == 'spectral':
                model = SpectralClustering(
                    n_clusters=k, random_state=seed,
                    affinity='nearest_neighbors', n_neighbors=min(10, X.shape[0] - 1)
                )
                return model.fit_predict(X)
            else:
                # Default to K-Means
                model = KMeans(n_clusters=k, random_state=seed, n_init=5)
                return model.fit_predict(X)
        except Exception:
            # Return random labels as fallback
            return np.random.randint(0, k, X.shape[0])

    def _calculate_metric(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        k: int,
        outcome_data: Optional[pd.DataFrame],
        outcome_cols: Optional[List[str]],
        metric: str
    ) -> Optional[float]:
        """Calculate the specified metric."""
        # Check for degenerate clustering
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return None

        try:
            if metric == 'silhouette':
                return silhouette_score(X, labels)

            elif metric == 'eta_squared':
                if outcome_data is None or outcome_cols is None:
                    return None
                return self._calculate_eta_squared(labels, outcome_data, outcome_cols)

            elif metric == 'composite':
                # Simplified composite: 0.5 * silhouette_normalized + 0.5 * eta_squared_normalized
                sil = silhouette_score(X, labels)
                sil_norm = (sil + 1) / 2  # Map [-1, 1] to [0, 1]

                if outcome_data is not None and outcome_cols is not None:
                    eta = self._calculate_eta_squared(labels, outcome_data, outcome_cols)
                    eta_norm = min(eta / 0.15, 1.0)  # Cap at 1.0
                    return 0.5 * sil_norm + 0.5 * eta_norm
                else:
                    return sil_norm

            else:
                return silhouette_score(X, labels)

        except Exception:
            return None

    def _calculate_eta_squared(
        self,
        labels: np.ndarray,
        outcome_data: pd.DataFrame,
        outcome_cols: List[str]
    ) -> float:
        """Calculate mean eta-squared across outcomes."""
        eta_values = []

        for outcome in outcome_cols:
            if outcome not in outcome_data.columns:
                continue

            values = outcome_data[outcome].values
            groups = pd.Series(labels)

            grand_mean = np.mean(values)
            group_means = outcome_data.groupby(groups)[outcome].mean()
            group_sizes = groups.value_counts()

            between_ss = sum(
                group_sizes.get(g, 0) * (group_means.get(g, grand_mean) - grand_mean)**2
                for g in group_means.index
            )
            total_ss = np.var(values) * len(values)

            if total_ss > 0:
                eta_values.append(between_ss / total_ss)

        return float(np.mean(eta_values)) if eta_values else 0.0

    def _pairwise_comparisons(
        self,
        algorithms: List[str],
        bootstrap_scores: Dict[str, List[float]]
    ) -> List[PairwiseComparison]:
        """Perform pairwise statistical comparisons."""
        comparisons = []

        for i, algo_a in enumerate(algorithms):
            for algo_b in algorithms[i+1:]:
                scores_a = np.array(bootstrap_scores[algo_a])
                scores_b = np.array(bootstrap_scores[algo_b])

                # Ensure same length (use min)
                n = min(len(scores_a), len(scores_b))
                if n < 10:
                    continue

                scores_a = scores_a[:n]
                scores_b = scores_b[:n]

                # Paired differences
                differences = scores_a - scores_b
                mean_diff = np.mean(differences)

                # Confidence interval on difference
                ci_lower = np.percentile(differences, (1 - self.confidence_level) / 2 * 100)
                ci_upper = np.percentile(differences, (1 + self.confidence_level) / 2 * 100)

                # P-value: proportion of differences with opposite sign
                # (Two-sided test: is the difference significantly different from 0?)
                if mean_diff > 0:
                    p_value = np.mean(differences <= 0) * 2  # Two-sided
                else:
                    p_value = np.mean(differences >= 0) * 2

                p_value = min(p_value, 1.0)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

                # Significance at alpha level
                significant = (ci_lower > 0) or (ci_upper < 0)

                # Winner
                if significant:
                    winner = algo_a if mean_diff > 0 else algo_b
                else:
                    winner = None

                comparisons.append(PairwiseComparison(
                    algorithm_a=algo_a,
                    algorithm_b=algo_b,
                    mean_difference=float(mean_diff),
                    ci_lower=float(ci_lower),
                    ci_upper=float(ci_upper),
                    p_value=float(p_value),
                    significant=significant,
                    effect_size=float(effect_size),
                    winner=winner
                ))

        return comparisons

    def _multiple_comparison_correction(
        self,
        comparisons: List[PairwiseComparison]
    ) -> List[PairwiseComparison]:
        """Apply Holm-Bonferroni correction for multiple comparisons."""
        if not comparisons:
            return comparisons

        # Sort by p-value
        sorted_comparisons = sorted(comparisons, key=lambda c: c.p_value)

        n_comparisons = len(comparisons)
        corrected = []

        for i, comp in enumerate(sorted_comparisons):
            # Holm-Bonferroni: compare p-value to alpha / (n - i)
            adjusted_alpha = self.alpha / (n_comparisons - i)
            corrected_significant = comp.p_value < adjusted_alpha

            # Create new comparison with corrected significance
            corrected.append(PairwiseComparison(
                algorithm_a=comp.algorithm_a,
                algorithm_b=comp.algorithm_b,
                mean_difference=comp.mean_difference,
                ci_lower=comp.ci_lower,
                ci_upper=comp.ci_upper,
                p_value=comp.p_value,
                significant=corrected_significant,
                effect_size=comp.effect_size,
                winner=comp.winner if corrected_significant else None
            ))

        return corrected

    def _rank_algorithms(
        self,
        algorithm_scores: Dict[str, AlgorithmScore]
    ) -> List[Dict]:
        """Rank algorithms by mean score."""
        rankings = []

        sorted_algos = sorted(
            algorithm_scores.items(),
            key=lambda x: x[1].mean_score if not np.isnan(x[1].mean_score) else -1,
            reverse=True
        )

        for rank, (algo, score) in enumerate(sorted_algos, 1):
            rankings.append({
                'rank': rank,
                'algorithm': algo,
                'mean_score': float(score.mean_score) if not np.isnan(score.mean_score) else None,
                'ci_width': float(score.ci_upper - score.ci_lower) if not np.isnan(score.ci_upper) else None
            })

        return rankings

    def _determine_best(
        self,
        algorithm_scores: Dict[str, AlgorithmScore],
        comparisons: List[PairwiseComparison]
    ) -> Dict:
        """Determine best algorithm with statistical justification."""
        # Find algorithm with highest mean score
        valid_algos = {a: s for a, s in algorithm_scores.items()
                       if not np.isnan(s.mean_score)}

        if not valid_algos:
            return {'algorithm': None, 'confidence': 'none', 'reason': 'No valid results'}

        best_algo = max(valid_algos.items(), key=lambda x: x[1].mean_score)[0]
        best_score = valid_algos[best_algo]

        # Check if best is significantly better than all others
        significantly_better = True
        for comp in comparisons:
            if comp.algorithm_a == best_algo and comp.significant and comp.winner != best_algo:
                significantly_better = False
                break
            if comp.algorithm_b == best_algo and comp.significant and comp.winner != best_algo:
                significantly_better = False
                break

        # Check if best is significantly better than at least one other
        beats_at_least_one = any(
            comp.significant and comp.winner == best_algo
            for comp in comparisons
        )

        # Determine confidence level
        if significantly_better and beats_at_least_one:
            confidence = 'high'
            reason = f'{best_algo} is statistically significantly better than all other algorithms'
        elif beats_at_least_one:
            confidence = 'moderate'
            reason = f'{best_algo} is significantly better than at least one alternative'
        else:
            confidence = 'low'
            reason = f'{best_algo} has highest mean but differences are not statistically significant'

        return {
            'algorithm': best_algo,
            'mean_score': float(best_score.mean_score),
            'ci': [float(best_score.ci_lower), float(best_score.ci_upper)],
            'confidence': confidence,
            'reason': reason
        }

    def _generate_summary(
        self,
        algorithm_scores: Dict[str, AlgorithmScore],
        comparisons: List[PairwiseComparison],
        best_algorithm: Dict
    ) -> Dict:
        """Generate statistical summary."""
        n_significant = sum(1 for c in comparisons if c.significant)
        n_comparisons = len(comparisons)

        return {
            'n_algorithms_compared': len(algorithm_scores),
            'n_pairwise_comparisons': n_comparisons,
            'n_significant_differences': n_significant,
            'pct_significant': float(n_significant / n_comparisons * 100) if n_comparisons > 0 else 0,
            'best_algorithm': best_algorithm['algorithm'],
            'selection_confidence': best_algorithm['confidence'],
            'bootstrap_iterations': self.n_bootstrap,
            'confidence_level': self.confidence_level
        }

    def _generate_recommendation(
        self,
        best_algorithm: Dict,
        comparisons: List[PairwiseComparison]
    ) -> str:
        """Generate recommendation based on comparison."""
        algo = best_algorithm['algorithm']
        confidence = best_algorithm['confidence']

        if confidence == 'high':
            return (
                f"STRONG RECOMMENDATION: Use {algo}. "
                f"It is statistically significantly better than all alternatives tested. "
                f"This result is robust across {self.n_bootstrap} bootstrap samples."
            )
        elif confidence == 'moderate':
            return (
                f"MODERATE RECOMMENDATION: Use {algo}. "
                f"It performs best on average, but some alternatives are statistically similar. "
                f"Consider practical factors (speed, interpretability) in final decision."
            )
        else:
            # Find close alternatives
            close_algos = [c.algorithm_b if c.algorithm_a == algo else c.algorithm_a
                          for c in comparisons
                          if not c.significant and (c.algorithm_a == algo or c.algorithm_b == algo)]

            return (
                f"WEAK RECOMMENDATION: {algo} has highest mean score, but differences are not "
                f"statistically significant vs. {', '.join(close_algos[:2])}. "
                f"Any of these algorithms would be defensible choices."
            )
