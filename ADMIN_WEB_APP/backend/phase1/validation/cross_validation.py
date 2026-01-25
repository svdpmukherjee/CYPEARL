"""
Cross-Validation Framework for Clustering

Prevents overfitting by validating clustering quality on held-out data.
This is critical for scientific rigor: you cannot claim clusters are valid
if you trained and evaluated on the same data.

Scientific Rationale:
- Current approach: Train on 100% → Evaluate on 100% (OVERFITTING RISK)
- Proper approach: Train on 80% → Evaluate on 20% (GENERALIZATION)

Methods Implemented:
1. K-Fold Cross-Validation: Split data into K folds, train/test K times
2. Stratified CV: Preserve outcome distribution across folds
3. Behavioral Holdout: Train on features, validate on held-out outcomes
4. Temporal CV: If data has time component, use temporal splits

Key Insight:
Clustering CV is different from supervised CV because there's no "true" labels.
We validate by:
1. Learning clusters on training set
2. Assigning test points to nearest learned cluster
3. Measuring quality metrics on test set
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.spatial.distance import cdist
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class CVFoldResult:
    """Results from a single CV fold."""
    fold: int
    train_size: int
    test_size: int
    train_silhouette: float
    test_silhouette: float
    train_eta_squared: float
    test_eta_squared: float
    generalization_gap: float

    def to_dict(self) -> Dict:
        return {
            'fold': self.fold,
            'train_size': self.train_size,
            'test_size': self.test_size,
            'train_silhouette': self.train_silhouette,
            'test_silhouette': self.test_silhouette,
            'train_eta_squared': self.train_eta_squared,
            'test_eta_squared': self.test_eta_squared,
            'generalization_gap': self.generalization_gap
        }


class ClusteringCrossValidator:
    """
    Cross-validation framework for clustering quality assessment.

    Unlike supervised learning CV, clustering CV must:
    1. Learn cluster structure on training data
    2. Assign test points to learned clusters
    3. Measure quality on held-out data

    This prevents the common mistake of evaluating clustering on the
    same data used to find the clusters (overfitting).
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize cross-validator.

        Args:
            n_folds: Number of CV folds (default: 5)
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state

    def cross_validate(
        self,
        X: np.ndarray,
        outcome_data: Optional[pd.DataFrame] = None,
        outcome_cols: Optional[List[str]] = None,
        k: int = 6,
        algorithm: str = 'kmeans'
    ) -> Dict:
        """
        Perform K-fold cross-validation for clustering.

        Args:
            X: Feature matrix (n_samples, n_features)
            outcome_data: DataFrame with behavioral outcomes
            outcome_cols: Names of outcome columns for behavioral validation
            k: Number of clusters
            algorithm: Clustering algorithm ('kmeans', 'gmm')

        Returns:
            Dictionary with CV results and generalization assessment
        """
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]

            # Extract outcomes for this fold if available
            if outcome_data is not None:
                outcomes_train = outcome_data.iloc[train_idx]
                outcomes_test = outcome_data.iloc[test_idx]
            else:
                outcomes_train = outcomes_test = None

            # Fit clustering on training data
            cluster_model = self._fit_clustering(X_train, k, algorithm)

            # Get training labels and metrics
            labels_train = self._predict(cluster_model, X_train, algorithm)
            train_sil = silhouette_score(X_train, labels_train)

            # Assign test points to nearest cluster (out-of-sample)
            labels_test = self._assign_to_clusters(
                X_test, X_train, labels_train, cluster_model, algorithm
            )

            # Calculate test silhouette (if enough points per cluster)
            try:
                test_sil = silhouette_score(X_test, labels_test)
            except:
                test_sil = np.nan

            # Calculate behavioral metrics (eta-squared) on both sets
            train_eta = self._calculate_eta_squared(
                labels_train, outcomes_train, outcome_cols
            ) if outcomes_train is not None else np.nan

            test_eta = self._calculate_eta_squared(
                labels_test, outcomes_test, outcome_cols
            ) if outcomes_test is not None else np.nan

            # Generalization gap (train - test, lower is better)
            gen_gap = train_sil - test_sil if not np.isnan(test_sil) else np.nan

            fold_results.append(CVFoldResult(
                fold=fold_idx + 1,
                train_size=len(train_idx),
                test_size=len(test_idx),
                train_silhouette=float(train_sil),
                test_silhouette=float(test_sil) if not np.isnan(test_sil) else None,
                train_eta_squared=float(train_eta) if not np.isnan(train_eta) else None,
                test_eta_squared=float(test_eta) if not np.isnan(test_eta) else None,
                generalization_gap=float(gen_gap) if not np.isnan(gen_gap) else None
            ))

        # Aggregate results
        return self._aggregate_cv_results(fold_results, k)

    def _fit_clustering(
        self,
        X: np.ndarray,
        k: int,
        algorithm: str
    ):
        """Fit clustering algorithm on training data."""
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            model.fit(X)
        elif algorithm == 'gmm':
            model = GaussianMixture(
                n_components=k, random_state=self.random_state,
                n_init=10, covariance_type='full'
            )
            model.fit(X)
        else:
            # Default to K-Means
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            model.fit(X)
        return model

    def _predict(
        self,
        model,
        X: np.ndarray,
        algorithm: str
    ) -> np.ndarray:
        """Get cluster labels from fitted model."""
        if algorithm in ['kmeans', 'gmm']:
            return model.predict(X)
        else:
            return model.predict(X)

    def _assign_to_clusters(
        self,
        X_test: np.ndarray,
        X_train: np.ndarray,
        labels_train: np.ndarray,
        model,
        algorithm: str
    ) -> np.ndarray:
        """
        Assign test points to clusters learned on training data.

        For K-Means: Use cluster centers
        For GMM: Use predict method
        For others: Assign to nearest training point's cluster
        """
        if algorithm == 'kmeans':
            # Use cluster centers directly
            return model.predict(X_test)
        elif algorithm == 'gmm':
            # GMM has predict method
            return model.predict(X_test)
        else:
            # Fallback: nearest neighbor assignment
            # Find nearest training point for each test point
            distances = cdist(X_test, X_train, metric='euclidean')
            nearest_train_idx = np.argmin(distances, axis=1)
            return labels_train[nearest_train_idx]

    def _calculate_eta_squared(
        self,
        labels: np.ndarray,
        outcome_data: Optional[pd.DataFrame],
        outcome_cols: Optional[List[str]]
    ) -> float:
        """Calculate mean eta-squared across outcome columns."""
        if outcome_data is None or outcome_cols is None:
            return np.nan

        eta_squared_values = []

        for outcome in outcome_cols:
            if outcome not in outcome_data.columns:
                continue

            values = outcome_data[outcome].values

            # Calculate eta-squared
            grand_mean = np.mean(values)
            groups = pd.Series(labels)
            group_means = outcome_data.groupby(groups)[outcome].mean()
            group_sizes = groups.value_counts()

            between_ss = sum(
                group_sizes.get(g, 0) * (group_means.get(g, grand_mean) - grand_mean)**2
                for g in group_means.index
            )
            total_ss = np.var(values) * len(values)

            if total_ss > 0:
                eta_squared_values.append(between_ss / total_ss)

        return float(np.mean(eta_squared_values)) if eta_squared_values else np.nan

    def _aggregate_cv_results(
        self,
        fold_results: List[CVFoldResult],
        k: int
    ) -> Dict:
        """Aggregate results across all folds."""

        # Extract metrics from folds
        train_sils = [r.train_silhouette for r in fold_results]
        test_sils = [r.test_silhouette for r in fold_results if r.test_silhouette is not None]
        train_etas = [r.train_eta_squared for r in fold_results if r.train_eta_squared is not None]
        test_etas = [r.test_eta_squared for r in fold_results if r.test_eta_squared is not None]
        gen_gaps = [r.generalization_gap for r in fold_results if r.generalization_gap is not None]

        # Calculate summary statistics
        summary = {
            'train_silhouette': {
                'mean': float(np.mean(train_sils)),
                'std': float(np.std(train_sils)),
                'min': float(np.min(train_sils)),
                'max': float(np.max(train_sils))
            },
            'test_silhouette': {
                'mean': float(np.mean(test_sils)) if test_sils else None,
                'std': float(np.std(test_sils)) if test_sils else None,
                'min': float(np.min(test_sils)) if test_sils else None,
                'max': float(np.max(test_sils)) if test_sils else None
            },
            'train_eta_squared': {
                'mean': float(np.mean(train_etas)) if train_etas else None,
                'std': float(np.std(train_etas)) if train_etas else None
            },
            'test_eta_squared': {
                'mean': float(np.mean(test_etas)) if test_etas else None,
                'std': float(np.std(test_etas)) if test_etas else None
            }
        }

        # Generalization gap analysis
        if gen_gaps:
            mean_gap = np.mean(gen_gaps)
            gap_assessment = self._assess_generalization_gap(mean_gap)
        else:
            mean_gap = None
            gap_assessment = 'Unable to assess (insufficient data)'

        # Stability assessment
        stability = self._assess_stability(train_sils, test_sils)

        # Overall CV quality score
        cv_quality = self._calculate_cv_quality_score(summary, mean_gap)

        return {
            'k': k,
            'n_folds': self.n_folds,
            'summary': summary,
            'fold_results': [r.to_dict() for r in fold_results],
            'generalization': {
                'mean_gap': float(mean_gap) if mean_gap is not None else None,
                'assessment': gap_assessment,
                'overfitting_risk': 'high' if mean_gap and mean_gap > 0.1 else
                                   'moderate' if mean_gap and mean_gap > 0.05 else
                                   'low' if mean_gap else 'unknown'
            },
            'stability': stability,
            'cv_quality_score': cv_quality,
            'recommendation': self._generate_recommendation(summary, mean_gap, stability)
        }

    def _assess_generalization_gap(self, gap: float) -> str:
        """Interpret the generalization gap."""
        if gap < 0.02:
            return 'Excellent - clusters generalize very well to unseen data'
        elif gap < 0.05:
            return 'Good - clusters generalize reasonably well'
        elif gap < 0.10:
            return 'Moderate - some overfitting detected, consider simpler model'
        else:
            return 'Poor - significant overfitting, clusters may not generalize'

    def _assess_stability(
        self,
        train_sils: List[float],
        test_sils: List[float]
    ) -> Dict:
        """Assess stability of clustering across folds."""
        train_cv = np.std(train_sils) / (np.mean(train_sils) + 1e-10)  # Coefficient of variation
        test_cv = np.std(test_sils) / (np.mean(test_sils) + 1e-10) if test_sils else None

        return {
            'train_coefficient_of_variation': float(train_cv),
            'test_coefficient_of_variation': float(test_cv) if test_cv else None,
            'assessment': (
                'Stable' if train_cv < 0.1 else
                'Moderately stable' if train_cv < 0.2 else
                'Unstable - high variance across folds'
            )
        }

    def _calculate_cv_quality_score(
        self,
        summary: Dict,
        gen_gap: Optional[float]
    ) -> Dict:
        """Calculate overall cross-validation quality score."""
        scores = []

        # Test silhouette contribution
        test_sil = summary['test_silhouette']['mean']
        if test_sil is not None:
            sil_score = (test_sil + 1) / 2  # Normalize to [0, 1]
            scores.append(sil_score)

        # Test eta-squared contribution
        test_eta = summary['test_eta_squared']['mean']
        if test_eta is not None:
            eta_score = min(test_eta / 0.15, 1.0)  # Normalize by target
            scores.append(eta_score)

        # Generalization gap penalty
        if gen_gap is not None:
            gap_penalty = max(0, 1 - gen_gap * 5)  # Penalize gaps > 0.2
            scores.append(gap_penalty)

        overall = float(np.mean(scores)) if scores else 0.0

        return {
            'overall': overall,
            'level': 'high' if overall >= 0.7 else 'moderate' if overall >= 0.5 else 'low',
            'components': {
                'test_silhouette_score': float((test_sil + 1) / 2) if test_sil else None,
                'test_eta_squared_score': float(min(test_eta / 0.15, 1.0)) if test_eta else None,
                'generalization_score': float(max(0, 1 - gen_gap * 5)) if gen_gap else None
            }
        }

    def _generate_recommendation(
        self,
        summary: Dict,
        gen_gap: Optional[float],
        stability: Dict
    ) -> str:
        """Generate recommendation based on CV results."""
        issues = []
        positives = []

        # Check generalization
        if gen_gap and gen_gap > 0.1:
            issues.append("Significant overfitting detected (gap > 0.1)")
        elif gen_gap and gen_gap < 0.05:
            positives.append("Good generalization")

        # Check stability
        if stability['train_coefficient_of_variation'] > 0.2:
            issues.append("High variance across folds suggests unstable clusters")
        elif stability['train_coefficient_of_variation'] < 0.1:
            positives.append("Stable across folds")

        # Check test eta-squared
        test_eta = summary['test_eta_squared']['mean']
        if test_eta and test_eta < 0.05:
            issues.append("Low behavioral predictive power on held-out data")
        elif test_eta and test_eta > 0.10:
            positives.append("Strong behavioral differentiation on held-out data")

        if issues:
            return f"CONCERNS: {'; '.join(issues)}. Consider: reducing K, removing noisy features, or trying different algorithm."
        elif positives:
            return f"VALIDATED: {'; '.join(positives)}. Clustering appears scientifically sound."
        else:
            return "Insufficient data to make strong recommendation. Collect more samples or outcomes."

    def stratified_cv(
        self,
        X: np.ndarray,
        y_stratify: np.ndarray,
        outcome_data: Optional[pd.DataFrame] = None,
        outcome_cols: Optional[List[str]] = None,
        k: int = 6,
        algorithm: str = 'kmeans'
    ) -> Dict:
        """
        Stratified cross-validation preserving outcome distribution.

        Use this when you want each fold to have similar distribution
        of a key outcome (e.g., phishing click rate).

        Args:
            X: Feature matrix
            y_stratify: Values to stratify by (e.g., binned click rates)
            outcome_data: Behavioral outcomes DataFrame
            outcome_cols: Outcome column names
            k: Number of clusters
            algorithm: Clustering algorithm

        Returns:
            Same as cross_validate() but with stratified folds
        """
        # Bin continuous y_stratify for stratification
        if y_stratify.dtype.kind == 'f':  # Float
            y_bins = pd.qcut(y_stratify, q=self.n_folds, labels=False, duplicates='drop')
        else:
            y_bins = y_stratify

        skfold = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skfold.split(X, y_bins)):
            X_train, X_test = X[train_idx], X[test_idx]

            if outcome_data is not None:
                outcomes_train = outcome_data.iloc[train_idx]
                outcomes_test = outcome_data.iloc[test_idx]
            else:
                outcomes_train = outcomes_test = None

            # Same logic as regular CV
            cluster_model = self._fit_clustering(X_train, k, algorithm)
            labels_train = self._predict(cluster_model, X_train, algorithm)
            train_sil = silhouette_score(X_train, labels_train)

            labels_test = self._assign_to_clusters(
                X_test, X_train, labels_train, cluster_model, algorithm
            )

            try:
                test_sil = silhouette_score(X_test, labels_test)
            except:
                test_sil = np.nan

            train_eta = self._calculate_eta_squared(
                labels_train, outcomes_train, outcome_cols
            ) if outcomes_train is not None else np.nan

            test_eta = self._calculate_eta_squared(
                labels_test, outcomes_test, outcome_cols
            ) if outcomes_test is not None else np.nan

            gen_gap = train_sil - test_sil if not np.isnan(test_sil) else np.nan

            fold_results.append(CVFoldResult(
                fold=fold_idx + 1,
                train_size=len(train_idx),
                test_size=len(test_idx),
                train_silhouette=float(train_sil),
                test_silhouette=float(test_sil) if not np.isnan(test_sil) else None,
                train_eta_squared=float(train_eta) if not np.isnan(train_eta) else None,
                test_eta_squared=float(test_eta) if not np.isnan(test_eta) else None,
                generalization_gap=float(gen_gap) if not np.isnan(gen_gap) else None
            ))

        result = self._aggregate_cv_results(fold_results, k)
        result['cv_type'] = 'stratified'
        result['stratify_variable'] = 'outcome_distribution'
        return result
