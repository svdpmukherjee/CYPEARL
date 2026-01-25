"""
Prediction Error Framework for Clustering Validation

The fundamental question: "Can cluster membership PREDICT individual behavior?"

Current approach validates that clusters DIFFER on outcomes (eta-squared).
But this doesn't tell you if you can PREDICT an individual's behavior
from their cluster membership.

Scientific Rationale:
- Eta-squared measures: "How much variance is explained by clusters?"
- Prediction error measures: "How accurately can we predict individual outcomes?"

Key Metrics:
1. R² (R-squared): Variance explained when using cluster mean as prediction
2. MAE (Mean Absolute Error): Average prediction error per individual
3. RMSE (Root Mean Squared Error): Emphasizes large errors
4. Prediction Interval Coverage: Do cluster-based intervals capture individuals?

This is the missing link between "clusters are different" and "clusters are useful".
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PredictionMetrics:
    """Prediction metrics for a single outcome."""
    outcome_name: str
    r_squared: float
    mae: float
    rmse: float
    mape: float  # Mean Absolute Percentage Error
    prediction_interval_coverage: float
    within_cluster_std: float
    between_cluster_std: float

    def to_dict(self) -> Dict:
        return {
            'outcome_name': self.outcome_name,
            'r_squared': self.r_squared,
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'prediction_interval_coverage': self.prediction_interval_coverage,
            'within_cluster_std': self.within_cluster_std,
            'between_cluster_std': self.between_cluster_std
        }


class PredictionErrorAnalyzer:
    """
    Analyze predictive power of clustering for individual-level outcomes.

    Unlike eta-squared (which measures group differences), this measures
    how well cluster membership predicts INDIVIDUAL behavior.

    Key Question: If I assign someone to cluster X, how accurately can I
    predict their phishing click rate using cluster X's mean?
    """

    def __init__(self, random_state: int = 42):
        """Initialize analyzer."""
        self.random_state = random_state

    def analyze(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        outcome_data: pd.DataFrame,
        outcome_cols: List[str]
    ) -> Dict:
        """
        Comprehensive prediction error analysis.

        Args:
            X: Feature matrix (n_samples, n_features)
            labels: Cluster assignments
            outcome_data: DataFrame with behavioral outcomes
            outcome_cols: Names of outcome columns to analyze

        Returns:
            Dictionary with prediction metrics and interpretations
        """
        results = []

        for outcome in outcome_cols:
            if outcome not in outcome_data.columns:
                continue

            y_true = outcome_data[outcome].values

            # Skip if no variance
            if np.std(y_true) < 1e-10:
                continue

            metrics = self._calculate_prediction_metrics(
                labels, y_true, outcome
            )
            results.append(metrics)

        # Aggregate across outcomes
        aggregated = self._aggregate_results(results)

        # Individual-level accuracy assessment
        individual_accuracy = self._individual_accuracy_analysis(
            labels, outcome_data, outcome_cols
        )

        # Compare to naive baselines
        baseline_comparison = self._compare_to_baselines(
            labels, outcome_data, outcome_cols
        )

        # Cluster-specific predictive power
        cluster_predictive_power = self._cluster_specific_analysis(
            labels, outcome_data, outcome_cols
        )

        return {
            'outcome_metrics': [r.to_dict() for r in results],
            'aggregated': aggregated,
            'individual_accuracy': individual_accuracy,
            'baseline_comparison': baseline_comparison,
            'cluster_predictive_power': cluster_predictive_power,
            'interpretation': self._interpret_results(aggregated, baseline_comparison),
            'recommendation': self._generate_recommendation(aggregated, baseline_comparison)
        }

    def _calculate_prediction_metrics(
        self,
        labels: np.ndarray,
        y_true: np.ndarray,
        outcome_name: str
    ) -> PredictionMetrics:
        """Calculate prediction metrics for a single outcome."""

        # Create DataFrame for grouping
        df = pd.DataFrame({'cluster': labels, 'outcome': y_true})

        # Cluster means as predictions
        cluster_means = df.groupby('cluster')['outcome'].mean()
        y_pred = df['cluster'].map(cluster_means)

        # Within-cluster standard deviations
        cluster_stds = df.groupby('cluster')['outcome'].std()
        within_std = float(cluster_stds.mean())

        # Between-cluster standard deviation
        between_std = float(cluster_means.std())

        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAPE (handle division by zero)
        non_zero_mask = np.abs(y_true) > 1e-10
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])))
        else:
            mape = np.nan

        # Prediction interval coverage (95% CI)
        coverage = self._calculate_interval_coverage(
            df, cluster_means, cluster_stds, confidence=0.95
        )

        return PredictionMetrics(
            outcome_name=outcome_name,
            r_squared=float(r2),
            mae=float(mae),
            rmse=float(rmse),
            mape=float(mape) if not np.isnan(mape) else None,
            prediction_interval_coverage=float(coverage),
            within_cluster_std=within_std,
            between_cluster_std=between_std
        )

    def _calculate_interval_coverage(
        self,
        df: pd.DataFrame,
        cluster_means: pd.Series,
        cluster_stds: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate what percentage of individuals fall within
        their cluster's prediction interval.

        A well-calibrated clustering should have ~95% coverage
        for a 95% interval.
        """
        z_score = stats.norm.ppf(1 - (1 - confidence) / 2)  # ~1.96 for 95%

        n_covered = 0
        n_total = len(df)

        for _, row in df.iterrows():
            cluster = row['cluster']
            value = row['outcome']

            mean = cluster_means.get(cluster, 0)
            std = cluster_stds.get(cluster, 0)

            if std == 0 or np.isnan(std):
                std = 0.01  # Small default to avoid division issues

            lower = mean - z_score * std
            upper = mean + z_score * std

            if lower <= value <= upper:
                n_covered += 1

        return n_covered / n_total if n_total > 0 else 0.0

    def _aggregate_results(
        self,
        results: List[PredictionMetrics]
    ) -> Dict:
        """Aggregate metrics across all outcomes."""
        if not results:
            return {}

        r2_values = [r.r_squared for r in results]
        mae_values = [r.mae for r in results]
        coverage_values = [r.prediction_interval_coverage for r in results]

        return {
            'mean_r_squared': float(np.mean(r2_values)),
            'std_r_squared': float(np.std(r2_values)),
            'mean_mae': float(np.mean(mae_values)),
            'mean_coverage': float(np.mean(coverage_values)),
            'n_outcomes': len(results),
            'best_predicted_outcome': max(results, key=lambda r: r.r_squared).outcome_name,
            'worst_predicted_outcome': min(results, key=lambda r: r.r_squared).outcome_name
        }

    def _individual_accuracy_analysis(
        self,
        labels: np.ndarray,
        outcome_data: pd.DataFrame,
        outcome_cols: List[str]
    ) -> Dict:
        """
        Analyze how many individuals are well-represented by their cluster.

        Key insight: Even if clusters differ on average, some individuals
        may be poorly represented (outliers within their cluster).
        """
        # Focus on primary outcome (phishing_click_rate)
        primary = 'phishing_click_rate' if 'phishing_click_rate' in outcome_cols else outcome_cols[0]

        if primary not in outcome_data.columns:
            return {'error': 'Primary outcome not found'}

        y = outcome_data[primary].values
        df = pd.DataFrame({'cluster': labels, 'outcome': y})

        cluster_means = df.groupby('cluster')['outcome'].mean()
        cluster_stds = df.groupby('cluster')['outcome'].std()

        # Calculate z-score for each individual within their cluster
        z_scores = []
        for _, row in df.iterrows():
            cluster = row['cluster']
            value = row['outcome']
            mean = cluster_means[cluster]
            std = cluster_stds.get(cluster, 0.01)
            if std < 0.01:
                std = 0.01
            z = abs(value - mean) / std
            z_scores.append(z)

        z_scores = np.array(z_scores)

        # Categorize individuals
        n_well_represented = np.sum(z_scores < 1)  # Within 1 std
        n_moderately = np.sum((z_scores >= 1) & (z_scores < 2))  # 1-2 std
        n_poorly = np.sum(z_scores >= 2)  # Beyond 2 std

        return {
            'primary_outcome': primary,
            'well_represented_pct': float(n_well_represented / len(z_scores)) * 100,
            'moderately_represented_pct': float(n_moderately / len(z_scores)) * 100,
            'poorly_represented_pct': float(n_poorly / len(z_scores)) * 100,
            'mean_z_score': float(np.mean(z_scores)),
            'interpretation': (
                f'{n_well_represented/len(z_scores)*100:.1f}% of individuals are well-represented '
                f'by their cluster mean (within 1 std). '
                f'{n_poorly/len(z_scores)*100:.1f}% are outliers (>2 std from cluster mean).'
            )
        }

    def _compare_to_baselines(
        self,
        labels: np.ndarray,
        outcome_data: pd.DataFrame,
        outcome_cols: List[str]
    ) -> Dict:
        """
        Compare cluster-based predictions to naive baselines.

        If clustering doesn't beat these baselines, it's not useful for prediction.

        Baselines:
        1. Grand mean: Predict everyone with population mean
        2. Random assignment: Random cluster membership
        """
        comparisons = {}

        for outcome in outcome_cols:
            if outcome not in outcome_data.columns:
                continue

            y = outcome_data[outcome].values
            if np.std(y) < 1e-10:
                continue

            # Cluster-based prediction
            df = pd.DataFrame({'cluster': labels, 'outcome': y})
            cluster_means = df.groupby('cluster')['outcome'].mean()
            y_pred_cluster = df['cluster'].map(cluster_means)
            cluster_mae = mean_absolute_error(y, y_pred_cluster)
            cluster_r2 = r2_score(y, y_pred_cluster)

            # Baseline 1: Grand mean
            grand_mean = np.mean(y)
            y_pred_grand = np.full_like(y, grand_mean)
            grand_mae = mean_absolute_error(y, y_pred_grand)
            grand_r2 = 0.0  # By definition, grand mean has R² = 0

            # Baseline 2: Random clusters
            np.random.seed(self.random_state)
            random_labels = np.random.randint(0, len(np.unique(labels)), len(labels))
            df_random = pd.DataFrame({'cluster': random_labels, 'outcome': y})
            random_means = df_random.groupby('cluster')['outcome'].mean()
            y_pred_random = df_random['cluster'].map(random_means)
            random_mae = mean_absolute_error(y, y_pred_random)
            random_r2 = r2_score(y, y_pred_random)

            # Improvement over baselines
            improvement_over_grand = (grand_mae - cluster_mae) / grand_mae * 100 if grand_mae > 0 else 0
            improvement_over_random = (random_mae - cluster_mae) / random_mae * 100 if random_mae > 0 else 0

            comparisons[outcome] = {
                'cluster_mae': float(cluster_mae),
                'cluster_r2': float(cluster_r2),
                'grand_mean_mae': float(grand_mae),
                'random_mae': float(random_mae),
                'random_r2': float(random_r2),
                'improvement_over_grand_pct': float(improvement_over_grand),
                'improvement_over_random_pct': float(improvement_over_random),
                'beats_grand_mean': cluster_mae < grand_mae,
                'beats_random': cluster_mae < random_mae
            }

        # Summary
        if comparisons:
            all_beat_grand = all(c['beats_grand_mean'] for c in comparisons.values())
            all_beat_random = all(c['beats_random'] for c in comparisons.values())
            mean_improvement = np.mean([c['improvement_over_grand_pct'] for c in comparisons.values()])
        else:
            all_beat_grand = False
            all_beat_random = False
            mean_improvement = 0

        return {
            'by_outcome': comparisons,
            'summary': {
                'beats_all_baselines': all_beat_grand and all_beat_random,
                'mean_improvement_over_grand_pct': float(mean_improvement),
                'all_outcomes_beat_grand_mean': all_beat_grand,
                'all_outcomes_beat_random': all_beat_random
            }
        }

    def _cluster_specific_analysis(
        self,
        labels: np.ndarray,
        outcome_data: pd.DataFrame,
        outcome_cols: List[str]
    ) -> Dict:
        """
        Analyze predictive power per cluster.

        Some clusters may be more homogeneous (better prediction)
        than others (high variance → poor prediction).
        """
        primary = 'phishing_click_rate' if 'phishing_click_rate' in outcome_cols else outcome_cols[0]

        if primary not in outcome_data.columns:
            return {}

        y = outcome_data[primary].values
        df = pd.DataFrame({'cluster': labels, 'outcome': y})

        cluster_stats = df.groupby('cluster')['outcome'].agg(['mean', 'std', 'count'])

        cluster_analysis = {}
        for cluster_id, row in cluster_stats.iterrows():
            cv = row['std'] / (row['mean'] + 1e-10)  # Coefficient of variation

            # Predictability assessment
            if cv < 0.3:
                predictability = 'high'
            elif cv < 0.5:
                predictability = 'moderate'
            else:
                predictability = 'low'

            cluster_analysis[int(cluster_id)] = {
                'mean': float(row['mean']),
                'std': float(row['std']),
                'n': int(row['count']),
                'coefficient_of_variation': float(cv),
                'predictability': predictability
            }

        # Identify best and worst clusters for prediction
        sorted_clusters = sorted(cluster_analysis.items(), key=lambda x: x[1]['coefficient_of_variation'])

        return {
            'by_cluster': cluster_analysis,
            'most_predictable_cluster': sorted_clusters[0][0] if sorted_clusters else None,
            'least_predictable_cluster': sorted_clusters[-1][0] if sorted_clusters else None,
            'primary_outcome': primary
        }

    def _interpret_results(
        self,
        aggregated: Dict,
        baseline_comparison: Dict
    ) -> Dict:
        """Interpret the prediction results."""
        if not aggregated:
            return {'error': 'No results to interpret'}

        r2 = aggregated.get('mean_r_squared', 0)
        beats_baselines = baseline_comparison.get('summary', {}).get('beats_all_baselines', False)
        improvement = baseline_comparison.get('summary', {}).get('mean_improvement_over_grand_pct', 0)

        # R² interpretation (Cohen's guidelines adapted for clustering)
        if r2 >= 0.25:
            r2_level = 'strong'
            r2_desc = 'Clusters explain substantial variance in outcomes'
        elif r2 >= 0.10:
            r2_level = 'moderate'
            r2_desc = 'Clusters explain meaningful variance'
        elif r2 >= 0.02:
            r2_level = 'weak'
            r2_desc = 'Clusters explain only small amount of variance'
        else:
            r2_level = 'negligible'
            r2_desc = 'Clusters have minimal predictive value'

        return {
            'predictive_power_level': r2_level,
            'predictive_power_description': r2_desc,
            'beats_naive_baselines': beats_baselines,
            'improvement_magnitude': (
                'substantial' if improvement > 20 else
                'moderate' if improvement > 10 else
                'marginal' if improvement > 5 else
                'minimal'
            ),
            'scientific_validity': (
                'VALID: Clustering provides meaningful predictive value beyond baselines'
                if r2 >= 0.10 and beats_baselines else
                'QUESTIONABLE: Predictive power is weak or doesn\'t beat baselines'
            )
        }

    def _generate_recommendation(
        self,
        aggregated: Dict,
        baseline_comparison: Dict
    ) -> str:
        """Generate actionable recommendation."""
        if not aggregated:
            return "Insufficient data for recommendation."

        r2 = aggregated.get('mean_r_squared', 0)
        beats_baselines = baseline_comparison.get('summary', {}).get('beats_all_baselines', False)

        if r2 >= 0.15 and beats_baselines:
            return (
                "PROCEED: Clustering has strong predictive power (R² = {:.2f}). "
                "Cluster membership meaningfully predicts individual behavior. "
                "This supports using clusters for LLM persona conditioning.".format(r2)
            )
        elif r2 >= 0.08 and beats_baselines:
            return (
                "PROCEED WITH CAUTION: Clustering has moderate predictive power (R² = {:.2f}). "
                "Consider augmenting cluster-level predictions with individual-level features "
                "for more accurate persona simulation.".format(r2)
            )
        elif beats_baselines:
            return (
                "RECONSIDER: Clustering beats baselines but predictive power is weak (R² = {:.2f}). "
                "The clusters may not provide sufficient individual-level accuracy. "
                "Consider: more features, different K, or hierarchical clustering.".format(r2)
            )
        else:
            return (
                "CONCERN: Clustering does not beat naive baselines. "
                "The current clustering may not be useful for prediction. "
                "Recommended: revisit feature selection, try different algorithms, "
                "or collect more discriminative features."
            )
