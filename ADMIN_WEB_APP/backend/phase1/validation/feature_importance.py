"""
Feature Importance Analysis for Clustering

Determines which psychological traits actually drive cluster separation
and behavioral differentiation. This addresses the question: "Do all 29
traits contribute meaningfully, or could we achieve the same clustering
with fewer features?"

Scientific Rationale:
1. Permutation Importance: How much does clustering quality degrade when
   a feature is randomly shuffled?
2. Behavioral Importance: Which features best predict behavioral outcomes?
3. Multicollinearity Detection: Are some features redundant?
4. Minimal Feature Set: What's the smallest subset that maintains quality?

This analysis is critical for:
- Identifying noise features that may harm clustering
- Understanding which traits CAUSE behavioral differences
- Simplifying the persona model without losing predictive power
- Scientific interpretability of the persona system
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureImportanceResult:
    """Importance scores for a single feature."""
    feature_name: str
    category: str  # cognitive, personality, etc.
    permutation_importance: float
    behavioral_importance: float
    cluster_separation_score: float
    combined_importance: float
    rank: int

    def to_dict(self) -> Dict:
        return {
            'feature_name': self.feature_name,
            'category': self.category,
            'permutation_importance': self.permutation_importance,
            'behavioral_importance': self.behavioral_importance,
            'cluster_separation_score': self.cluster_separation_score,
            'combined_importance': self.combined_importance,
            'rank': self.rank
        }


class FeatureImportanceAnalyzer:
    """
    Analyze which features contribute most to clustering quality.

    Key Methods:
    1. Permutation Importance: Shuffle feature → measure quality drop
    2. Cluster Separation: How well does each feature separate clusters?
    3. Behavioral Importance: Which features predict outcomes?
    4. Recursive Feature Elimination: Find minimal feature set
    """

    # Feature categories for interpretability
    FEATURE_CATEGORIES = {
        'crt_score': 'cognitive',
        'need_for_cognition': 'cognitive',
        'working_memory': 'cognitive',
        'big5_extraversion': 'personality',
        'big5_agreeableness': 'personality',
        'big5_conscientiousness': 'personality',
        'big5_neuroticism': 'personality',
        'big5_openness': 'personality',
        'impulsivity_total': 'personality',
        'sensation_seeking': 'personality',
        'trust_propensity': 'personality',
        'risk_taking': 'personality',
        'state_anxiety': 'state',
        'current_stress': 'state',
        'fatigue_level': 'state',
        'phishing_self_efficacy': 'attitudes',
        'perceived_risk': 'attitudes',
        'security_attitudes': 'attitudes',
        'privacy_concern': 'attitudes',
        'phishing_knowledge': 'experience',
        'technical_expertise': 'experience',
        'prior_victimization': 'experience',
        'security_training': 'experience',
        'email_volume_numeric': 'habits',
        'link_click_tendency': 'habits',
        'social_media_usage': 'habits',
        'authority_susceptibility': 'susceptibility',
        'urgency_susceptibility': 'susceptibility',
        'scarcity_susceptibility': 'susceptibility'
    }

    def __init__(
        self,
        n_permutations: int = 30,
        random_state: int = 42
    ):
        """
        Initialize analyzer.

        Args:
            n_permutations: Number of permutation iterations for stability
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.random_state = random_state

    def analyze(
        self,
        X: np.ndarray,
        feature_names: List[str],
        outcome_data: Optional[pd.DataFrame] = None,
        outcome_cols: Optional[List[str]] = None,
        k: int = 6
    ) -> Dict:
        """
        Comprehensive feature importance analysis.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features (must match X.shape[1])
            outcome_data: DataFrame with behavioral outcomes (optional)
            outcome_cols: Names of outcome columns
            k: Number of clusters for evaluation

        Returns:
            Dictionary with importance scores, rankings, and recommendations

        Note: If PCA was applied, X.shape[1] may differ from len(feature_names).
              In this case, we truncate feature_names or handle appropriately.
        """
        results = []

        # Handle PCA case: if X has fewer columns than feature_names
        n_features_actual = X.shape[1]
        if n_features_actual < len(feature_names):
            # PCA was applied - we can only analyze PCA components, not original features
            # Create synthetic names for PCA components
            feature_names = [f'PC{i+1}' for i in range(n_features_actual)]
            pca_applied = True
        elif n_features_actual > len(feature_names):
            raise ValueError(f"X has {n_features_actual} features but only {len(feature_names)} names provided")
        else:
            pca_applied = False

        # 1. Calculate baseline clustering quality
        baseline_quality = self._calculate_clustering_quality(X, k)
        baseline_quality['pca_applied'] = pca_applied
        baseline_quality['n_features_analyzed'] = n_features_actual

        # 2. Permutation importance for each feature
        perm_importance = self._permutation_importance(X, feature_names, k, baseline_quality)

        # 3. Cluster separation scores (how well each feature separates clusters)
        separation_scores = self._cluster_separation_scores(X, feature_names, k)

        # 4. Behavioral importance (if outcome data provided)
        if outcome_data is not None and outcome_cols is not None:
            behavioral_importance = self._behavioral_importance(
                X, feature_names, outcome_data, outcome_cols
            )
        else:
            behavioral_importance = {f: 0.0 for f in feature_names}

        # 5. Combine scores and rank features
        for i, feature in enumerate(feature_names):
            perm_score = perm_importance.get(feature, 0.0)
            sep_score = separation_scores.get(feature, 0.0)
            beh_score = behavioral_importance.get(feature, 0.0)

            # Combined score (weighted average)
            # Permutation: 40%, Separation: 30%, Behavioral: 30%
            combined = 0.4 * perm_score + 0.3 * sep_score + 0.3 * beh_score

            category = self.FEATURE_CATEGORIES.get(feature, 'unknown')

            results.append(FeatureImportanceResult(
                feature_name=feature,
                category=category,
                permutation_importance=perm_score,
                behavioral_importance=beh_score,
                cluster_separation_score=sep_score,
                combined_importance=combined,
                rank=0  # Will be set after sorting
            ))

        # Sort by combined importance and assign ranks
        results.sort(key=lambda x: x.combined_importance, reverse=True)
        for rank, result in enumerate(results, 1):
            result.rank = rank

        # 6. Category-level analysis
        category_importance = self._analyze_by_category(results)

        # 7. Multicollinearity analysis
        multicollinearity = self._analyze_multicollinearity(X, feature_names)

        # 8. Find minimal feature set
        minimal_features = self._find_minimal_feature_set(X, feature_names, k, baseline_quality)

        # 9. Generate recommendations
        recommendations = self._generate_recommendations(results, minimal_features, multicollinearity)

        return {
            'feature_importance': [r.to_dict() for r in results],
            'top_features': [r.feature_name for r in results[:10]],
            'bottom_features': [r.feature_name for r in results[-5:]],
            'category_importance': category_importance,
            'multicollinearity': multicollinearity,
            'minimal_feature_set': minimal_features,
            'baseline_quality': baseline_quality,
            'recommendations': recommendations
        }

    def _calculate_clustering_quality(self, X: np.ndarray, k: int) -> Dict:
        """Calculate baseline clustering quality metrics."""
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        return {
            'silhouette': float(silhouette_score(X, labels)),
            'inertia': float(kmeans.inertia_),
            'k': k
        }

    def _permutation_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        k: int,
        baseline: Dict
    ) -> Dict[str, float]:
        """
        Calculate permutation importance for each feature.

        Importance = how much clustering quality degrades when feature is shuffled.
        """
        baseline_sil = baseline['silhouette']
        importance = {}

        for i, feature in enumerate(feature_names):
            degradations = []

            for perm in range(self.n_permutations):
                # Create permuted version of X
                X_perm = X.copy()
                np.random.seed(self.random_state + perm)
                X_perm[:, i] = np.random.permutation(X_perm[:, i])

                # Cluster and measure quality
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X_perm)

                try:
                    perm_sil = silhouette_score(X_perm, labels)
                    degradation = baseline_sil - perm_sil
                    degradations.append(degradation)
                except:
                    continue

            # Mean degradation (positive = feature was important)
            if degradations:
                mean_deg = np.mean(degradations)
                # Normalize to [0, 1] scale
                importance[feature] = max(mean_deg / (baseline_sil + 0.01), 0)
            else:
                importance[feature] = 0.0

        # Normalize so max is 1.0
        max_imp = max(importance.values()) if importance else 1.0
        if max_imp > 0:
            importance = {k: v / max_imp for k, v in importance.items()}

        return importance

    def _cluster_separation_scores(
        self,
        X: np.ndarray,
        feature_names: List[str],
        k: int
    ) -> Dict[str, float]:
        """
        Measure how well each feature separates clusters (univariate).

        Uses ANOVA F-statistic: higher F = better separation.
        """
        # First, get cluster labels
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        scores = {}
        f_values = []

        for i, feature in enumerate(feature_names):
            # Group feature values by cluster
            groups = [X[labels == c, i] for c in range(k)]

            try:
                f_stat, p_value = stats.f_oneway(*groups)
                scores[feature] = f_stat if not np.isnan(f_stat) else 0.0
                f_values.append(scores[feature])
            except:
                scores[feature] = 0.0
                f_values.append(0.0)

        # Normalize to [0, 1]
        max_f = max(f_values) if f_values else 1.0
        if max_f > 0:
            scores = {k: v / max_f for k, v in scores.items()}

        return scores

    def _behavioral_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        outcome_data: pd.DataFrame,
        outcome_cols: List[str]
    ) -> Dict[str, float]:
        """
        Calculate how well each feature predicts behavioral outcomes.

        Uses Random Forest feature importance or mutual information.
        """
        importance_scores = {f: [] for f in feature_names}

        # Create DataFrame for easier manipulation
        X_df = pd.DataFrame(X, columns=feature_names)

        for outcome in outcome_cols:
            if outcome not in outcome_data.columns:
                continue

            y = outcome_data[outcome].values

            # Skip if target has no variance
            if np.std(y) < 1e-10:
                continue

            # Use Random Forest for feature importance
            try:
                if np.unique(y).shape[0] <= 10:
                    # Classification task
                    model = RandomForestClassifier(
                        n_estimators=100, random_state=self.random_state, n_jobs=-1
                    )
                    model.fit(X, y.astype(int))
                else:
                    # Regression task
                    model = RandomForestRegressor(
                        n_estimators=100, random_state=self.random_state, n_jobs=-1
                    )
                    model.fit(X, y)

                # Get feature importances
                for i, feature in enumerate(feature_names):
                    importance_scores[feature].append(model.feature_importances_[i])
            except:
                continue

        # Average across outcomes and normalize
        final_scores = {}
        for feature in feature_names:
            if importance_scores[feature]:
                final_scores[feature] = float(np.mean(importance_scores[feature]))
            else:
                final_scores[feature] = 0.0

        # Normalize to [0, 1]
        max_score = max(final_scores.values()) if final_scores else 1.0
        if max_score > 0:
            final_scores = {k: v / max_score for k, v in final_scores.items()}

        return final_scores

    def _analyze_by_category(
        self,
        results: List[FeatureImportanceResult]
    ) -> Dict:
        """Aggregate importance by feature category."""
        category_scores = {}

        for result in results:
            cat = result.category
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(result.combined_importance)

        category_summary = {}
        for cat, scores in category_scores.items():
            category_summary[cat] = {
                'mean_importance': float(np.mean(scores)),
                'max_importance': float(np.max(scores)),
                'n_features': len(scores),
                'top_feature': next(
                    (r.feature_name for r in results if r.category == cat),
                    None
                )
            }

        # Rank categories
        sorted_cats = sorted(
            category_summary.items(),
            key=lambda x: x[1]['mean_importance'],
            reverse=True
        )
        for rank, (cat, _) in enumerate(sorted_cats, 1):
            category_summary[cat]['rank'] = rank

        return category_summary

    def _analyze_multicollinearity(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Detect multicollinearity (redundant features).

        Features with high correlation may be redundant and
        one could be removed without losing information.
        """
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)

        # Find highly correlated pairs (|r| > 0.7)
        high_corr_pairs = []
        threshold = 0.7

        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr = corr_matrix[i, j]
                if abs(corr) > threshold:
                    high_corr_pairs.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'correlation': float(corr),
                        'recommendation': f'Consider removing one of these features'
                    })

        # Calculate Variance Inflation Factor (VIF) approximation
        # VIF = 1 / (1 - R^2) where R^2 is from regressing feature on all others
        vif_scores = {}
        for i, feature in enumerate(feature_names):
            # R^2 approximation using correlation
            r_squared = np.max(np.abs(np.delete(corr_matrix[i], i))) ** 2
            vif = 1 / (1 - r_squared + 0.01)  # Add small epsilon
            vif_scores[feature] = float(vif)

        # Flag features with VIF > 5 (common threshold)
        high_vif_features = [f for f, v in vif_scores.items() if v > 5]

        return {
            'high_correlation_pairs': high_corr_pairs,
            'vif_scores': vif_scores,
            'high_vif_features': high_vif_features,
            'n_redundant_candidates': len(high_vif_features),
            'recommendation': (
                f'{len(high_vif_features)} features show signs of multicollinearity. '
                f'Consider removing: {", ".join(high_vif_features[:3])}' if high_vif_features
                else 'No significant multicollinearity detected.'
            )
        }

    def _find_minimal_feature_set(
        self,
        X: np.ndarray,
        feature_names: List[str],
        k: int,
        baseline_quality: Dict,
        quality_threshold: float = 0.90
    ) -> Dict:
        """
        Find minimal subset of features that maintains clustering quality.

        Uses forward selection: start with empty set, add features one at a time.
        Stop when quality reaches threshold of baseline.
        """
        target_silhouette = baseline_quality['silhouette'] * quality_threshold

        # Get feature importance order (use permutation importance)
        perm_importance = self._permutation_importance(X, feature_names, k, baseline_quality)
        sorted_features = sorted(
            feature_names,
            key=lambda f: perm_importance.get(f, 0),
            reverse=True
        )

        selected_features = []
        selected_indices = []
        quality_progression = []

        for feature in sorted_features:
            idx = feature_names.index(feature)
            test_indices = selected_indices + [idx]

            # Cluster with selected features only
            X_subset = X[:, test_indices]
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_subset)

            try:
                sil = silhouette_score(X_subset, labels)
            except:
                sil = -1

            quality_progression.append({
                'feature_added': feature,
                'n_features': len(test_indices),
                'silhouette': float(sil),
                'pct_of_baseline': float(sil / baseline_quality['silhouette']) * 100
            })

            selected_features.append(feature)
            selected_indices.append(idx)

            # Check if we've reached target quality
            if sil >= target_silhouette:
                break

        return {
            'minimal_features': selected_features,
            'n_features_needed': len(selected_features),
            'n_features_total': len(feature_names),
            'reduction_pct': float(1 - len(selected_features) / len(feature_names)) * 100,
            'quality_retained': float(sil / baseline_quality['silhouette']) * 100,
            'quality_threshold_used': quality_threshold * 100,
            'selection_progression': quality_progression,
            'recommendation': (
                f'Can achieve {quality_threshold*100:.0f}% of baseline quality with '
                f'{len(selected_features)}/{len(feature_names)} features '
                f'({100-len(selected_features)/len(feature_names)*100:.0f}% reduction).'
            )
        }

    def _generate_recommendations(
        self,
        results: List[FeatureImportanceResult],
        minimal_features: Dict,
        multicollinearity: Dict
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # 1. Feature removal recommendations
        low_importance = [r for r in results if r.combined_importance < 0.1]
        if low_importance:
            names = [r.feature_name for r in low_importance[:3]]
            recommendations.append(
                f"Consider removing low-importance features: {', '.join(names)}. "
                f"These contribute <10% to clustering quality."
            )

        # 2. Minimal set recommendation
        if minimal_features['reduction_pct'] > 20:
            recommendations.append(
                f"Potential simplification: {minimal_features['n_features_needed']} features "
                f"(out of {minimal_features['n_features_total']}) achieve "
                f"{minimal_features['quality_retained']:.0f}% of baseline quality."
            )

        # 3. Multicollinearity recommendation
        if multicollinearity['n_redundant_candidates'] > 0:
            recommendations.append(multicollinearity['recommendation'])

        # 4. Category-level insights
        top_categories = sorted(
            [(r.category, r.combined_importance) for r in results],
            key=lambda x: x[1],
            reverse=True
        )
        dominant_cats = list(set([c for c, _ in top_categories[:5]]))
        recommendations.append(
            f"Most influential categories: {', '.join(dominant_cats)}. "
            f"These drive cluster differentiation."
        )

        # 5. Susceptibility features
        susc_features = [r for r in results if r.category == 'susceptibility']
        if susc_features:
            avg_importance = np.mean([r.combined_importance for r in susc_features])
            if avg_importance > 0.5:
                recommendations.append(
                    "Susceptibility traits (authority, urgency, scarcity) are highly influential. "
                    "This aligns with phishing attack vectors."
                )

        return recommendations
