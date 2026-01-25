"""
Unified Clustering Validator

Orchestrates all validation modules to provide comprehensive,
scientifically rigorous clustering validation.

This is the main entry point for Phase 1 clustering validation.
It runs all validation analyses and produces a single comprehensive
report with clear recommendations.

Validation Levels:
1. BASIC: Gap statistic + cross-validation (fast, essential)
2. STANDARD: Basic + feature importance + prediction error (recommended)
3. COMPREHENSIVE: Standard + consensus + algorithm comparison (thorough)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
import time
import json

from .gap_statistic import GapStatisticAnalyzer
from .feature_importance import FeatureImportanceAnalyzer
from .cross_validation import ClusteringCrossValidator
from .prediction_error import PredictionErrorAnalyzer
from .consensus_clustering import ConsensusClusteringAnalyzer
from .algorithm_comparison import AlgorithmComparisonAnalyzer
from .soft_assignments import SoftAssignmentAnalyzer


@dataclass
class ValidationReport:
    """Complete validation report."""
    validation_level: str
    overall_score: float
    overall_assessment: str
    passes_validation: bool
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    detailed_results: Dict

    def to_dict(self) -> Dict:
        return {
            'validation_level': self.validation_level,
            'overall_score': self.overall_score,
            'overall_assessment': self.overall_assessment,
            'passes_validation': self.passes_validation,
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'detailed_results': self.detailed_results
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class ClusteringValidator:
    """
    Comprehensive clustering validation orchestrator.

    Combines all validation modules to provide:
    1. Scientific rigor for academic defensibility
    2. Practical insights for improving clustering
    3. Clear pass/fail criteria for pipeline progression
    """

    # Validation thresholds
    THRESHOLDS = {
        'min_cv_test_silhouette': 0.1,  # Minimum test silhouette
        'max_generalization_gap': 0.15,  # Maximum train-test gap
        'min_prediction_r2': 0.05,  # Minimum predictive R²
        'max_boundary_case_pct': 0.30,  # Maximum % of uncertain participants
        'min_consensus_stability': 0.5,  # Minimum consensus stability
    }

    def __init__(
        self,
        n_bootstrap: int = 50,
        n_cv_folds: int = 5,
        n_consensus_iterations: int = 50,
        random_state: int = 42
    ):
        """
        Initialize validator.

        Args:
            n_bootstrap: Bootstrap iterations for various analyses
            n_cv_folds: Number of cross-validation folds
            n_consensus_iterations: Iterations for consensus clustering
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.n_cv_folds = n_cv_folds
        self.n_consensus_iterations = n_consensus_iterations
        self.random_state = random_state

        # Initialize analyzers
        self.gap_analyzer = GapStatisticAnalyzer(
            n_references=20, random_state=random_state
        )
        self.feature_analyzer = FeatureImportanceAnalyzer(
            n_permutations=20, random_state=random_state
        )
        self.cv_analyzer = ClusteringCrossValidator(
            n_folds=n_cv_folds, random_state=random_state
        )
        self.prediction_analyzer = PredictionErrorAnalyzer(
            random_state=random_state
        )
        self.consensus_analyzer = ConsensusClusteringAnalyzer(
            n_iterations=n_consensus_iterations, random_state=random_state
        )
        self.algorithm_analyzer = AlgorithmComparisonAnalyzer(
            n_bootstrap=n_bootstrap, random_state=random_state
        )
        self.soft_analyzer = SoftAssignmentAnalyzer(
            random_state=random_state
        )

    def validate(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        k: int,
        feature_names: List[str],
        outcome_data: Optional[pd.DataFrame] = None,
        outcome_cols: Optional[List[str]] = None,
        level: Literal['basic', 'standard', 'comprehensive'] = 'standard'
    ) -> ValidationReport:
        """
        Run clustering validation at specified level.

        Args:
            X: Feature matrix (n_samples, n_features)
            labels: Cluster assignments (from best algorithm/K)
            k: Number of clusters
            feature_names: Names of features used in clustering
            outcome_data: Behavioral outcomes DataFrame
            outcome_cols: Names of outcome columns
            level: Validation level ('basic', 'standard', 'comprehensive')

        Returns:
            ValidationReport with comprehensive results
        """
        print(f"[VALIDATOR] Starting {level.upper()} validation...")
        start_time = time.time()

        results = {}
        issues = []
        warnings = []
        recommendations = []

        # =====================================================================
        # BASIC VALIDATION (always run)
        # =====================================================================

        # 1. Gap Statistic for optimal K confirmation
        print("[VALIDATOR] Running Gap Statistic analysis...")
        gap_results = self.gap_analyzer.analyze(X, k_min=max(2, k-3), k_max=k+3)
        results['gap_statistic'] = gap_results

        if gap_results['optimal_k'] != k:
            warnings.append(
                f"Gap statistic suggests K={gap_results['optimal_k']}, but you chose K={k}. "
                f"Confidence: {gap_results['confidence']['level']}"
            )

        # 2. Cross-Validation
        print("[VALIDATOR] Running Cross-Validation...")
        cv_results = self.cv_analyzer.cross_validate(
            X, outcome_data, outcome_cols, k, algorithm='kmeans'
        )
        results['cross_validation'] = cv_results

        # Check CV metrics
        test_sil = cv_results['summary']['test_silhouette']['mean']
        if test_sil and test_sil < self.THRESHOLDS['min_cv_test_silhouette']:
            issues.append(
                f"Test silhouette ({test_sil:.3f}) below threshold "
                f"({self.THRESHOLDS['min_cv_test_silhouette']}). "
                f"Clusters may not generalize."
            )

        gen_gap = cv_results['generalization']['mean_gap']
        if gen_gap and gen_gap > self.THRESHOLDS['max_generalization_gap']:
            issues.append(
                f"Generalization gap ({gen_gap:.3f}) exceeds threshold "
                f"({self.THRESHOLDS['max_generalization_gap']}). "
                f"Overfitting detected."
            )

        # =====================================================================
        # STANDARD VALIDATION
        # =====================================================================

        if level in ['standard', 'comprehensive']:
            # 3. Feature Importance
            print("[VALIDATOR] Running Feature Importance analysis...")
            feature_results = self.feature_analyzer.analyze(
                X, feature_names, outcome_data, outcome_cols, k
            )
            results['feature_importance'] = feature_results

            # Check if many features have low importance
            low_importance = [f for f in feature_results['feature_importance']
                            if f['combined_importance'] < 0.1]
            if len(low_importance) > len(feature_names) * 0.3:
                warnings.append(
                    f"{len(low_importance)} features have low importance. "
                    f"Consider feature reduction."
                )

            recommendations.extend(feature_results.get('recommendations', []))

            # 4. Prediction Error
            print("[VALIDATOR] Running Prediction Error analysis...")
            pred_results = self.prediction_analyzer.analyze(
                X, labels, outcome_data, outcome_cols
            )
            results['prediction_error'] = pred_results

            # Check predictive power
            mean_r2 = pred_results['aggregated'].get('mean_r_squared', 0)
            if mean_r2 < self.THRESHOLDS['min_prediction_r2']:
                issues.append(
                    f"Predictive R² ({mean_r2:.3f}) below threshold "
                    f"({self.THRESHOLDS['min_prediction_r2']}). "
                    f"Clusters may not be useful for prediction."
                )

            if not pred_results['baseline_comparison']['summary']['beats_all_baselines']:
                warnings.append(
                    "Clustering doesn't beat all naive baselines. "
                    "Consider whether clustering adds value."
                )

            # 5. Soft Assignments
            print("[VALIDATOR] Running Soft Assignment analysis...")
            soft_results = self.soft_analyzer.analyze(X, k, algorithm='gmm')
            results['soft_assignments'] = {
                'uncertainty_distribution': soft_results['uncertainty_distribution'],
                'cluster_purity': soft_results['cluster_purity'],
                'handling_strategy': soft_results['handling_strategy'],
                'summary': soft_results['summary']
            }

            high_uncertainty_pct = soft_results['uncertainty_distribution']['uncertainty_percentages']['high_pct']
            if high_uncertainty_pct > self.THRESHOLDS['max_boundary_case_pct'] * 100:
                warnings.append(
                    f"{high_uncertainty_pct:.1f}% of participants are boundary cases. "
                    f"Consider soft assignment handling."
                )

        # =====================================================================
        # COMPREHENSIVE VALIDATION
        # =====================================================================

        if level == 'comprehensive':
            # 6. Consensus Clustering
            print("[VALIDATOR] Running Consensus Clustering analysis...")
            consensus_results = self.consensus_analyzer.analyze(
                X, k, algorithms=['kmeans', 'gmm']
            )
            results['consensus_clustering'] = {
                'stability_metrics': consensus_results['stability_metrics'],
                'core_members': consensus_results['core_members'],
                'boundary_cases': consensus_results['boundary_cases'],
                'quality_comparison': consensus_results['quality_comparison'],
                'recommendation': consensus_results['recommendation']
            }

            stability = consensus_results['stability_metrics']['stability_score']
            if stability < self.THRESHOLDS['min_consensus_stability']:
                warnings.append(
                    f"Consensus stability ({stability:.2f}) is low. "
                    f"Clusters may not be robust."
                )

            # 7. Algorithm Comparison
            print("[VALIDATOR] Running Algorithm Comparison...")
            algo_results = self.algorithm_analyzer.compare(
                X, ['kmeans', 'gmm', 'hierarchical'], k,
                outcome_data, outcome_cols, metric='composite'
            )
            results['algorithm_comparison'] = algo_results

            if algo_results['best_algorithm']['confidence'] == 'low':
                warnings.append(
                    "No algorithm is significantly better than others. "
                    "Choice may be arbitrary."
                )

        # =====================================================================
        # AGGREGATE RESULTS
        # =====================================================================

        elapsed = time.time() - start_time
        print(f"[VALIDATOR] Validation completed in {elapsed:.2f}s")

        # Calculate overall score
        overall_score, score_breakdown = self._calculate_overall_score(results, level)

        # Determine pass/fail
        passes = len(issues) == 0
        assessment = self._generate_assessment(overall_score, issues, warnings)

        # Add timing info
        results['timing'] = {
            'elapsed_seconds': elapsed,
            'validation_level': level
        }
        results['score_breakdown'] = score_breakdown

        return ValidationReport(
            validation_level=level,
            overall_score=overall_score,
            overall_assessment=assessment,
            passes_validation=passes,
            critical_issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            detailed_results=results
        )

    def _calculate_overall_score(
        self,
        results: Dict,
        level: str
    ) -> tuple:
        """Calculate overall validation score (0-1)."""
        scores = {}

        # Cross-validation score
        cv = results.get('cross_validation', {})
        test_sil = cv.get('summary', {}).get('test_silhouette', {}).get('mean')
        if test_sil is not None:
            scores['cv_silhouette'] = (test_sil + 1) / 2  # Normalize to [0,1]

        gen_gap = cv.get('generalization', {}).get('mean_gap')
        if gen_gap is not None:
            scores['generalization'] = max(0, 1 - gen_gap * 5)  # Penalize gaps

        # Gap statistic confidence
        gap = results.get('gap_statistic', {})
        gap_conf = gap.get('confidence', {}).get('overall', 0.5)
        scores['gap_confidence'] = gap_conf

        if level in ['standard', 'comprehensive']:
            # Prediction score
            pred = results.get('prediction_error', {})
            r2 = pred.get('aggregated', {}).get('mean_r_squared', 0)
            scores['prediction_r2'] = min(r2 / 0.20, 1.0)  # Cap at R²=0.20

            # Soft assignment score (inverse of uncertainty)
            soft = results.get('soft_assignments', {})
            high_unc = soft.get('uncertainty_distribution', {}).get(
                'uncertainty_percentages', {}
            ).get('high_pct', 30)
            scores['cluster_purity'] = max(0, 1 - high_unc / 50)

        if level == 'comprehensive':
            # Consensus stability
            consensus = results.get('consensus_clustering', {})
            stability = consensus.get('stability_metrics', {}).get('stability_score', 0.5)
            scores['consensus_stability'] = stability

        # Weighted average
        if scores:
            overall = sum(scores.values()) / len(scores)
        else:
            overall = 0.0

        return float(overall), scores

    def _generate_assessment(
        self,
        score: float,
        issues: List[str],
        warnings: List[str]
    ) -> str:
        """Generate human-readable assessment."""
        if len(issues) > 0:
            return (
                f"VALIDATION FAILED: {len(issues)} critical issue(s) detected. "
                f"Overall score: {score:.2f}. "
                f"Address critical issues before proceeding to LLM conditioning."
            )
        elif len(warnings) > 2:
            return (
                f"VALIDATION PASSED WITH CONCERNS: {len(warnings)} warning(s). "
                f"Overall score: {score:.2f}. "
                f"Clustering is usable but could be improved."
            )
        elif score >= 0.7:
            return (
                f"VALIDATION PASSED: Strong clustering quality. "
                f"Overall score: {score:.2f}. "
                f"Proceed to LLM conditioning with confidence."
            )
        else:
            return (
                f"VALIDATION PASSED: Acceptable clustering quality. "
                f"Overall score: {score:.2f}. "
                f"Consider addressing warnings for improved results."
            )

    def quick_validate(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        k: int,
        outcome_data: Optional[pd.DataFrame] = None,
        outcome_cols: Optional[List[str]] = None
    ) -> Dict:
        """
        Quick validation check (minimal analysis).

        Returns essential metrics without full analysis.
        Use for rapid iteration during development.
        """
        from sklearn.metrics import silhouette_score

        results = {
            'k': k,
            'n_samples': len(labels),
            'n_clusters_actual': len(np.unique(labels))
        }

        # Silhouette
        results['silhouette'] = float(silhouette_score(X, labels))

        # Eta-squared (if outcomes provided)
        if outcome_data is not None and outcome_cols is not None:
            eta_values = []
            for outcome in outcome_cols:
                if outcome in outcome_data.columns:
                    values = outcome_data[outcome].values
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
                        eta_values.append(between_ss / total_ss)
            results['eta_squared_mean'] = float(np.mean(eta_values)) if eta_values else 0.0

        # Quick pass/fail
        passes = results['silhouette'] > 0.1
        if 'eta_squared_mean' in results:
            passes = passes and results['eta_squared_mean'] > 0.05

        results['quick_pass'] = passes
        results['recommendation'] = (
            "Quick check PASSED - proceed to full validation"
            if passes else
            "Quick check FAILED - review clustering before full validation"
        )

        return results

    def validate_for_llm_readiness(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        k: int,
        feature_names: List[str],
        outcome_data: pd.DataFrame,
        outcome_cols: List[str]
    ) -> Dict:
        """
        Specialized validation for LLM persona conditioning readiness.

        Focuses on metrics most relevant to whether clusters will
        produce effective persona prompts.
        """
        print("[VALIDATOR] Running LLM Readiness validation...")

        readiness = {
            'ready_for_llm': True,
            'blocking_issues': [],
            'concerns': [],
            'strengths': []
        }

        # 1. Prediction power (can cluster predict behavior?)
        pred_results = self.prediction_analyzer.analyze(
            X, labels, outcome_data, outcome_cols
        )

        r2 = pred_results['aggregated'].get('mean_r_squared', 0)
        if r2 < 0.05:
            readiness['blocking_issues'].append(
                f"Predictive power too low (R²={r2:.3f}). "
                f"LLM personas won't accurately reflect behavior."
            )
            readiness['ready_for_llm'] = False
        elif r2 < 0.10:
            readiness['concerns'].append(
                f"Moderate predictive power (R²={r2:.3f}). "
                f"Personas will capture trends but not individuals."
            )
        else:
            readiness['strengths'].append(
                f"Good predictive power (R²={r2:.3f})."
            )

        # 2. Cluster distinctiveness
        soft_results = self.soft_analyzer.analyze(X, k, algorithm='gmm')
        high_unc = soft_results['uncertainty_distribution']['uncertainty_percentages']['high_pct']

        if high_unc > 40:
            readiness['blocking_issues'].append(
                f"Too many boundary cases ({high_unc:.1f}%). "
                f"Personas won't be distinct."
            )
            readiness['ready_for_llm'] = False
        elif high_unc > 25:
            readiness['concerns'].append(
                f"{high_unc:.1f}% boundary cases. "
                f"Consider soft persona assignments for these participants."
            )
        else:
            readiness['strengths'].append(
                f"Clear cluster separation ({100-high_unc:.1f}% confident assignments)."
            )

        # 3. Behavioral differentiation
        outcome = 'phishing_click_rate' if 'phishing_click_rate' in outcome_cols else outcome_cols[0]
        df = pd.DataFrame({'cluster': labels, 'outcome': outcome_data[outcome].values})
        cluster_means = df.groupby('cluster')['outcome'].mean()

        mean_range = cluster_means.max() - cluster_means.min()
        if mean_range < 0.10:
            readiness['concerns'].append(
                f"Small behavioral range across clusters ({mean_range:.2f}). "
                f"Personas may be hard to distinguish behaviorally."
            )
        else:
            readiness['strengths'].append(
                f"Good behavioral differentiation (range={mean_range:.2f})."
            )

        # 4. Cross-validation stability
        cv_results = self.cv_analyzer.cross_validate(
            X, outcome_data, outcome_cols, k, algorithm='kmeans'
        )
        cv_quality = cv_results.get('cv_quality_score', {}).get('overall', 0)

        if cv_quality < 0.5:
            readiness['concerns'].append(
                f"CV quality is moderate ({cv_quality:.2f}). "
                f"Clusters may not generalize perfectly."
            )
        else:
            readiness['strengths'].append(
                f"Good CV performance ({cv_quality:.2f})."
            )

        # Summary
        if readiness['ready_for_llm']:
            if not readiness['concerns']:
                readiness['summary'] = (
                    "READY: Clustering is well-suited for LLM persona conditioning. "
                    "Proceed with confidence."
                )
            else:
                readiness['summary'] = (
                    "CONDITIONALLY READY: Clustering can be used for LLM personas "
                    f"but address {len(readiness['concerns'])} concern(s) for best results."
                )
        else:
            readiness['summary'] = (
                f"NOT READY: {len(readiness['blocking_issues'])} blocking issue(s) "
                "must be resolved before LLM conditioning will be effective."
            )

        readiness['detailed_results'] = {
            'prediction': pred_results['aggregated'],
            'soft_assignments': soft_results['uncertainty_distribution'],
            'cv': cv_results['summary']
        }

        return readiness
