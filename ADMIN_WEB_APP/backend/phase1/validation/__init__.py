"""
Phase 1 Clustering Validation Module

Scientific validation framework for clustering quality assessment.
Implements industry-standard validation methods for defensible persona discovery.

Modules:
- gap_statistic: Data-driven optimal K selection
- feature_importance: Identify which traits drive cluster separation
- cross_validation: Prevent overfitting via held-out validation
- prediction_error: Measure predictive power of clusters
- consensus_clustering: Robust cluster discovery via ensemble
- algorithm_comparison: Statistical comparison between algorithms
- soft_assignments: Handle cluster boundary uncertainty
- validator: Unified validation orchestrator
"""

from .gap_statistic import GapStatisticAnalyzer
from .feature_importance import FeatureImportanceAnalyzer
from .cross_validation import ClusteringCrossValidator
from .prediction_error import PredictionErrorAnalyzer
from .consensus_clustering import ConsensusClusteringAnalyzer
from .algorithm_comparison import AlgorithmComparisonAnalyzer
from .soft_assignments import SoftAssignmentAnalyzer
from .validator import ClusteringValidator

__all__ = [
    'GapStatisticAnalyzer',
    'FeatureImportanceAnalyzer',
    'ClusteringCrossValidator',
    'PredictionErrorAnalyzer',
    'ConsensusClusteringAnalyzer',
    'AlgorithmComparisonAnalyzer',
    'SoftAssignmentAnalyzer',
    'ClusteringValidator'
]
