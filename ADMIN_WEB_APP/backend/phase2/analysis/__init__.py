"""
CYPEARL Phase 2 - Analysis Package

This package handles fidelity analysis and model comparison:
- FidelityAnalyzer: Calculates behavioral fidelity metrics
- Boundary condition detection
- Model comparison and ranking
"""

from .fidelity_analyzer import (
    FidelityAnalyzer,
    ConditionResult,
    EmailLevelResult
)

__all__ = [
    'FidelityAnalyzer',
    'ConditionResult',
    'EmailLevelResult'
]