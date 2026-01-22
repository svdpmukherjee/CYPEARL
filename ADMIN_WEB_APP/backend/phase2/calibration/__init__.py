"""
CYPEARL Phase 2 - Prompt Calibration Module

This module implements held-out validation for prompt configurations:
1. Split Phase 1 behavioral data into train/test sets
2. Build prompts from training data
3. Test LLM predictions against held-out human responses
4. Use LLM self-reflection to improve prompts

Usage:
    from calibration import CalibrationEngine, DataSplitter

    splitter = DataSplitter(persona_data, split_ratio=0.8)
    train_data, test_data = splitter.split()

    engine = CalibrationEngine(router)
    results = await engine.run_calibration(
        persona, train_data, test_data,
        prompt_config='stats', model_id='gpt-4o'
    )

    if results.accuracy < 0.8:
        suggestions = await engine.self_reflect(results)
"""

from .data_splitter import DataSplitter, SplitResult, generate_synthetic_trials
from .calibration_engine import CalibrationEngine, CalibrationResult, CalibrationTrial
from .self_reflection import SelfReflectionEngine, PromptSuggestion
from .icl_builder import ICLExampleBuilder, ICLExample, build_icl_block
from .pattern_extractor import DecisionPatternExtractor, DecisionPattern, PatternSummary, extract_patterns_for_prompt

__all__ = [
    'DataSplitter',
    'SplitResult',
    'generate_synthetic_trials',
    'CalibrationEngine',
    'CalibrationResult',
    'CalibrationTrial',
    'SelfReflectionEngine',
    'PromptSuggestion',
    # ICL support
    'ICLExampleBuilder',
    'ICLExample',
    'build_icl_block',
    'DecisionPatternExtractor',
    'DecisionPattern',
    'PatternSummary',
    'extract_patterns_for_prompt',
]
