"""
CYPEARL Core Package

This package contains shared components for Phase 1 and Phase 2:
- Configuration (config.py)
- Data schemas (schemas.py) 
- Data loading (data_loader.py)
- Preprocessing (preprocessor.py)
"""

# Configuration exports
from .config import (
    # Phase 1 Configuration
    PortalConfig,
    portal_config,
    
    # Phase 2 Configuration
    Phase2Config,
    phase2_config,
    
    # Model & Provider registries
    MODEL_REGISTRY,
    PROVIDER_CONFIGS,
    
    # Unified config (defaults to phase2_config)
    config,
)

# Schema exports (if schemas.py exists)
try:
    from .schemas import (
        # Enums
        RiskLevel,
        CognitiveStyle,
        PromptConfiguration,
        ModelTier,
        ProviderType,
        ExperimentStatus,
        ActionType,
        ConfidenceLevel,
        DecisionSpeed,
        
        # Persona schemas
        TraitZScores,
        BehavioralStatistics,
        EmailInteractionEffects,
        ProcessMetrics,
        BoundaryCondition,
        ReasoningExample,
        ExpertValidation,
        Persona,
        
        # Email schemas
        EmailContent,
        EmailStimulus,
        
        # Model schemas
        ModelConfig,
        ProviderConfig,
        
        # Experiment schemas
        ExperimentConfig,
        SimulationTrial,
        
        # Analysis schemas
        FidelityMetrics,
        ModelComparisonResult,
        BoundaryConditionResult,
        CostPerformancePoint,
        
        # API schemas
        ImportPersonasRequest,
        CreateExperimentRequest,
        RunExperimentRequest,
        ProviderSetupRequest,
    )
except ImportError:
    # schemas.py may not exist or may have different exports
    pass

# Data loader exports (if data_loader.py exists)
try:
    from .data_loader import DataLoader
except ImportError:
    pass

# Preprocessor exports (if preprocessor.py exists)
try:
    from .preprocessor import DataPreprocessor
except ImportError:
    pass

__all__ = [
    # Configuration
    'PortalConfig',
    'portal_config',
    'Phase2Config', 
    'phase2_config',
    'MODEL_REGISTRY',
    'PROVIDER_CONFIGS',
    'config',
    
    # Data
    'DataLoader',
    'DataPreprocessor',
]