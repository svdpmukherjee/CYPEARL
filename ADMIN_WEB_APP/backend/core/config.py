"""
CYPEARL Configuration - Combined Phase 1 & Phase 2

This module contains all configuration for both phases:
- Phase 1 (Persona Discovery): PortalConfig
- Phase 2 (AI Simulation): Phase2Config, MODEL_REGISTRY, PROVIDER_CONFIGS
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


# =============================================================================
# PHASE 1 CONFIGURATION - Persona Discovery
# =============================================================================

@dataclass
class PortalConfig:
    """Configuration for the Phase 1 admin portal."""
    
    # Algorithm settings
    algorithms: List[str] = field(default_factory=lambda: ['kmeans', 'gmm', 'hierarchical'])
    k_min: int = 4
    k_max: int = 16
    
    # Composite score weights (must sum to 1.0)
    w_behavioral: float = 0.35      # Behavioral outcome prediction (eta-squared)
    w_silhouette: float = 0.25      # Geometric cluster separation
    w_stability: float = 0.20       # Cluster size balance/stability
    w_statistical: float = 0.20     # Calinski-Harabasz, Davies-Bouldin combined
    
    # Constraints
    min_cluster_size: int = 30      # Minimum viable cluster size
    min_cluster_pct: float = 0.03   # Minimum 3% of population per cluster
    
    # PCA settings
    pca_variance: float = 0.90      # Variance to retain
    use_pca: bool = True
    
    # Stability settings
    n_bootstrap: int = 100          # Bootstrap iterations for stability
    bootstrap_sample_ratio: float = 0.8
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path('outputs/phase1_portal'))
    
    # Random state
    random_state: int = 42
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        total = self.w_behavioral + self.w_silhouette + self.w_stability + self.w_statistical
        if abs(total - 1.0) > 0.01:
            print(f"⚠ Weights sum to {total:.2f}, normalizing to 1.0")
            self.w_behavioral /= total
            self.w_silhouette /= total
            self.w_stability /= total
            self.w_statistical /= total
        
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PHASE 2 CONFIGURATION - AI Simulation
# =============================================================================

@dataclass
class Phase2Config:
    """Configuration for Phase 2 AI persona simulation."""
    
    # Output directories
    output_dir: Path = field(default_factory=lambda: Path('outputs/phase2'))
    checkpoints_dir: Path = field(default_factory=lambda: Path('outputs/phase2/checkpoints'))
    results_dir: Path = field(default_factory=lambda: Path('outputs/phase2/results'))
    
    # Execution settings
    max_concurrent_requests: int = 5
    default_temperature: float = 0.3
    default_max_tokens: int = 500
    trials_per_condition: int = 30
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Fidelity thresholds
    fidelity_threshold: float = 0.85
    acceptance_range: List[float] = field(default_factory=lambda: [0.80, 0.90])
    
    # Checkpointing
    checkpoint_interval: int = 10  # Save checkpoint every N trials
    
    # Rate limiting
    default_requests_per_minute: int = 60
    
    # =========================================================================
    # PHASE 1 ATTRIBUTES (for backward compatibility)
    # These are duplicated from PortalConfig so code using generic 'config'
    # doesn't break when it expects Phase 1 attributes
    # =========================================================================
    
    # Algorithm settings (from Phase 1)
    algorithms: List[str] = field(default_factory=lambda: ['kmeans', 'gmm', 'hierarchical'])
    k_min: int = 4
    k_max: int = 16
    
    # Composite score weights (must sum to 1.0)
    w_behavioral: float = 0.35
    w_silhouette: float = 0.25
    w_stability: float = 0.20
    w_statistical: float = 0.20
    
    # Constraints
    min_cluster_size: int = 30
    min_cluster_pct: float = 0.03
    
    # PCA settings
    pca_variance: float = 0.90
    use_pca: bool = True
    
    # Stability settings
    n_bootstrap: int = 100
    bootstrap_sample_ratio: float = 0.8
    
    # Random state
    random_state: int = 42
    
    def __post_init__(self):
        """Create output directories."""
        self.output_dir = Path(self.output_dir)
        self.checkpoints_dir = Path(self.checkpoints_dir)
        self.results_dir = Path(self.results_dir)
        
        for dir_path in [self.output_dir, self.checkpoints_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL REGISTRY - All supported LLM models (via OpenRouter)
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # CLAUDE MODELS (via OpenRouter)
    # =========================================================================
    "claude-sonnet-4": {
        "display_name": "Claude Sonnet 4",
        "provider": "openrouter",
        "provider_model_id": "anthropic/claude-sonnet-4",
        "openrouter_model_id": "anthropic/claude-sonnet-4",
        "tier": "frontier",
        "max_tokens": 8192,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "requests_per_minute": 50,
        "supports_system_prompt": True,
    },
    "claude-3-5-sonnet": {
        "display_name": "Claude 3.5 Sonnet",
        "provider": "openrouter",
        "provider_model_id": "anthropic/claude-3.5-sonnet",
        "openrouter_model_id": "anthropic/claude-3.5-sonnet",
        "tier": "frontier",
        "max_tokens": 8192,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "requests_per_minute": 50,
        "supports_system_prompt": True,
    },
    "claude-3-5-haiku": {
        "display_name": "Claude 3.5 Haiku",
        "provider": "openrouter",
        "provider_model_id": "anthropic/claude-3.5-haiku",
        "openrouter_model_id": "anthropic/claude-3.5-haiku",
        "tier": "mid_tier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.0008,
        "cost_per_1k_output": 0.004,
        "requests_per_minute": 100,
        "supports_system_prompt": True,
    },
    "claude-3-opus": {
        "display_name": "Claude 3 Opus",
        "provider": "openrouter",
        "provider_model_id": "anthropic/claude-3-opus",
        "openrouter_model_id": "anthropic/claude-3-opus",
        "tier": "frontier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
        "requests_per_minute": 50,
        "supports_system_prompt": True,
    },
    "claude-3-sonnet": {
        "display_name": "Claude 3 Sonnet",
        "provider": "openrouter",
        "provider_model_id": "anthropic/claude-3-sonnet",
        "openrouter_model_id": "anthropic/claude-3-sonnet",
        "tier": "mid_tier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "requests_per_minute": 50,
        "supports_system_prompt": True,
    },
    "claude-3-haiku": {
        "display_name": "Claude 3 Haiku",
        "provider": "openrouter",
        "provider_model_id": "anthropic/claude-3-haiku",
        "openrouter_model_id": "anthropic/claude-3-haiku",
        "tier": "budget",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.00125,
        "requests_per_minute": 100,
        "supports_system_prompt": True,
    },

    # =========================================================================
    # GPT MODELS (via OpenRouter)
    # =========================================================================
    "gpt-4o": {
        "display_name": "GPT-4o",
        "provider": "openrouter",
        "provider_model_id": "openai/gpt-4o",
        "openrouter_model_id": "openai/gpt-4o",
        "tier": "frontier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },
    "gpt-4o-mini": {
        "display_name": "GPT-4o Mini",
        "provider": "openrouter",
        "provider_model_id": "openai/gpt-4o-mini",
        "openrouter_model_id": "openai/gpt-4o-mini",
        "tier": "mid_tier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "requests_per_minute": 100,
        "supports_system_prompt": True,
    },
    "gpt-4-turbo": {
        "display_name": "GPT-4 Turbo",
        "provider": "openrouter",
        "provider_model_id": "openai/gpt-4-turbo",
        "openrouter_model_id": "openai/gpt-4-turbo",
        "tier": "frontier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },
    "gpt-4": {
        "display_name": "GPT-4",
        "provider": "openrouter",
        "provider_model_id": "openai/gpt-4",
        "openrouter_model_id": "openai/gpt-4",
        "tier": "frontier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },

    # =========================================================================
    # MISTRAL MODELS (via OpenRouter)
    # =========================================================================
    "mistral-large": {
        "display_name": "Mistral Large",
        "provider": "openrouter",
        "provider_model_id": "mistralai/mistral-large",
        "openrouter_model_id": "mistralai/mistral-large",
        "tier": "frontier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.002,
        "cost_per_1k_output": 0.006,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },
    "mistral-medium": {
        "display_name": "Mistral Medium",
        "provider": "openrouter",
        "provider_model_id": "mistralai/mistral-medium",
        "openrouter_model_id": "mistralai/mistral-medium",
        "tier": "mid_tier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.0027,
        "cost_per_1k_output": 0.0081,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },
    "mistral-small": {
        "display_name": "Mistral Small",
        "provider": "openrouter",
        "provider_model_id": "mistralai/mistral-small",
        "openrouter_model_id": "mistralai/mistral-small",
        "tier": "mid_tier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.003,
        "requests_per_minute": 100,
        "supports_system_prompt": True,
    },
    "mistral-7b": {
        "display_name": "Mistral 7B Instruct",
        "provider": "openrouter",
        "provider_model_id": "mistralai/mistral-7b-instruct",
        "openrouter_model_id": "mistralai/mistral-7b-instruct",
        "tier": "budget",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.00007,
        "cost_per_1k_output": 0.00007,
        "requests_per_minute": 100,
        "supports_system_prompt": True,
    },
    "mixtral-8x7b": {
        "display_name": "Mixtral 8x7B Instruct",
        "provider": "openrouter",
        "provider_model_id": "mistralai/mixtral-8x7b-instruct",
        "openrouter_model_id": "mistralai/mixtral-8x7b-instruct",
        "tier": "open_source",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.00024,
        "cost_per_1k_output": 0.00024,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },
    "mixtral-8x22b": {
        "display_name": "Mixtral 8x22B Instruct",
        "provider": "openrouter",
        "provider_model_id": "mistralai/mixtral-8x22b-instruct",
        "openrouter_model_id": "mistralai/mixtral-8x22b-instruct",
        "tier": "mid_tier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.00065,
        "cost_per_1k_output": 0.00065,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },

    # =========================================================================
    # AMAZON NOVA MODELS (via OpenRouter)
    # =========================================================================
    "nova-pro": {
        "display_name": "Amazon Nova Pro",
        "provider": "openrouter",
        "provider_model_id": "amazon/nova-pro-v1",
        "openrouter_model_id": "amazon/nova-pro-v1",
        "tier": "frontier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.0008,
        "cost_per_1k_output": 0.0032,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },
    "nova-lite": {
        "display_name": "Amazon Nova Lite",
        "provider": "openrouter",
        "provider_model_id": "amazon/nova-lite-v1",
        "openrouter_model_id": "amazon/nova-lite-v1",
        "tier": "mid_tier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.00006,
        "cost_per_1k_output": 0.00024,
        "requests_per_minute": 100,
        "supports_system_prompt": True,
    },
    "nova-micro": {
        "display_name": "Amazon Nova Micro",
        "provider": "openrouter",
        "provider_model_id": "amazon/nova-micro-v1",
        "openrouter_model_id": "amazon/nova-micro-v1",
        "tier": "budget",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.000035,
        "cost_per_1k_output": 0.00014,
        "requests_per_minute": 100,
        "supports_system_prompt": True,
    },

    # =========================================================================
    # LLAMA 4 MODELS (Meta via OpenRouter)
    # =========================================================================
    "llama-4-maverick": {
        "display_name": "Llama 4 Maverick",
        "provider": "openrouter",
        "provider_model_id": "meta-llama/llama-4-maverick",
        "openrouter_model_id": "meta-llama/llama-4-maverick",
        "tier": "frontier",
        "max_tokens": 8192,
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },
    "llama-4-scout": {
        "display_name": "Llama 4 Scout",
        "provider": "openrouter",
        "provider_model_id": "meta-llama/llama-4-scout",
        "openrouter_model_id": "meta-llama/llama-4-scout",
        "tier": "mid_tier",
        "max_tokens": 8192,
        "cost_per_1k_input": 0.00008,
        "cost_per_1k_output": 0.0003,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },

    # =========================================================================
    # LLAMA 3.x MODELS (Meta via OpenRouter)
    # =========================================================================
    "llama-3.3-70b": {
        "display_name": "Llama 3.3 70B Instruct",
        "provider": "openrouter",
        "provider_model_id": "meta-llama/llama-3.3-70b-instruct",
        "openrouter_model_id": "meta-llama/llama-3.3-70b-instruct",
        "tier": "open_source",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.0001,
        "cost_per_1k_output": 0.00032,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },
    "llama-3.1-405b": {
        "display_name": "Llama 3.1 405B Instruct",
        "provider": "openrouter",
        "provider_model_id": "meta-llama/llama-3.1-405b-instruct",
        "openrouter_model_id": "meta-llama/llama-3.1-405b-instruct",
        "tier": "frontier",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.0035,
        "cost_per_1k_output": 0.0035,
        "requests_per_minute": 30,
        "supports_system_prompt": True,
    },
    "llama-3.1-70b": {
        "display_name": "Llama 3.1 70B Instruct",
        "provider": "openrouter",
        "provider_model_id": "meta-llama/llama-3.1-70b-instruct",
        "openrouter_model_id": "meta-llama/llama-3.1-70b-instruct",
        "tier": "open_source",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.0004,
        "cost_per_1k_output": 0.0004,
        "requests_per_minute": 60,
        "supports_system_prompt": True,
    },

    # =========================================================================
    # LOCAL MODELS (Ollama) - Optional fallback
    # =========================================================================
    "local-llama3": {
        "display_name": "Llama 3 (Local)",
        "provider": "local",
        "provider_model_id": "llama3",
        "tier": "budget",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "requests_per_minute": 30,
        "supports_system_prompt": True,
    },
    "local-mistral": {
        "display_name": "Mistral (Local)",
        "provider": "local",
        "provider_model_id": "mistral",
        "tier": "budget",
        "max_tokens": 4096,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "requests_per_minute": 30,
        "supports_system_prompt": True,
    },
}


# =============================================================================
# PROVIDER CONFIGURATIONS
# =============================================================================

PROVIDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "anthropic": {
        "display_name": "Anthropic",
        "auth_type": "api_key",
        "base_url": "https://api.anthropic.com",
        "requires_region": False,
        "env_var": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "display_name": "OpenAI",
        "auth_type": "api_key",
        "base_url": "https://api.openai.com/v1",
        "requires_region": False,
        "env_var": "OPENAI_API_KEY",
    },
    "aws_bedrock": {
        "display_name": "AWS Bedrock",
        "auth_type": "aws_credentials",
        "requires_region": True,
        "default_region": "us-east-1",
        "env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    },
    "together_ai": {
        "display_name": "Together AI",
        "auth_type": "api_key",
        "base_url": "https://api.together.xyz/v1",
        "requires_region": False,
        "env_var": "TOGETHER_API_KEY",
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "auth_type": "api_key",
        "base_url": "https://openrouter.ai/api/v1",
        "requires_region": False,
        "env_var": "OPENROUTER_API_KEY",
    },
    "local": {
        "display_name": "Local (Ollama)",
        "auth_type": "none",
        "base_url": "http://localhost:11434",
        "requires_region": False,
    },
}


# =============================================================================
# GLOBAL CONFIGURATION INSTANCES
# =============================================================================

# Phase 1 configuration instance
portal_config = PortalConfig()

# Phase 2 configuration instance
phase2_config = Phase2Config()

# Unified config object that includes both (for backwards compatibility)
# When code imports 'config', they get the phase2_config which has checkpoints_dir
config = phase2_config