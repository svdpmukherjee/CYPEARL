"""
CYPEARL Phase 2 - Data Schemas
Pydantic models for all Phase 2 data structures.

Enhanced with:
- Cohen's d effect sizes
- Effect preservation details
- Deployment recommendations
- Model versioning metadata
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


# =============================================================================
# ENUMS
# =============================================================================

class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CognitiveStyle(str, Enum):
    IMPULSIVE = "impulsive"
    ANALYTICAL = "analytical"
    BALANCED = "balanced"


class PromptConfiguration(str, Enum):
    BASELINE = "baseline"  # Task-only
    STATS = "stats"  # + Behavioral statistics
    COT = "cot"  # + Chain-of-thought reasoning


class ModelTier(str, Enum):
    FRONTIER = "frontier"
    MID_TIER = "mid_tier"
    OPEN_SOURCE = "open_source"
    BUDGET = "budget"


class ProviderType(str, Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AWS_BEDROCK = "aws_bedrock"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(str, Enum):
    CLICK = "click"
    REPORT = "report"
    IGNORE = "ignore"
    INVALID = "invalid"
    ERROR = "error"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DecisionSpeed(str, Enum):
    FAST = "fast"  # <3 seconds
    MODERATE = "moderate"  # 3-10 seconds
    SLOW = "slow"  # >10 seconds


class EffectSizeInterpretation(str, Enum):
    """Cohen's d effect size interpretation"""
    NEGLIGIBLE = "negligible"  # d < 0.2
    SMALL = "small"  # 0.2 <= d < 0.5
    MEDIUM = "medium"  # 0.5 <= d < 0.8
    LARGE = "large"  # d >= 0.8


class BoundaryConditionType(str, Enum):
    """Types of boundary conditions"""
    OVER_DELIBERATION = "over_deliberation"
    EMOTIONAL_UNRESPONSIVE = "emotional_unresponsive"
    OVER_CLICKING = "over_clicking"
    TRUST_CALIBRATION = "trust_calibration"
    FAST_HEURISTIC = "fast_heuristic"


class SeverityLevel(str, Enum):
    """Severity levels for boundary conditions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# PERSONA SCHEMAS (Imported from Phase 1)
# =============================================================================

class TraitZScores(BaseModel):
    """All psychological trait z-scores from Phase 1"""
    crt_score: float = 0.0
    need_for_cognition: float = 0.0
    working_memory: float = 0.0
    big5_extraversion: float = 0.0
    big5_agreeableness: float = 0.0
    big5_conscientiousness: float = 0.0
    big5_neuroticism: float = 0.0
    big5_openness: float = 0.0
    impulsivity_total: float = 0.0
    sensation_seeking: float = 0.0
    trust_propensity: float = 0.0
    risk_taking: float = 0.0
    state_anxiety: float = 0.0
    current_stress: float = 0.0
    fatigue_level: float = 0.0
    phishing_self_efficacy: float = 0.0
    perceived_risk: float = 0.0
    security_attitudes: float = 0.0
    privacy_concern: float = 0.0
    phishing_knowledge: float = 0.0
    technical_expertise: float = 0.0
    prior_victimization: float = 0.0
    security_training: float = 0.0
    email_volume_numeric: float = 0.0
    link_click_tendency: float = 0.0
    social_media_usage: float = 0.0
    authority_susceptibility: float = 0.0
    urgency_susceptibility: float = 0.0
    scarcity_susceptibility: float = 0.0


class BehavioralStatistics(BaseModel):
    """Empirical behavioral data from Phase 1"""
    phishing_click_rate: float
    overall_accuracy: float
    report_rate: float
    false_positive_rate: Optional[float] = None
    mean_response_latency_ms: float
    median_response_latency_ms: Optional[float] = None
    hover_rate: float
    sender_inspection_rate: float


class EmailInteractionEffects(BaseModel):
    """How persona responds to email manipulations"""
    urgency_effect: float = 0.0
    familiarity_effect: float = 0.0
    framing_effect: float = 0.0
    aggressive_content_effect: Optional[float] = None


class ProcessMetrics(BaseModel):
    """Process-level behavioral data"""
    fast_decision_rate: float = 0.0
    cognitive_style_label: CognitiveStyle = CognitiveStyle.BALANCED
    mean_confidence: float = 0.0
    mean_suspicion_phishing: float = 0.0
    mean_suspicion_legit: float = 0.0


class BoundaryCondition(BaseModel):
    """Predicted AI failure condition"""
    type: str
    description: str
    severity: str  # high, medium, low


class ReasoningExample(BaseModel):
    """Chain-of-thought example for prompting"""
    scenario: str
    email_cues: str
    reasoning: str
    action: str
    confidence: str


class ExpertValidation(BaseModel):
    """Expert ratings from Phase 1 Delphi method"""
    realism_score: Optional[float] = None
    distinctiveness_score: Optional[float] = None
    actionability_score: Optional[float] = None
    icc_achieved: Optional[float] = None
    delphi_rounds_needed: Optional[int] = None


class Persona(BaseModel):
    """Complete persona definition imported from Phase 1"""
    persona_id: str
    cluster_id: int
    name: str
    archetype: Optional[str] = None
    risk_level: RiskLevel
    n_participants: int
    pct_of_population: float
    description: str
    
    # Psychological profile
    trait_zscores: Dict[str, float] = {}
    distinguishing_high_traits: List[str] = []
    distinguishing_low_traits: List[str] = []
    cognitive_style: CognitiveStyle = CognitiveStyle.BALANCED
    
    # Behavioral data
    behavioral_statistics: BehavioralStatistics
    email_interaction_effects: EmailInteractionEffects
    process_metrics: Optional[ProcessMetrics] = None
    
    # For prompting
    boundary_conditions: List[BoundaryCondition] = []
    reasoning_examples: List[ReasoningExample] = []
    
    # Validation
    expert_validation: Optional[ExpertValidation] = None
    
    # Phase 2 targets
    target_accuracy: float = 0.85
    acceptance_range: List[float] = [0.80, 0.90]


# =============================================================================
# EMAIL SCHEMAS
# =============================================================================

class EmailContent(BaseModel):
    """Full email content for simulation"""
    subject: str
    sender_display: str
    sender_email: str
    body_text: str
    link_url: Optional[str] = None
    link_display_text: Optional[str] = None


class EmailStimulus(BaseModel):
    """Complete email stimulus definition.

    Note: Removed unused columns that were never processed in analysis:
    - content_domain (always hardcoded as "general")
    - has_aggressive_content (use framing_type == "threat" instead)
    - has_spelling_errors (never used)
    - has_suspicious_url (never used)
    - requests_sensitive_info (never used)
    - aggression_level (never used)

    For emotional susceptibility analysis, use framing_type from factorial design.
    """
    email_id: str
    email_type: str  # phishing, legitimate
    sender_familiarity: str  # familiar, unfamiliar
    urgency_level: str  # high, medium, low
    framing_type: str  # threat, reward, neutral
    phishing_quality: Optional[str] = None  # high, medium, low
    ground_truth: int  # 1=phishing, 0=legitimate

    # Email content fields (from JSON import)
    subject_line: Optional[str] = None
    sender_display: Optional[str] = None
    sender_email: Optional[str] = None
    body_text: Optional[str] = None
    link_url: Optional[str] = None
    link_display_text: Optional[str] = None

    # Legacy fields for backward compatibility
    sender: Optional[str] = None  # deprecated, use sender_display
    body: Optional[str] = None  # deprecated, use body_text

    content: Optional[EmailContent] = None


# =============================================================================
# MODEL SCHEMAS
# =============================================================================

class ModelConfig(BaseModel):
    """Configuration for a single LLM model"""
    model_id: str
    display_name: str
    provider: ProviderType
    tier: ModelTier
    provider_model_id: str  # The actual ID used in API calls
    
    # Version tracking
    version: Optional[str] = None
    version_date: Optional[str] = None
    
    # Capabilities
    max_tokens: int = 4096
    supports_temperature: bool = True
    supports_system_prompt: bool = True
    
    # Costs (per 1K tokens)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    
    # Rate limits
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    
    # Status
    enabled: bool = True
    last_tested: Optional[datetime] = None
    health_status: str = "unknown"  # healthy, degraded, unavailable


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider"""
    provider_type: ProviderType
    display_name: str
    enabled: bool = False
    api_key_configured: bool = False
    base_url: Optional[str] = None
    region: Optional[str] = None
    models: List[ModelConfig] = []


# =============================================================================
# EXPERIMENT SCHEMAS
# =============================================================================

class ExperimentConfig(BaseModel):
    """Configuration for an experiment run"""
    experiment_id: str
    name: str
    description: Optional[str] = None
    
    # Selections
    persona_ids: List[str]
    model_ids: List[str]
    prompt_configs: List[PromptConfiguration]
    email_ids: List[str]
    
    # Parameters
    trials_per_condition: int = 30
    temperature: float = 0.3
    max_tokens: int = 500
    
    # Status
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    total_trials: int = 0
    completed_trials: int = 0
    failed_trials: int = 0


class SimulationTrial(BaseModel):
    """A single simulation trial"""
    trial_id: str
    experiment_id: str
    
    # Condition
    persona_id: str
    model_id: str
    prompt_config: PromptConfiguration
    email_id: str
    trial_number: int
    
    # Input
    system_prompt: str
    user_prompt: str
    temperature: float
    
    # Output
    action: ActionType = ActionType.ERROR
    confidence: Optional[ConfidenceLevel] = None
    simulated_speed: Optional[DecisionSpeed] = None
    reasoning_text: Optional[str] = None
    raw_response: str = ""
    
    # Metadata
    model_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    
    # Status
    parse_success: bool = False
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# ANALYSIS SCHEMAS - ENHANCED
# =============================================================================

class FidelityMetrics(BaseModel):
    """Behavioral fidelity metrics for a model-persona combination"""
    persona_id: str
    model_id: str
    prompt_config: PromptConfiguration
    
    # Primary metrics (from ARA proposal)
    normalized_accuracy: float
    decision_agreement: float
    effect_preservation_r: Optional[float] = None
    
    # Breakdown by action
    ai_click_rate: float
    human_click_rate: float
    click_rate_diff: float
    
    ai_report_rate: float
    human_report_rate: float
    report_rate_diff: float
    
    # Statistical
    n_trials: int
    n_emails: int
    ci_lower: float
    ci_upper: float
    
    # Pass/Fail
    meets_threshold: bool
    threshold_used: float = 0.85


class EnhancedFidelityMetrics(BaseModel):
    """Extended fidelity metrics with statistical rigor"""
    persona_id: str
    persona_name: Optional[str] = None
    model_id: str
    prompt_config: str
    
    # Primary metrics
    normalized_accuracy: float
    decision_agreement: float
    effect_preservation_r: Optional[float] = None
    
    # Action rates
    ai_click_rate: float
    human_click_rate: float
    click_rate_diff: float
    ai_report_rate: float
    human_report_rate: float
    report_rate_diff: float
    
    # Statistical
    n_trials: int
    n_emails: int
    ci_lower: float
    ci_upper: float
    
    # Cohen's d effect size
    cohens_d: Optional[float] = None
    cohens_d_interpretation: Optional[str] = None  # negligible, small, medium, large
    
    # Effect preservation details
    urgency_effect_ai: Optional[float] = None
    urgency_effect_human: Optional[float] = None
    familiarity_effect_ai: Optional[float] = None
    familiarity_effect_human: Optional[float] = None
    
    # Pass/Fail for each threshold
    meets_threshold: bool = False  # Normalized accuracy >= 85%
    meets_decision_threshold: bool = False  # Decision agreement >= 80%
    meets_effect_threshold: bool = False  # Effect preservation r >= 0.80
    threshold_used: float = 0.85
    
    # Metadata
    model_version: Optional[str] = None
    validation_timestamp: Optional[str] = None


class ModelComparisonResult(BaseModel):
    """Comparison across models for a single persona"""
    persona_id: str
    persona_name: str
    
    # Results per model
    model_results: Dict[str, FidelityMetrics]
    
    # Rankings
    best_fidelity_model: str
    best_cost_efficiency_model: str
    recommended_model: str
    recommendation_reason: str


class EnhancedModelComparison(BaseModel):
    """Enhanced model comparison with Pareto analysis"""
    model_id: str
    display_name: Optional[str] = None
    tier: Optional[str] = None
    
    # Fidelity metrics
    mean_fidelity: float
    std_fidelity: float = 0.0
    min_fidelity: float = 0.0
    max_fidelity: float = 0.0
    
    # Decision agreement
    mean_decision_agreement: float = 0.0
    
    # Click rate error
    mean_click_error: float = 0.0
    
    # Cost metrics
    total_cost: float = 0.0
    cost_per_decision: float = 0.0
    
    # Performance metrics
    total_trials: int = 0
    parse_success_rate: float = 0.0
    n_personas_tested: int = 0
    
    # Latency statistics
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Pareto analysis
    is_pareto_optimal: bool = False


class BoundaryConditionResult(BaseModel):
    """Analysis of where AI fails - Enhanced"""
    type: str
    persona_id: str
    persona_name: Optional[str] = None
    
    description: str
    severity: str  # high, medium, low
    
    human_pattern: str
    ai_pattern: str
    discrepancy: float  # Quantified difference
    
    recommendation: str
    
    # Optional: affected models
    affected_models: List[str] = []


class EffectPreservationResult(BaseModel):
    """Detailed effect preservation analysis"""
    persona_id: str
    persona_name: Optional[str] = None
    n_trials: int
    
    # Urgency effect
    urgency_effect_human: Optional[float] = None
    urgency_effect_ai: Optional[float] = None
    urgency_preservation: Optional[float] = None
    
    # Familiarity effect
    familiarity_effect_human: Optional[float] = None
    familiarity_effect_ai: Optional[float] = None
    familiarity_preservation: Optional[float] = None
    
    # Overall
    overall_effect_preservation: Optional[float] = None
    meets_threshold: bool = False


class CostPerformancePoint(BaseModel):
    """Single point on cost-performance curve"""
    model_id: str
    model_name: str
    tier: ModelTier
    
    fidelity_score: float
    cost_per_decision: float
    latency_p50_ms: float
    latency_p95_ms: float
    
    is_pareto_optimal: bool = False


class DeploymentRecommendation(BaseModel):
    """Recommendations for deployment"""
    best_fidelity_model: Optional[str] = None
    best_fidelity_score: float = 0.0
    
    best_value_model: Optional[str] = None
    best_value_score: float = 0.0
    best_value_cost: float = 0.0
    
    pareto_optimal_models: List[str] = []
    
    problematic_personas: List[str] = []
    high_severity_issues: int = 0
    
    overall_recommendation: str = ""


class AnalysisSummary(BaseModel):
    """Summary statistics for analysis results"""
    total_conditions: int = 0
    passing_conditions: int = 0
    pass_rate: float = 0.0
    
    mean_normalized_accuracy: float = 0.0
    std_normalized_accuracy: float = 0.0
    mean_decision_agreement: float = 0.0


# =============================================================================
# API REQUEST/RESPONSE SCHEMAS
# =============================================================================

class ImportPersonasRequest(BaseModel):
    """Request to import personas from Phase 1"""
    phase1_export: Dict[str, Any]  # The full Phase 1 export JSON


class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment"""
    name: str
    description: Optional[str] = None
    persona_ids: List[str]
    model_ids: List[str]
    prompt_configs: List[PromptConfiguration]
    email_ids: List[str]
    trials_per_condition: int = 30
    temperature: float = 0.3


class RunExperimentRequest(BaseModel):
    """Request to start running an experiment"""
    experiment_id: str
    resume_from_checkpoint: bool = False


class ProviderSetupRequest(BaseModel):
    """Request to configure a provider"""
    provider_type: Optional[ProviderType] = None
    
    # Common auth
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # AWS Bedrock specific
    region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    
    # Additional options
    timeout: Optional[int] = None
    max_retries: Optional[int] = None


class FidelityAnalysisResponse(BaseModel):
    """Response from fidelity analysis endpoint"""
    experiment_id: str
    fidelity_results: List[Dict[str, Any]]
    summary: AnalysisSummary
    thresholds: Dict[str, float]


class ModelComparisonResponse(BaseModel):
    """Response from model comparison endpoint"""
    experiment_id: str
    model_comparison: Dict[str, EnhancedModelComparison]
    ranking: List[Dict[str, Any]]
    pareto_frontier: List[str]
    best_model: Optional[str] = None
    best_value_model: Optional[str] = None


class BoundaryConditionResponse(BaseModel):
    """Response from boundary condition analysis endpoint"""
    experiment_id: str
    boundary_conditions: List[BoundaryConditionResult]
    by_severity: Dict[str, List[BoundaryConditionResult]]
    by_type: Dict[str, List[BoundaryConditionResult]]
    summary: Dict[str, int]