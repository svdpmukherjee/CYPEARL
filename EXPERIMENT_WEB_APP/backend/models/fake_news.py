"""
CYPEARL Fake News / Misinformation Experiment Models

Data models for the fake news susceptibility experiment.
Tracks user interactions with news headlines of varying veracity.

Factorial Design (2x2x2x2 = 16 news items):
- Veracity: Fake vs Real
- Political Congruence: Congruent vs Incongruent (computed at runtime)
- Source Credibility: High (mainstream) vs Low (unknown)
- Emotional Valence: High vs Neutral
"""

from pydantic import BaseModel, Field, computed_field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


class Veracity(str, Enum):
    FAKE = "fake"
    REAL = "real"


class PoliticalLean(str, Enum):
    LIBERAL = "liberal"
    CONSERVATIVE = "conservative"
    NEUTRAL = "neutral"


class Congruence(str, Enum):
    CONGRUENT = "congruent"
    INCONGRUENT = "incongruent"
    NEUTRAL = "neutral"


class SourceCredibility(str, Enum):
    HIGH = "high"
    LOW = "low"


class EmotionalValence(str, Enum):
    HIGH = "high"
    NEUTRAL = "neutral"


class QuestionOrder(str, Enum):
    ACCURACY_FIRST = "accuracy_first"
    SHARING_FIRST = "sharing_first"


# ============================================================================
# News Item Definition
# ============================================================================

class NewsItem(BaseModel):
    """Definition of a single news item in the experiment"""
    item_id: str
    veracity: Veracity
    political_lean: PoliticalLean
    source_credibility: SourceCredibility
    emotional_valence: EmotionalValence
    headline: str
    source_name: str
    source_logo: Optional[str] = None
    source_verified: bool = False
    thumbnail: Optional[str] = None
    thumbnail_type: Optional[str] = "image"  # "image" or "video"
    engagement_counts: dict = Field(default_factory=lambda: {"comments": 0, "shares": 0, "likes": 0})
    topic: str
    fact_check_status: str  # "true", "false", "mixed"
    manipulation_techniques: List[str] = []


# ============================================================================
# Interaction Events
# ============================================================================

class SourceHoverEvent(BaseModel):
    """Record of hovering over source info"""
    hover_start: datetime
    hover_end: Optional[datetime] = None
    duration_ms: int = 0


class ScrollEvent(BaseModel):
    """Record of scroll event"""
    scroll_depth: float  # 0.0 to 1.0
    timestamp: datetime


class EngagementHoverEvent(BaseModel):
    """Record of hovering over engagement buttons (share, like, etc.)"""
    element_type: str  # 'share_button', 'like_button', 'comment_button'
    timestamp: datetime


# ============================================================================
# Session/Response Models
# ============================================================================

class NewsEvaluationSession(BaseModel):
    """
    Complete session data for a single news item evaluation.
    This is what gets saved to MongoDB after each item.
    """
    # Identifiers
    item_id: str
    participant_id: str

    # Factorial Conditions
    veracity: str  # "fake" or "real"
    ground_truth: int = Field(description="1 = real, 0 = fake")
    political_lean: str  # "liberal", "conservative", "neutral"
    congruence: str  # "congruent" or "incongruent" - COMPUTED from participant ideology
    source_credibility: str  # "high" or "low"
    emotional_valence: str  # "high" or "neutral"
    topic: str

    # Question Order (counterbalanced)
    question_order: str  # "accuracy_first" or "sharing_first"

    # Timing
    item_displayed_at: datetime
    judgment_completed_at: Optional[datetime] = None
    reading_time_ms: int = 0
    time_to_accuracy_judgment_ms: int = 0
    time_to_sharing_judgment_ms: int = 0

    # Primary Responses
    accuracy_rating: Optional[int] = None  # 1-7 (1 = definitely false, 7 = definitely true)
    sharing_intention: Optional[int] = None  # 1-7 (1 = definitely would not share, 7 = definitely would share)
    seen_before: Optional[bool] = None  # "Have you seen this headline before?"
    confidence: Optional[int] = None  # 1-7

    # Qualitative Responses (for chain-of-thought LLM conditioning)
    reason: Optional[str] = None  # Combined qualitative response string
    cues_noticed: Optional[str] = None  # "What specific cues or details helped you evaluate this headline?"
    evaluation_process: Optional[str] = None  # "How did you decide whether this headline was accurate?"
    influencing_factors: Optional[str] = None  # "What factors most influenced your judgment?"
    uncertainty_points: Optional[str] = None  # "What were you unsure about?"

    # Process Metrics (tracked silently)
    source_hover: bool = False  # Did they hover over source name/logo?
    source_hover_time_ms: int = 0
    source_click: bool = False  # Did they click to see source details?
    headline_reread: bool = False  # Did they scroll back to re-read headline?
    engagement_hover: bool = False  # Did they hover over share/like buttons?
    scroll_depth: float = 0.0  # How far they scrolled (0.0 to 1.0)
    external_search_attempted: bool = False  # Did they attempt to fact-check?

    # Raw event data (optional, for detailed analysis)
    scroll_events: List[dict] = []
    hover_events: List[dict] = []

    @computed_field
    @property
    def believed(self) -> int:
        """1 if accuracy_rating > 4, 0 otherwise"""
        if self.accuracy_rating is None:
            return 0
        return 1 if self.accuracy_rating > 4 else 0

    @computed_field
    @property
    def would_share(self) -> int:
        """1 if sharing_intention > 4, 0 otherwise"""
        if self.sharing_intention is None:
            return 0
        return 1 if self.sharing_intention > 4 else 0

    @computed_field
    @property
    def correct_judgment(self) -> int:
        """1 if correctly identified veracity"""
        if self.accuracy_rating is None:
            return 0
        if self.veracity == "fake":
            return 1 if self.accuracy_rating <= 4 else 0
        else:
            return 1 if self.accuracy_rating > 4 else 0


class FakeNewsParticipant(BaseModel):
    """
    Participant record for fake news experiment.
    Created on login, updated throughout experiment.
    """
    participant_id: str
    prolific_id: str
    session_id: str

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    scenario: str = "fake-news"

    # Political Ideology (from pre-survey, used for congruence calculation)
    political_ideology: Optional[int] = None  # 1-7 scale (1=Very Liberal, 7=Very Conservative)
    party_id: Optional[str] = None  # democrat, republican, independent, other, none

    # Question Order Assignment (counterbalanced)
    question_order: str = "accuracy_first"  # or "sharing_first"

    # Progress tracking
    current_item_index: int = 0
    items_completed: int = 0
    total_items: int = 16
    is_finished: bool = False

    # Item order (randomized per participant)
    item_order: List[str] = []  # List of item_ids in presentation order

    # Survey completion flags
    pre_survey_completed: bool = False
    post_survey_completed: bool = False

    # Client info
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None

    # Aggregated outcome metrics (computed after completion)
    # Accuracy Discernment
    fake_belief_rate: Optional[float] = None  # % of fake news believed
    real_belief_rate: Optional[float] = None  # % of real news believed
    accuracy_discernment: Optional[float] = None  # Real belief rate - Fake belief rate

    # Sharing Discernment
    fake_share_rate: Optional[float] = None  # % of fake news would share
    real_share_rate: Optional[float] = None  # % of real news would share
    sharing_discernment: Optional[float] = None  # Real share rate - Fake share rate

    # Signal Detection Theory metrics
    d_prime: Optional[float] = None  # Discrimination ability
    criterion: Optional[float] = None  # Response bias

    # Partisan Bias
    congruence_bias: Optional[float] = None  # d' congruent - d' incongruent

    # Process metrics
    mean_reading_time_ms: Optional[float] = None
    source_inspection_rate: Optional[float] = None


# ============================================================================
# Survey Models
# ============================================================================

class FakeNewsPreSurvey(BaseModel):
    """
    Pre-survey data specific to fake news experiment.
    Includes core psychological traits + domain-specific measures.
    """
    participant_id: str
    session_id: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    # Demographics (shared with other experiments)
    age: Optional[int] = None
    gender: Optional[str] = None
    education: Optional[str] = None
    technical_field: Optional[bool] = None
    employment: Optional[str] = None
    industry: Optional[str] = None

    # Political Ideology (CRITICAL for congruence calculation)
    political_ideology: Optional[int] = None  # 1-7 scale
    party_id: Optional[str] = None  # democrat, republican, independent, other, none

    # Core Psychological Traits (shared - 53 items)
    # CRT-7
    crt_responses: List[dict] = []
    crt_score: Optional[int] = None

    # NFC-6
    nfc_responses: List[dict] = []
    need_for_cognition: Optional[float] = None

    # Working Memory
    working_memory: Optional[int] = None

    # BFI-10
    big5_extraversion: Optional[float] = None
    big5_agreeableness: Optional[float] = None
    big5_conscientiousness: Optional[float] = None
    big5_neuroticism: Optional[float] = None
    big5_openness: Optional[float] = None

    # UPPS-P Short
    impulsivity_total: Optional[float] = None

    # BSSS-4
    sensation_seeking: Optional[float] = None

    # Trust Propensity
    trust_propensity: Optional[float] = None

    # DOSPERT-6
    risk_taking: Optional[float] = None

    # Influence Susceptibility
    authority_susceptibility: Optional[float] = None
    urgency_susceptibility: Optional[float] = None
    scarcity_susceptibility: Optional[float] = None

    # Fake News Specific Measures (23 items)
    # Conspiracy Mentality Questionnaire - CMQ-5
    cmq_responses: List[dict] = []
    conspiracy_mentality: Optional[float] = None

    # News Media Literacy - NML-6
    nml_responses: List[dict] = []
    news_media_literacy: Optional[float] = None

    # Bullshit Receptivity Scale - BSR-4
    bsr_responses: List[dict] = []
    bullshit_receptivity: Optional[float] = None

    # Trust in Media (3 items)
    trust_in_media_responses: List[dict] = []
    trust_in_media: Optional[float] = None

    # Fear of Missing Out - FoMOs-3
    fomo_responses: List[dict] = []
    fomo: Optional[float] = None

    # Modified from phishing
    fake_news_detection_efficacy: Optional[float] = None  # Was: phishing_self_efficacy
    perceived_misinfo_harm: Optional[float] = None  # Was: perceived_risk

    # News Consumption Habits (4 items)
    news_consumption_responses: List[dict] = []
    news_consumption_frequency: Optional[str] = None
    primary_news_source: Optional[str] = None
    social_media_news_exposure: Optional[str] = None


class FakeNewsPostSurvey(BaseModel):
    """
    Post-survey data specific to fake news experiment.
    State measures + fake news specific items.
    """
    participant_id: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    # State measures (shared)
    state_anxiety: Optional[int] = None  # 1-7
    current_stress: Optional[int] = None  # 1-7
    fatigue_level: Optional[int] = None  # 1-7

    # Fake news specific
    felt_deceived: Optional[int] = None  # 1-7: Did you feel deceived by any headlines?
    confidence_in_judgments: Optional[int] = None  # 1-7: Confidence in accuracy judgments

    # Fact-checking behavior
    fact_check_frequency: Optional[str] = None  # Never to Always

    # Source Memory Test (show 4 headlines, ask which source)
    source_memory_responses: List[dict] = []
    source_memory_score: Optional[int] = None

    # Open-ended
    strategy_description: Optional[str] = None  # How do you decide if a headline is true/false?
    general_feedback: Optional[str] = None


# ============================================================================
# API Request/Response Models
# ============================================================================

class NewsItemStartRequest(BaseModel):
    """Request to start viewing a new news item"""
    item_id: str


class NewsItemActionRequest(BaseModel):
    """Request to submit evaluation and metrics for a news item"""
    item_id: str

    # Primary responses (Optional for demo/testing — TODO: make required for production)
    accuracy_rating: Optional[int] = None  # 1-7
    sharing_intention: Optional[int] = None  # 1-7
    seen_before: Optional[bool] = None
    confidence: Optional[int] = None  # 1-7

    # Qualitative responses (for chain-of-thought LLM conditioning)
    reason: Optional[str] = None  # Combined qualitative response string
    cues_noticed: Optional[str] = None
    evaluation_process: Optional[str] = None
    influencing_factors: Optional[str] = None
    uncertainty_points: Optional[str] = None

    # Timing
    reading_time_ms: int
    time_to_accuracy_judgment_ms: int
    time_to_sharing_judgment_ms: int

    # Process metrics
    source_hover: bool = False
    source_hover_time_ms: int = 0
    source_click: bool = False
    headline_reread: bool = False
    engagement_hover: bool = False
    scroll_depth: float = 0.0

    # Raw events (optional)
    scroll_events: List[dict] = []
    hover_events: List[dict] = []


class ParticipantProgressResponse(BaseModel):
    """Response with participant progress"""
    participant_id: str
    current_item_index: int
    items_completed: int
    total_items: int
    is_finished: bool
    next_item_id: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_congruence(participant_ideology: Optional[int], headline_lean: str) -> str:
    """
    Calculate political congruence between participant and headline.

    Args:
        participant_ideology: 1-7 (1=Very Liberal, 7=Very Conservative)
        headline_lean: "liberal", "conservative", or "neutral"

    Returns:
        "congruent", "incongruent", or "neutral"
    """
    if headline_lean == "neutral":
        return "neutral"

    if participant_ideology is None:
        participant_ideology = 4  # Default to moderate

    is_liberal = participant_ideology < 4
    is_conservative = participant_ideology > 4

    if (is_liberal and headline_lean == "liberal") or \
       (is_conservative and headline_lean == "conservative"):
        return "congruent"
    else:
        return "incongruent"


def assign_question_order(participant_id: str) -> str:
    """
    Assign question order (accuracy_first or sharing_first) for counterbalancing.
    Uses participant_id hash for consistent assignment.
    """
    hash_value = sum(ord(c) for c in participant_id)
    return "accuracy_first" if hash_value % 2 == 0 else "sharing_first"


# ============================================================================
# Export Models (for CSV generation)
# ============================================================================

class FakeNewsParticipantExport(BaseModel):
    """Flattened participant data for CSV export"""
    participant_id: str
    prolific_id: str
    age: Optional[int]
    gender: Optional[str]
    education: Optional[str]
    technical_field: Optional[bool]

    # Political
    political_ideology: Optional[int]
    party_id: Optional[str]

    # Core traits
    crt_score: Optional[int]
    need_for_cognition: Optional[float]
    working_memory: Optional[int]
    big5_extraversion: Optional[float]
    big5_agreeableness: Optional[float]
    big5_conscientiousness: Optional[float]
    big5_neuroticism: Optional[float]
    big5_openness: Optional[float]
    impulsivity_total: Optional[float]
    sensation_seeking: Optional[float]
    trust_propensity: Optional[float]
    risk_taking: Optional[float]
    authority_susceptibility: Optional[float]
    urgency_susceptibility: Optional[float]
    scarcity_susceptibility: Optional[float]

    # Fake news specific traits
    conspiracy_mentality: Optional[float]
    news_media_literacy: Optional[float]
    bullshit_receptivity: Optional[float]
    trust_in_media: Optional[float]
    fomo: Optional[float]
    fake_news_detection_efficacy: Optional[float]
    perceived_misinfo_harm: Optional[float]

    # Outcome metrics
    fake_belief_rate: Optional[float]
    real_belief_rate: Optional[float]
    accuracy_discernment: Optional[float]
    fake_share_rate: Optional[float]
    real_share_rate: Optional[float]
    sharing_discernment: Optional[float]
    d_prime: Optional[float]
    criterion: Optional[float]
    congruence_bias: Optional[float]
    mean_reading_time_ms: Optional[float]
    source_inspection_rate: Optional[float]


class FakeNewsResponseExport(BaseModel):
    """Flattened response data for CSV export"""
    participant_id: str
    item_id: str
    veracity: str
    ground_truth: int
    political_lean: str
    congruence: str
    source_credibility: str
    emotional_valence: str
    topic: str
    question_order: str
    accuracy_rating: Optional[int]
    sharing_intention: Optional[int]
    believed: int
    would_share: int
    correct_judgment: int
    seen_before: Optional[bool]
    confidence: Optional[int]
    # Qualitative responses
    reason: Optional[str]
    cues_noticed: Optional[str]
    evaluation_process: Optional[str]
    influencing_factors: Optional[str]
    uncertainty_points: Optional[str]
    # Timing and behavioral metrics
    reading_time_ms: int
    time_to_accuracy_ms: int
    time_to_sharing_ms: int
    source_hover: bool
    source_hover_time_ms: int
    source_click: bool
    headline_reread: bool
    engagement_hover: bool
    scroll_depth: float
