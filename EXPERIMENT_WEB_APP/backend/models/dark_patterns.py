"""
CYPEARL Dark Patterns Experiment Models

Data models for the dark patterns UI/UX experiment.
Tracks user interactions with potentially deceptive interface elements.

Factorial Design (2×2×2×2 = 16 tasks):
- UI Type: Dark Pattern vs Clean UI
- Intensity: Aggressive vs Mild
- Time Pressure: Timer Present vs No Timer
- Visual Manipulation: High Salience vs Neutral
"""

from pydantic import BaseModel, Field, computed_field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


class UIType(str, Enum):
    DARK = "dark"
    CLEAN = "clean"


class Intensity(str, Enum):
    AGGRESSIVE = "aggressive"
    MILD = "mild"
    NONE = "none"


class VisualManipulation(str, Enum):
    HIGH = "high"
    NEUTRAL = "neutral"


class TaskContext(str, Enum):
    COOKIE_CONSENT = "cookie_consent"
    NEWSLETTER_UNSUBSCRIBE = "newsletter_unsubscribe"
    FREE_TRIAL = "free_trial"
    ACCOUNT_DELETION = "account_deletion"
    CHECKOUT_ADDON = "checkout_addon"
    SHIPPING_UPGRADE = "shipping_upgrade"
    PRIVACY_SETTINGS = "privacy_settings"
    DECLINE_OFFER = "decline_offer"


# ============================================================================
# Task Definition Models
# ============================================================================

class DarkPatternTask(BaseModel):
    """Definition of a single UI task in the experiment"""
    task_id: str
    ui_type: UIType
    intensity: Intensity
    time_pressure: bool
    visual_manipulation: VisualManipulation
    context: TaskContext
    description: str
    dark_patterns_present: List[str] = []
    desired_action: str  # What user SHOULD do
    manipulated_action: Optional[str] = None  # What dark pattern pushes toward
    time_limit_seconds: Optional[int] = None  # For time pressure tasks


class ClickEvent(BaseModel):
    """Record of a single click interaction"""
    element_id: str
    element_type: str  # 'button', 'link', 'checkbox', 'toggle', etc.
    timestamp: datetime
    is_correct_option: bool = False


class HoverEvent(BaseModel):
    """Record of a hover interaction"""
    element_id: str
    element_type: str  # 'correct_option', 'manipulated_option', 'fine_print', 'help_icon'
    hover_start: datetime
    hover_end: Optional[datetime] = None
    duration_ms: int = 0


class ScrollEvent(BaseModel):
    """Record of a scroll event"""
    scroll_depth: float  # 0.0 to 1.0
    timestamp: datetime


# ============================================================================
# Session/Response Models
# ============================================================================

class DarkPatternSession(BaseModel):
    """
    Complete session data for a single task interaction.
    This is what gets saved to MongoDB after each task.
    """
    # Identifiers
    task_id: str
    participant_id: str

    # Factorial Conditions
    ui_type: UIType
    intensity: Intensity
    time_pressure: bool
    visual_manipulation: VisualManipulation
    context: TaskContext
    ground_truth: int = Field(description="1 if dark pattern present, 0 if clean UI")

    # Timing
    task_started_at: datetime
    task_completed_at: Optional[datetime] = None
    task_completion_time_ms: int = 0
    time_to_first_action_ms: int = 0

    # Primary Outcomes
    final_action: str  # The action user took (e.g., "accept_all", "reject_all", "abandon")
    manipulated: bool = False  # Did user do what the dark pattern wanted?
    resisted: bool = False  # Did user achieve their desired outcome?
    abandoned: bool = False  # Did user give up without completing?

    # Process Metrics (tracked silently)
    scroll_depth: float = 0.0  # 0.0 to 1.0 - how far they scrolled
    scroll_to_target_ms: Optional[int] = None  # Time to scroll to correct option
    fine_print_expand_count: int = 0  # Times expanded terms/details
    fine_print_hover_time_ms: int = 0
    option_hover_count: int = 0  # Hovers on interactive elements
    correct_option_hover: bool = False  # Did they hover over the "right" choice?
    correct_option_hover_time_ms: int = 0
    backtrack_count: int = 0  # Times changed selection or went back
    click_path: List[str] = []  # Sequence of element IDs clicked
    click_path_length: int = 0

    # For obstruction patterns (multi-step flows)
    steps_completed: int = 0  # How many steps in multi-step process
    steps_total: int = 1

    # Self-Report (after each task)
    perceived_difficulty: Optional[int] = None  # 1-7
    noticed_unusual: Optional[bool] = None  # "Did you notice anything unusual?"
    confidence_rating: Optional[int] = None  # 1-7

    # Raw event data (optional, for detailed analysis)
    scroll_events: List[ScrollEvent] = []
    hover_events: List[HoverEvent] = []
    click_events: List[ClickEvent] = []

    @computed_field
    @property
    def correct_response(self) -> int:
        """1 if user achieved desired outcome, 0 otherwise"""
        return 1 if self.resisted else 0


class DarkPatternParticipant(BaseModel):
    """
    Participant record for dark patterns experiment.
    Created on login, updated throughout experiment.
    """
    participant_id: str
    prolific_id: str
    session_id: str

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    scenario: str = "dark-patterns"

    # Progress tracking
    current_task_index: int = 0
    tasks_completed: int = 0
    total_tasks: int = 16
    is_finished: bool = False

    # Task order (randomized per participant)
    task_order: List[str] = []  # List of task_ids in presentation order

    # Survey completion flags
    pre_survey_completed: bool = False
    post_survey_completed: bool = False

    # Client info
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None

    # Aggregated outcome metrics (computed after completion)
    manipulation_rate: Optional[float] = None  # % tasks where user was manipulated
    resistance_rate: Optional[float] = None  # % tasks where user resisted
    detection_rate: Optional[float] = None  # % tasks where user noticed unusual
    false_alarm_rate: Optional[float] = None  # % clean UIs flagged as unusual
    mean_task_time_ms: Optional[float] = None
    mean_scroll_depth: Optional[float] = None
    fine_print_inspection_rate: Optional[float] = None
    abandonment_rate: Optional[float] = None


# ============================================================================
# Survey Models
# ============================================================================

class DigitalLiteracyItem(BaseModel):
    """Digital literacy scale item"""
    item_id: str
    response: int  # 1-5 Likert


class ImpulseBuyingItem(BaseModel):
    """Impulse buying tendency item"""
    item_id: str
    response: int  # 1-5 Likert


class DarkPatternAwarenessItem(BaseModel):
    """Dark pattern awareness item"""
    item_id: str
    response: str  # Various response types


class DarkPatternsPreSurvey(BaseModel):
    """
    Pre-survey data specific to dark patterns experiment.
    Extends the core psychological traits with domain-specific measures.
    """
    participant_id: str
    session_id: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    # Demographics (shared with phishing)
    age: Optional[int] = None
    gender: Optional[str] = None
    education: Optional[str] = None
    technical_field: Optional[bool] = None
    employment: Optional[str] = None
    industry: Optional[str] = None

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

    # Dark Patterns Specific (13 items)
    # Digital Literacy Scale (4 items)
    digital_literacy_responses: List[dict] = []
    digital_literacy: Optional[float] = None

    # Impulse Buying Tendency (4 items)
    impulse_buying_responses: List[dict] = []
    impulse_buying: Optional[float] = None

    # Online Shopping Experience (2 items)
    shopping_frequency: Optional[str] = None  # Never to Daily
    years_online_shopping: Optional[int] = None

    # Dark Pattern Awareness (3 items)
    dp_awareness_heard_term: Optional[str] = None  # Yes/No/Not sure
    dp_awareness_felt_tricked: Optional[str] = None  # Never to Many times
    dp_awareness_confidence: Optional[int] = None  # 1-5

    # Modified from phishing
    manipulation_detection_efficacy: Optional[float] = None
    perceived_online_risk: Optional[float] = None
    online_shopping_habits: Optional[dict] = None


class DarkPatternsPostSurvey(BaseModel):
    """
    Post-survey data specific to dark patterns experiment.
    State measures + dark pattern specific items.
    """
    participant_id: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    # State measures (shared)
    state_anxiety: Optional[int] = None  # 1-7
    current_stress: Optional[int] = None  # 1-7
    fatigue_level: Optional[int] = None  # 1-7

    # Dark pattern specific
    frustration_level: Optional[int] = None  # 1-7
    felt_manipulated: Optional[int] = None  # 1-7
    regret_choices: Optional[int] = None  # 1-7

    # Detection quiz results (show 4 UI screenshots)
    detection_quiz_responses: List[dict] = []
    detection_quiz_score: Optional[int] = None

    # Open-ended feedback
    noticed_patterns: Optional[str] = None
    general_feedback: Optional[str] = None


# ============================================================================
# API Request/Response Models
# ============================================================================

class TaskStartRequest(BaseModel):
    """Request to start a new task"""
    task_id: str


class TaskActionRequest(BaseModel):
    """Request to submit task action and metrics"""
    task_id: str
    final_action: str

    # Timing
    task_completion_time_ms: int
    time_to_first_action_ms: int

    # Process metrics
    scroll_depth: float = 0.0
    scroll_to_target_ms: Optional[int] = None
    fine_print_expand_count: int = 0
    fine_print_hover_time_ms: int = 0
    option_hover_count: int = 0
    correct_option_hover: bool = False
    correct_option_hover_time_ms: int = 0
    backtrack_count: int = 0
    click_path: List[str] = []
    steps_completed: int = 0
    steps_total: int = 1

    # Self-report (Optional for demo/testing — TODO: make required for production)
    perceived_difficulty: Optional[int] = None
    noticed_unusual: Optional[bool] = None
    confidence_rating: Optional[int] = None

    # Raw events (optional)
    scroll_events: List[dict] = []
    hover_events: List[dict] = []
    click_events: List[dict] = []


class TaskCompleteRequest(BaseModel):
    """Request to complete a task and get next"""
    pass


class ParticipantProgressResponse(BaseModel):
    """Response with participant progress"""
    participant_id: str
    current_task_index: int
    tasks_completed: int
    total_tasks: int
    is_finished: bool
    next_task_id: Optional[str] = None


# ============================================================================
# Export Models (for CSV generation)
# ============================================================================

class DarkPatternsParticipantExport(BaseModel):
    """Flattened participant data for CSV export"""
    participant_id: str
    prolific_id: str
    age: Optional[int]
    gender: Optional[str]
    education: Optional[str]
    technical_field: Optional[bool]

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

    # Dark patterns specific
    digital_literacy: Optional[float]
    impulse_buying: Optional[float]
    shopping_frequency: Optional[str]
    years_online_shopping: Optional[int]
    dp_awareness_heard_term: Optional[str]
    dp_awareness_felt_tricked: Optional[str]
    manipulation_detection_efficacy: Optional[float]
    perceived_online_risk: Optional[float]

    # Outcome metrics
    manipulation_rate: Optional[float]
    resistance_rate: Optional[float]
    detection_rate: Optional[float]
    false_alarm_rate: Optional[float]
    mean_task_time_ms: Optional[float]
    mean_scroll_depth: Optional[float]
    fine_print_inspection_rate: Optional[float]
    abandonment_rate: Optional[float]


class DarkPatternsResponseExport(BaseModel):
    """Flattened response data for CSV export"""
    participant_id: str
    task_id: str
    ui_type: str
    intensity: str
    time_pressure: bool
    visual_manipulation: str
    context: str
    ground_truth: int
    final_action: str
    manipulated: bool
    resisted: bool
    abandoned: bool
    task_completion_time_ms: int
    time_to_first_action_ms: int
    scroll_depth: float
    fine_print_expand_count: int
    option_hover_count: int
    correct_option_hover: bool
    backtrack_count: int
    click_path_length: int
    steps_completed: int
    steps_total: int
    perceived_difficulty: Optional[int]
    noticed_unusual: Optional[bool]
    confidence_rating: Optional[int]
