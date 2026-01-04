"""
CYPEARL Experiment Web App - Data Models (UPDATED v2)

UPDATES IN THIS VERSION:
1. sender_hover replaced with sender_click tracking
2. Added sender_click_count field
3. Bonus fields remain but are hidden from user during study

Key observational data tracked:
- clicked: Did participant click any link in email (0/1)
- reported: Did participant report email as phishing (0/1)
- ignored: Did participant ignore/skip email (0/1)
- response_latency_ms: Time from email opened to action submitted
- dwell_time_ms: Total time email was visible/open
- hovered_link: Did participant hover over any link (0/1)
- inspected_sender: Did participant click to expand sender details (0/1)
- confidence_rating: Self-reported confidence (1-10)
- suspicion_rating: Self-reported suspicion level (1-10)
- bonus_total: Accumulated bonus points (hidden during study)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from enum import Enum


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class ActionType(str, Enum):
    """Possible actions a participant can take on an email"""
    SAFE = "safe"           # Mark as safe/legitimate
    REPORT = "report"       # Report as phishing/suspicious
    DELETE = "delete"       # Delete email
    IGNORE = "ignore"       # Ignore/skip email
    LINK_CLICK = "link_click"       # Clicked a link in email body
    LINK_HOVER = "link_hover"       # Hovered over a link
    SENDER_CLICK = "sender_click"   # Clicked to expand sender details (NEW)
    MARK_READ = "mark_read"         # Marked email as read
    EMAIL_OPEN = "email_open"       # Opened/selected email
    BONUS_UPDATE = "bonus_update"   # Bonus points update (silent)


class Email(BaseModel):
    """Email stimulus model with factorial design metadata"""
    id: Optional[str] = None
    sender_name: str
    sender_email: str
    subject: str
    body: str
    is_phishing: bool
    order_id: int
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Factorial design fields (from email_stimuli.csv)
    experimental: bool = True
    factorial_category: Optional[Dict[str, str]] = None  # {type, sender, urgency, framing}
    aggression_level: Optional[str] = None  # very_high, high, medium, low
    tactics: Optional[List[str]] = None
    
    # Additional metadata for analysis
    email_type: Optional[str] = None  # phishing, legitimate
    sender_familiarity: Optional[str] = None  # known_internal, known_external, unknown_external
    urgency_level: Optional[str] = None  # high, low
    framing_type: Optional[str] = None  # threat, reward
    phishing_quality: Optional[str] = None  # high, low (for phishing emails)
    has_aggressive_content: Optional[bool] = None
    has_spelling_errors: Optional[bool] = None
    has_suspicious_url: Optional[bool] = None
    requests_sensitive_info: Optional[bool] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class HoverData(BaseModel):
    """Detailed hover tracking data"""
    link_hovers: List[Dict[str, Any]] = Field(default_factory=list)  # [{link, duration_ms, timestamp}]
    total_link_hover_time_ms: int = 0
    link_hover_count: int = 0


class BonusEntry(BaseModel):
    """Single bonus update entry (hidden from user during study)"""
    amount: int  # Positive for rewards, negative for losses
    total: int  # Running total after this change
    timestamp: datetime = Field(default_factory=datetime.now)
    link_clicked: Optional[str] = None  # URL of link that triggered bonus change
    email_id: Optional[str] = None
    link_type: Optional[str] = None  # 'legitimate' or 'phishing'


class UserAction(BaseModel):
    """
    Record of a single user action on an email.
    
    This captures both micro-actions (hover, click) and final decision actions.
    Multiple UserActions can exist per email per participant.
    """
    email_id: str
    action_type: str  # ActionType enum value
    reason: Optional[str] = None  # Qualitative explanation (for final actions)
    confidence: Optional[int] = None  # 1-10 scale
    suspicion: Optional[int] = None  # 1-10 scale
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Behavioral tracking
    hover_data: Optional[Dict[str, Any]] = None  # Link hover details
    latency_ms: Optional[int] = None  # Time since email opened
    
    # Context
    email_snapshot: Optional[Dict[str, Any]] = None  # Email metadata at action time
    client_info: Optional[Dict[str, Any]] = None  # Browser/device info


class EmailSession(BaseModel):
    """
    Tracks all interactions for a single email viewing session.
    
    This aggregates micro-actions into the final response record
    that maps to phishing_study_responses.csv
    """
    email_id: str
    participant_id: str
    
    # Timing (OBSERVATIONAL)
    opened_at: datetime = Field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    dwell_time_ms: int = 0  # Total viewing time
    response_latency_ms: int = 0  # Time to final action
    
    # Link interaction (OBSERVATIONAL)
    clicked_link: bool = False  # Did they click ANY link?
    clicked_links: List[str] = Field(default_factory=list)  # Which links clicked
    hovered_link: bool = False  # Did they hover ANY link?
    link_hover_count: int = 0
    total_link_hover_time_ms: int = 0
    
    # Sender inspection (OBSERVATIONAL) - Now tracked via click
    inspected_sender: bool = False  # Did they click to expand sender details?
    sender_click_count: int = 0  # How many times they expanded
    
    # Final action
    final_action: Optional[str] = None  # safe, report, delete, ignore
    confidence_rating: Optional[int] = None
    suspicion_rating: Optional[int] = None
    action_reason: Optional[str] = None  # Qualitative
    
    # Computed fields (for CSV export)
    @property
    def clicked(self) -> int:
        """Binary: 1 if any link was clicked, 0 otherwise"""
        return 1 if self.clicked_link else 0
    
    @property
    def reported(self) -> int:
        """Binary: 1 if final action was report, 0 otherwise"""
        return 1 if self.final_action == "report" else 0
    
    @property
    def ignored(self) -> int:
        """Binary: 1 if final action was ignore, 0 otherwise"""
        return 1 if self.final_action == "ignore" else 0
    
    @property
    def deleted(self) -> int:
        """Binary: 1 if final action was delete, 0 otherwise"""
        return 1 if self.final_action == "delete" else 0
    
    @property
    def marked_safe(self) -> int:
        """Binary: 1 if final action was safe, 0 otherwise"""
        return 1 if self.final_action == "safe" else 0


class Participant(BaseModel):
    """Participant session tracking"""
    id: Optional[str] = Field(None, alias="_id")
    prolific_id: Optional[str] = None  # External ID from recruitment platform
    
    # Progress tracking
    current_email_order: int = 0
    completed: bool = False
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Email state
    email_timestamps: Dict[str, datetime] = Field(default_factory=dict)  # order_id → delivery time
    read_email_ids: List[str] = Field(default_factory=list)
    deleted_email_ids: List[str] = Field(default_factory=list)
    
    # Session tracking (for response time calculations)
    email_sessions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # email_id → session data
    
    # Bonus tracking (HIDDEN from user during study, revealed at end)
    bonus_total: int = 0  # Current accumulated bonus
    bonus_history: List[Dict[str, Any]] = Field(default_factory=list)  # List of BonusEntry-like dicts
    
    # Client info
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ResponseExport(BaseModel):
    """
    Final export format matching phishing_study_responses.csv
    
    This is the schema for data export after experiment completion.
    """
    participant_id: str
    email_id: str
    
    # Email metadata (from email_stimuli.csv)
    email_type: str  # phishing, legitimate
    sender_familiarity: str  # known_internal, known_external, unknown_external
    urgency_level: str  # high, low
    framing_type: str  # threat, reward
    has_aggressive_content: bool
    phishing_quality: Optional[str]  # high, low, null for legitimate
    ground_truth: int  # 1 for phishing, 0 for legitimate
    
    # Final action
    action: str  # safe, report, delete, ignore
    
    # Binary outcome indicators
    clicked: int  # 0 or 1 - did they click any link
    reported: int  # 0 or 1
    ignored: int  # 0 or 1
    correct_response: int  # 0 or 1 - did they respond appropriately
    
    # Timing metrics (OBSERVATIONAL)
    response_latency_ms: int  # Time from email open to action
    dwell_time_ms: int  # Total time viewing email
    
    # Behavioral indicators (OBSERVATIONAL)
    hovered_link: int  # 0 or 1
    inspected_sender: int  # 0 or 1 (via sender_click)
    link_hover_count: int = 0
    sender_click_count: int = 0
    
    # Self-reported (from action modal)
    confidence_rating: int  # 1-10
    suspicion_rating: int  # 1-10
    
    @classmethod
    def calculate_correct_response(cls, is_phishing: bool, action: str) -> int:
        """
        Calculate if the response was correct.
        
        Correct responses:
        - Phishing email → report or delete
        - Legitimate email → safe or ignore
        """
        if is_phishing:
            return 1 if action in ["report", "delete"] else 0
        else:
            return 1 if action in ["safe", "ignore"] else 0