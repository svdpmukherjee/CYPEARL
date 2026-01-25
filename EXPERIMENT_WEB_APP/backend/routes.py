"""
CYPEARL Experiment Web App - API Routes (UPDATED v5 - Survey Collections)

UPDATES IN THIS VERSION (v5):
1. Pre-survey responses stored in dedicated 'pre_survey_responses' collection
2. Post-survey responses stored in dedicated 'post_survey_responses' collection
3. New export endpoints: /export/pre-survey, /export/post-survey
4. S3 export endpoints for survey data: /admin/s3/export-pre-survey, /admin/s3/export-post-survey

MongoDB Collections:
- participants: User sessions with embedded survey data
- emails: Email stimuli for the experiment
- responses: Per-email observational data
- pre_survey_responses: Dedicated pre-experiment survey responses (NEW)
- post_survey_responses: Dedicated post-experiment survey responses (NEW)
- logs: Audit/activity logs

PREVIOUS CHANGES (v4):
1. Complete redesign for LuxConsultancy factorial experiment
2. New known domains: luxconsultancy.com, securenebula.com, lockgrid.com,
   superfinance.com, trendyletter.com, greenenvi.com, wattvoltbridge.com
3. Phishing domain detection for spoofed versions
4. Added comprehension check endpoint

PREVIOUS CHANGES (v3):
1. sender_hover replaced with sender_click tracking
2. Silent bonus calculation (no feedback to frontend)
3. Bonus revealed only at end via /complete endpoint
4. Fixed duplicate event logging with deduplication
"""

from fastapi import APIRouter, HTTPException, Body, Depends, Request
from typing import List, Dict, Any, Optional
from models import Email, UserAction, Participant
from database import db
from bson import ObjectId
from datetime import datetime, timedelta
from pydantic import BaseModel

# S3 Export service for automated data pipeline
from services.s3_export import s3_exporter

router = APIRouter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fix_id(doc):
    """Convert MongoDB _id to string id"""
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc


def extract_factorial_fields(email: dict) -> dict:
    """Extract factorial design fields from email metadata."""
    if not email:
        return {}
        
    factorial = email.get("factorial_category", {}) or {}
    
    email_type = "phishing" if email.get("is_phishing") else "legitimate"
    
    # Updated sender mapping for known/unknown design
    sender_map = {
        "known": "known",
        "unknown": "unknown",
        "known_internal": "known",
        "known_external": "known", 
        "unknown_external": "unknown",
    }
    sender_familiarity = sender_map.get(
        str(factorial.get("sender", "")).lower(), 
        email.get("sender_familiarity", "unknown")
    )
    
    urgency_level = str(factorial.get("urgency", "low")).lower()
    if urgency_level not in ["high", "low"]:
        urgency_level = "high" if "urgent" in email.get("subject", "").lower() else "low"
    
    framing_type = str(factorial.get("framing", "neutral")).lower()
    if framing_type not in ["threat", "reward"]:
        framing_type = "threat"
    
    return {
        "email_type": email_type,
        "sender_familiarity": sender_familiarity,
        "urgency_level": urgency_level,
        "framing_type": framing_type,
        # Note: has_aggressive_content removed - use framing_type for emotional manipulation analysis
        "phishing_quality": email.get("phishing_quality") if email.get("is_phishing") else None,
        "ground_truth": 1 if email.get("is_phishing") else 0
    }


def calculate_correct_response(is_phishing: bool, action: str) -> int:
    """Calculate if the response was correct."""
    if is_phishing:
        return 1 if action in ["report", "delete"] else 0
    else:
        return 1 if action in ["safe", "ignore"] else 0


# =============================================================================
# BONUS CALCULATION HELPERS (Silent - no user feedback)
# UPDATED v4: LuxConsultancy Design
# =============================================================================

# Known trusted domains for LuxConsultancy
KNOWN_TRUSTED_DOMAINS = [
    'luxconsultancy.com',
    'securenebula.com',
    'lockgrid.com',
    'superfinance.com',
    'trendyletter.com',
    'greenenvi.com',
    'wattvoltbridge.com',
]

# All legitimate domains (known + unknown legitimate external)
LEGITIMATE_DOMAINS = [
    # Known trusted domains (LuxConsultancy partners)
    'luxconsultancy.com',
    'securenebula.com',
    'lockgrid.com',
    'superfinance.com',
    'trendyletter.com',
    'greenenvi.com',
    'wattvoltbridge.com',
    # Academy/training subdomains
    'academy.securenebula.com',
    'dashboard.securenebula.com',
    'portal.lockgrid.com',
    'docs.greenenvi.com',
]

# Phishing domains (spoofed known domains + unknown phishing)
PHISHING_DOMAINS = [
    # Spoofed "known" domains (hyphenation, TLD swap, typosquatting)
    'secure-nebula.com',           # Email 1: hyphenation
    'securenebula-verify.net',     # Phishing URL
    'superfinance.org',            # Email 3: TLD swap
    'superfinance-refunds.com',    # Phishing URL
    'lockgrid.net',                # Email 9: TLD swap
    'lockgrid-portal.net',         # Phishing URL
    'wattvoltbrdige.com',          # Email 11: typosquatting
    'wattvoltbridge-partners.com', # Phishing URL
    # Unknown phishing domains
    'eurobank-secure.net',         # Email 5: fake bank
    'eurobank-verify.com',
    'luxprizes-eu.com',            # Email 7: fake awards
    'lux-business-awards.com',
    'it-compliance-eu.net',        # Email 13: fake compliance
    'gdpr-audit-portal.eu',
    'eu-survey-rewards.com',       # Email 15: fake survey
    'business-research-eu.com',
]

BONUS_REWARDS = {
    'legitimate': {
        'default': 30,
        # Known trusted partners
        'lockgrid': 40,
        'securenebula': 35,
        'superfinance': 40,
        'greenenvi': 30,
        'wattvoltbridge': 35,
        'trendyletter': 25,
        # Unknown legitimate
        'cssf': 45,               # Government portal - higher value
        'ecb': 50,                # ECB conference - high value
        'cnpd': 35,               # Data protection authority
        'paperjam': 30,           # Business networking
    },
    'phishing': {
        'default': -80,
        # Spoofed known domains (should've been caught)
        'secure-nebula': -90,     # Hyphenation attack
        'superfinance_org': -95,  # TLD swap for financial
        'lockgrid_net': -85,      # TLD swap
        'wattvoltbrdige': -75,    # Typosquatting (harder to spot)
        # Unknown phishing
        'eurobank': -100,         # Fake bank - most severe
        'luxprizes': -80,         # Fake awards
        'compliance_eu': -85,     # Fake compliance audit
        'survey_rewards': -70,    # Survey scam
    }
}


def calculate_link_bonus(link: str) -> tuple:
    """
    Calculate bonus/loss for a link click.
    Returns (amount, link_type) where link_type is 'legitimate' or 'phishing'
    """
    if not link:
        return (0, 'unknown')
    
    link_lower = link.lower()
    
    # Check if legitimate
    for domain in LEGITIMATE_DOMAINS:
        if domain in link_lower:
            # Determine specific reward
            if 'lockgrid.com' in link_lower:
                return (BONUS_REWARDS['legitimate']['lockgrid'], 'legitimate')
            elif 'securenebula.com' in link_lower:
                return (BONUS_REWARDS['legitimate']['securenebula'], 'legitimate')
            elif 'superfinance.com' in link_lower:
                return (BONUS_REWARDS['legitimate']['superfinance'], 'legitimate')
            elif 'greenenvi.com' in link_lower:
                return (BONUS_REWARDS['legitimate']['greenenvi'], 'legitimate')
            elif 'wattvoltbridge.com' in link_lower:
                return (BONUS_REWARDS['legitimate']['wattvoltbridge'], 'legitimate')
            elif 'trendyletter.com' in link_lower:
                return (BONUS_REWARDS['legitimate']['trendyletter'], 'legitimate')
            elif 'cssf.lu' in link_lower or 'portal.cssf.lu' in link_lower:
                return (BONUS_REWARDS['legitimate']['cssf'], 'legitimate')
            elif 'ecb.europa.eu' in link_lower:
                return (BONUS_REWARDS['legitimate']['ecb'], 'legitimate')
            elif 'cnpd' in link_lower:
                return (BONUS_REWARDS['legitimate']['cnpd'], 'legitimate')
            elif 'paperjam' in link_lower:
                return (BONUS_REWARDS['legitimate']['paperjam'], 'legitimate')
            else:
                return (BONUS_REWARDS['legitimate']['default'], 'legitimate')
    
    # Check if phishing
    for domain in PHISHING_DOMAINS:
        if domain in link_lower:
            # Determine specific penalty
            if 'secure-nebula' in link_lower or 'securenebula-verify' in link_lower:
                return (BONUS_REWARDS['phishing']['secure-nebula'], 'phishing')
            elif 'superfinance.org' in link_lower or 'superfinance-refunds' in link_lower:
                return (BONUS_REWARDS['phishing']['superfinance_org'], 'phishing')
            elif 'lockgrid.net' in link_lower or 'lockgrid-portal.net' in link_lower:
                return (BONUS_REWARDS['phishing']['lockgrid_net'], 'phishing')
            elif 'wattvoltbrdige' in link_lower or 'wattvoltbridge-partners' in link_lower:
                return (BONUS_REWARDS['phishing']['wattvoltbrdige'], 'phishing')
            elif 'eurobank' in link_lower:
                return (BONUS_REWARDS['phishing']['eurobank'], 'phishing')
            elif 'luxprizes' in link_lower or 'lux-business-awards' in link_lower:
                return (BONUS_REWARDS['phishing']['luxprizes'], 'phishing')
            elif 'it-compliance-eu' in link_lower or 'gdpr-audit-portal' in link_lower:
                return (BONUS_REWARDS['phishing']['compliance_eu'], 'phishing')
            elif 'eu-survey-rewards' in link_lower or 'business-research-eu' in link_lower:
                return (BONUS_REWARDS['phishing']['survey_rewards'], 'phishing')
            else:
                return (BONUS_REWARDS['phishing']['default'], 'phishing')
    
    # Unknown - no bonus change
    return (0, 'unknown')


# =============================================================================
# PRE-SURVEY ENDPOINT (Save immediately on completion)
# =============================================================================

class PreSurveyRequest(BaseModel):
    session_id: str  # Temporary session ID (generated by frontend)
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None
    pre_survey_data: Dict[str, Any]


@router.post("/survey/pre")
async def submit_pre_survey(body: PreSurveyRequest, request: Request):
    """
    Save pre-experiment survey data immediately when user completes it.
    Called BEFORE the Prolific ID is entered - uses temporary session_id.
    The session_id will be linked to participant_id later during /auth/login.
    """
    now = datetime.now()

    # Check if this session already has a pre-survey record
    existing = await db.pre_survey_responses.find_one({"session_id": body.session_id})
    if existing:
        # Update existing record
        await db.pre_survey_responses.update_one(
            {"session_id": body.session_id},
            {"$set": {
                "updated_at": now,
                "user_agent": body.user_agent or request.headers.get("user-agent"),
                "screen_resolution": body.screen_resolution,
                **body.pre_survey_data
            }}
        )
        return {"status": "updated", "session_id": body.session_id}

    # Create new pre-survey record
    pre_survey_record = {
        "session_id": body.session_id,
        "participant_id": None,  # Will be linked during /auth/login
        "prolific_id": None,  # Will be linked during /auth/login
        "submitted_at": now,
        "user_agent": body.user_agent or request.headers.get("user-agent"),
        "screen_resolution": body.screen_resolution,
        **body.pre_survey_data
    }
    await db.pre_survey_responses.insert_one(pre_survey_record)

    # Log the submission
    await db.logs.insert_one({
        "session_id": body.session_id,
        "action_type": "pre_survey_submitted_early",
        "timestamp": now,
        "details": {"fields_count": len(body.pre_survey_data)}
    })

    return {"status": "created", "session_id": body.session_id}


# =============================================================================
# AUTH ENDPOINTS
# =============================================================================

class LoginRequest(BaseModel):
    prolific_id: str
    session_id: Optional[str] = None  # Link to pre-survey if submitted early
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None
    pre_survey_data: Optional[Dict[str, Any]] = None  # Pre-experiment survey data (fallback)


@router.post("/auth/login")
async def login(body: LoginRequest, request: Request):
    """Create new participant session with pre-experiment survey data."""
    # Create initial timestamp for the welcome email (order_id=0)
    welcome_email = await db.emails.find_one({"order_id": 0})
    email_timestamps = {}

    if welcome_email:
        email_timestamps[str(welcome_email["_id"])] = datetime.now()

    now = datetime.now()

    new_participant = {
        "prolific_id": body.prolific_id,
        "current_email_order": 0,
        "completed": False,
        "email_timestamps": email_timestamps,
        "read_email_ids": [],
        "deleted_email_ids": [],
        "email_sessions": {},
        "bonus_total": 0,  # Track total bonus (hidden from user)
        "bonus_history": [],  # Track bonus changes
        "comprehension_check_passed": False,  # Track if comprehension check completed
        "comprehension_check_attempts": 0,
        "started_at": now,
        "user_agent": body.user_agent or request.headers.get("user-agent"),
        "screen_resolution": body.screen_resolution,
        # Pre-experiment survey data (all validated instruments)
        "pre_survey_data": body.pre_survey_data,
        "post_survey_data": None  # Will be filled after email experiment
    }

    result = await db.participants.insert_one(new_participant)
    participant_id = str(result.inserted_id)

    # Link pre-survey data to this participant
    # First check if there's already a pre-survey record with the session_id
    pre_survey_linked = False

    if body.session_id:
        # Try to link existing pre-survey record to this participant
        update_result = await db.pre_survey_responses.update_one(
            {"session_id": body.session_id},
            {"$set": {
                "participant_id": participant_id,
                "prolific_id": body.prolific_id,
                "linked_at": now
            }}
        )
        if update_result.modified_count > 0:
            pre_survey_linked = True
            await db.logs.insert_one({
                "participant_id": participant_id,
                "action_type": "pre_survey_linked",
                "timestamp": now,
                "details": {"session_id": body.session_id, "prolific_id": body.prolific_id}
            })

    # If no pre-survey was linked via session_id, try to create from body.pre_survey_data
    if not pre_survey_linked:
        has_pre_survey = body.pre_survey_data and len(body.pre_survey_data) > 0

        if has_pre_survey:
            pre_survey_record = {
                "participant_id": participant_id,
                "prolific_id": body.prolific_id,
                "submitted_at": now,
                "user_agent": body.user_agent or request.headers.get("user-agent"),
                "screen_resolution": body.screen_resolution,
                **body.pre_survey_data  # Flatten all survey fields to top level
            }
            await db.pre_survey_responses.insert_one(pre_survey_record)

            # Log the submission
            await db.logs.insert_one({
                "participant_id": participant_id,
                "action_type": "pre_survey_submitted",
                "timestamp": now,
                "details": {"prolific_id": body.prolific_id, "fields_count": len(body.pre_survey_data)}
            })
        else:
            # Log that no pre-survey data was received (for debugging)
            await db.logs.insert_one({
                "participant_id": participant_id,
                "action_type": "pre_survey_missing",
                "timestamp": now,
                "details": {
                    "prolific_id": body.prolific_id,
                    "session_id": body.session_id,
                    "pre_survey_data_received": body.pre_survey_data is not None,
                    "pre_survey_data_type": str(type(body.pre_survey_data))
            }
        })

    return {"participant_id": participant_id}


# =============================================================================
# COMPREHENSION CHECK ENDPOINT
# =============================================================================

class ComprehensionCheckRequest(BaseModel):
    answers: Dict[str, int]  # {"cc1": 1, "cc2": 1, "cc3": 2}


CORRECT_ANSWERS = {
    "cc1": 1,  # lockgrid.com
    "cc2": 1,  # Unknown to LuxConsultancy
    "cc3": 2,  # Could be legitimate or malicious
}


@router.post("/comprehension-check/{participant_id}")
async def validate_comprehension_check(participant_id: str, body: ComprehensionCheckRequest):
    """Validate comprehension check answers."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    # Calculate score
    correct_count = 0
    total = len(CORRECT_ANSWERS)
    results = {}
    
    for q_id, correct_answer in CORRECT_ANSWERS.items():
        user_answer = body.answers.get(q_id)
        is_correct = user_answer == correct_answer
        if is_correct:
            correct_count += 1
        results[q_id] = {
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        }
    
    passed = correct_count >= 2  # Allow 1 mistake
    
    # Update participant record
    await db.participants.update_one(
        {"_id": ObjectId(participant_id)},
        {
            "$set": {"comprehension_check_passed": passed},
            "$inc": {"comprehension_check_attempts": 1}
        }
    )
    
    # Log the attempt
    await db.logs.insert_one({
        "participant_id": participant_id,
        "action_type": "comprehension_check",
        "timestamp": datetime.now(),
        "details": {
            "answers": body.answers,
            "results": results,
            "score": f"{correct_count}/{total}",
            "passed": passed
        }
    })
    
    return {
        "passed": passed,
        "score": correct_count,
        "total": total,
        "results": results,
        "message": "Passed! You can now begin the study." if passed else "Please review the instructions and try again."
    }


@router.get("/comprehension-check/status/{participant_id}")
async def get_comprehension_check_status(participant_id: str):
    """Check if participant has passed comprehension check."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    return {
        "passed": participant.get("comprehension_check_passed", False),
        "attempts": participant.get("comprehension_check_attempts", 0)
    }


# =============================================================================
# POST-EXPERIMENT SURVEY ENDPOINT
# =============================================================================

class PostSurveyRequest(BaseModel):
    state_anxiety: Optional[float] = 0.0
    current_stress: Optional[float] = 0.0
    fatigue_level: Optional[int] = 4  # Default to middle of 1-7 scale
    raw_responses: Optional[Dict[str, Any]] = None
    completed_at: Optional[str] = None


@router.post("/survey/post/{participant_id}")
async def submit_post_survey(participant_id: str, body: PostSurveyRequest):
    """
    Submit post-experiment survey data (STAI-6, PSS-4, Fatigue).
    Called after participant finishes all 16 emails.

    Triggers S3 export in background after survey is saved.
    """
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")

    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    now = datetime.now()

    post_survey_data = {
        "state_anxiety": body.state_anxiety,
        "current_stress": body.current_stress,
        "fatigue_level": body.fatigue_level,
        "raw_responses": body.raw_responses,
        "completed_at": body.completed_at or now.isoformat()
    }

    # Update participant with post-survey data
    await db.participants.update_one(
        {"_id": ObjectId(participant_id)},
        {
            "$set": {
                "post_survey_data": post_survey_data,
                "study_completed_at": now
            }
        }
    )

    # Store post-survey data in a dedicated collection for easier querying/export
    try:
        post_survey_record = {
            "participant_id": participant_id,
            "prolific_id": participant.get("prolific_id"),
            "submitted_at": now,
            "state_anxiety": body.state_anxiety or 0.0,
            "current_stress": body.current_stress or 0.0,
            "fatigue_level": body.fatigue_level or 4,
            "raw_responses": body.raw_responses,
            "study_started_at": participant.get("started_at"),
            "study_completed_at": now,
            "bonus_total": participant.get("bonus_total", 0)
        }
        await db.post_survey_responses.insert_one(post_survey_record)

        # Log the submission
        await db.logs.insert_one({
            "participant_id": participant_id,
            "action_type": "post_survey_submitted",
            "timestamp": now,
            "details": post_survey_data
        })
    except Exception as e:
        # Log error but don't fail - the participant data was already updated
        await db.logs.insert_one({
            "participant_id": participant_id,
            "action_type": "post_survey_collection_error",
            "timestamp": now,
            "details": {"error": str(e), "post_survey_data": post_survey_data}
        })

    # Export to S3 (run synchronously since it's fast and we want to ensure it completes)
    # For truly async operation, could use Celery or similar task queue
    try:
        export_result = await s3_exporter.export_participant_completion(participant_id, db)
        await db.logs.insert_one({
            "participant_id": participant_id,
            "action_type": "s3_export",
            "timestamp": datetime.now(),
            "details": export_result
        })
    except Exception as e:
        # Log error but don't fail the response - data is still in MongoDB
        await db.logs.insert_one({
            "participant_id": participant_id,
            "action_type": "s3_export_error",
            "timestamp": datetime.now(),
            "details": {"error": str(e)}
        })

    return {"status": "success", "message": "Post-experiment survey saved"}


# =============================================================================
# SILENT BONUS CALCULATION ENDPOINT
# =============================================================================

class BonusCalculateRequest(BaseModel):
    link: str
    email_id: str
    timestamp: Optional[str] = None


@router.post("/bonus/calculate/{participant_id}")
async def calculate_bonus_silent(participant_id: str, body: BonusCalculateRequest):
    """
    Calculate and store bonus silently - NO feedback to user.
    User will only see their bonus at the end of the study.
    
    IMPORTANT: Bonus is only calculated ONCE per email_id per participant.
    Subsequent clicks on the same email will be recorded but won't affect bonus.
    """
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    # CHECK FOR DUPLICATE: Has bonus already been calculated for this email?
    bonus_history = participant.get("bonus_history", [])
    already_clicked_emails = {entry.get("email_id") for entry in bonus_history if entry.get("email_id")}
    
    if body.email_id in already_clicked_emails:
        # Bonus already calculated for this email - don't add again
        await db.logs.insert_one({
            "participant_id": participant_id,
            "action_type": "duplicate_link_click",
            "timestamp": datetime.now(),
            "details": {
                "email_id": body.email_id,
                "link": body.link,
                "message": "Bonus already calculated for this email"
            }
        })
        return {"status": "already_clicked", "message": "Link already clicked for this email"}
    
    # Calculate bonus
    bonus_amount, link_type = calculate_link_bonus(body.link)
    
    if bonus_amount == 0:
        # No change for unknown links
        return {"status": "recorded", "link_type": "unknown"}
    
    now = datetime.now()
    current_total = participant.get("bonus_total", 0)
    new_total = current_total + bonus_amount
    
    bonus_entry = {
        "amount": bonus_amount,
        "total": new_total,
        "timestamp": now,
        "link_clicked": body.link,
        "email_id": body.email_id,
        "link_type": link_type
    }
    
    # Update participant bonus (silently)
    await db.participants.update_one(
        {"_id": ObjectId(participant_id)},
        {
            "$set": {"bonus_total": new_total},
            "$push": {"bonus_history": bonus_entry}
        }
    )
    
    # Log to main logs collection
    await db.logs.insert_one({
        "participant_id": participant_id,
        "action_type": "bonus_update",
        "timestamp": now,
        "details": bonus_entry
    })
    
    return {"status": "recorded", "first_click": True}


@router.get("/bonus/{participant_id}")
async def get_bonus(participant_id: str):
    """
    Get participant's current bonus.
    This should ONLY be called at the end of the study!
    """
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    # Only return bonus if study is completed
    if not participant.get("completed", False):
        return {"status": "study_in_progress", "message": "Bonus will be revealed at the end"}
    
    return {
        "bonus_total": participant.get("bonus_total", 0),
        "bonus_history": participant.get("bonus_history", [])
    }


# =============================================================================
# EMAIL ENDPOINTS
# =============================================================================

@router.get("/emails/inbox/{participant_id}")
async def get_inbox(participant_id: str, folder: str = "inbox"):
    """Get emails for a participant's inbox."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    current_order = participant.get("current_email_order", 0)
    deleted_ids = participant.get("deleted_email_ids", [])
    read_ids = participant.get("read_email_ids", [])
    
    # Get emails up to current order
    if folder == "deleted":
        # For deleted folder, show only deleted emails
        if not deleted_ids:
            return {"emails": [], "counts": {"unread": 0, "deleted": 0}, "is_finished": False}
        
        deleted_object_ids = [ObjectId(eid) for eid in deleted_ids if ObjectId.is_valid(eid)]
        emails_cursor = db.emails.find({"_id": {"$in": deleted_object_ids}})
    else:
        # For inbox, exclude deleted emails
        emails_cursor = db.emails.find({
            "order_id": {"$lte": current_order}
        }).sort("order_id", -1)
    
    emails = []
    # Get previously stored timestamps
    email_timestamps = participant.get("email_timestamps", {})
    
    async for email in emails_cursor:
        email = fix_id(email)
        email_id = email.get("id")
        
        # Skip deleted emails in inbox view
        if folder != "deleted" and email_id in deleted_ids:
            continue
            
        email["is_read"] = email_id in read_ids
        
        # Attach stored timestamp if available, else fallback
        if email_id in email_timestamps:
            email["timestamp"] = email_timestamps[email_id]
        else:
            # Fallback for old sessions or missing data
            email["timestamp"] = participant.get("started_at", datetime.now())
            
        emails.append(email)
    
    # Calculate counts
    all_emails = await db.emails.find({"order_id": {"$lte": current_order}}).to_list(length=100)
    unread_count = sum(1 for e in all_emails 
                      if str(e["_id"]) not in read_ids 
                      and str(e["_id"]) not in deleted_ids)
    
    # Check if study is finished
    max_email = await db.emails.find_one(sort=[("order_id", -1)])
    max_order = max_email["order_id"] if max_email else 0
    is_finished = current_order > max_order
    
    return {
        "emails": emails,
        "counts": {
            "unread": unread_count,
            "deleted": len(deleted_ids)
        },
        "is_finished": is_finished
    }


@router.get("/emails/{email_id}")
async def get_email(email_id: str):
    """Get a single email by ID."""
    if not ObjectId.is_valid(email_id):
        raise HTTPException(status_code=404, detail="Invalid email ID")
    
    email = await db.emails.find_one({"_id": ObjectId(email_id)})
    if not email:
        raise HTTPException(status_code=404, detail="Email not found")
    
    # Get participant to find timestamp (need to modify this endpoint to optionally accept participant_id context)
    # But for now, just return the email. The frontend might need to rely on the list view content 
    # OR we need to update this endpoint.
    # Actually, ReadingPane uses the props passed from the list, so it might already have the timestamp.
    # But if it fetches fresh, we need it. 
    # Let's check frontend: ReadingPane takes 'email' prop. EmailList fetches 'emails' which we updated above.
    # So ReadingPane should be fine if it uses the object from the list.
    # However, if we refresh, we might need it.
    # As the current signature doesn't have participant_id, we can't easily look up the specific user's timestamp.
    # Front-end typically uses the list data.
    
    return fix_id(email)


# =============================================================================
# SESSION TRACKING ENDPOINTS
# =============================================================================

@router.post("/session/open/{participant_id}")
async def open_email_session(participant_id: str, email_id: str = Body(..., embed=True)):
    """Record when participant opens an email."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    now = datetime.now()
    
    # Initialize session data
    session_data = {
        "opened_at": now,
        "closed_at": None,
        "clicked_link": False,
        "clicked_links": [],
        "hovered_link": False,
        "link_hover_count": 0,
        "total_link_hover_time_ms": 0,
        "inspected_sender": False,
        "sender_click_count": 0,
    }
    
    result = await db.participants.update_one(
        {"_id": ObjectId(participant_id)},
        {"$set": {f"email_sessions.{email_id}": session_data}}
    )
    
    await db.logs.insert_one({
        "participant_id": participant_id,
        "email_id": email_id,
        "action_type": "email_open",
        "timestamp": now
    })
    
    return {"status": "session_opened", "opened_at": now.isoformat()}


class MicroActionRequest(BaseModel):
    """Request model for micro-actions (hover, click, sender inspection)"""
    email_id: str
    action_type: str  # link_hover, link_click, sender_click
    link: Optional[str] = None
    duration_ms: Optional[int] = None
    timestamp: Optional[str] = None


@router.post("/session/micro-action/{participant_id}")
async def record_micro_action(participant_id: str, body: MicroActionRequest):
    """Record micro-actions during email viewing."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    now = datetime.now()
    session_key = f"email_sessions.{body.email_id}"
    
    update_ops = {}
    
    if body.action_type == "link_hover":
        update_ops = {
            "$set": {f"{session_key}.hovered_link": True},
            "$inc": {
                f"{session_key}.link_hover_count": 1,
                f"{session_key}.total_link_hover_time_ms": body.duration_ms or 0
            }
        }
    elif body.action_type == "link_click":
        update_ops = {
            "$set": {f"{session_key}.clicked_link": True},
            "$push": {f"{session_key}.clicked_links": body.link}
        }
    elif body.action_type == "sender_click":
        update_ops = {
            "$set": {f"{session_key}.inspected_sender": True},
            "$inc": {f"{session_key}.sender_click_count": 1}
        }
    
    if update_ops:
        await db.participants.update_one(
            {"_id": ObjectId(participant_id)},
            update_ops
        )
    
    # Log the micro-action
    await db.logs.insert_one({
        "participant_id": participant_id,
        "email_id": body.email_id,
        "action_type": body.action_type,
        "timestamp": now,
        "details": {
            "link": body.link,
            "duration_ms": body.duration_ms
        }
    })
    
    return {"status": "recorded"}


# =============================================================================
# FINAL ACTION ENDPOINTS
# =============================================================================

class FinalActionRequest(BaseModel):
    """Request model for final email action."""
    email_id: str
    action_type: str  # safe, report, delete, ignore
    confidence: Optional[int] = None
    suspicion: Optional[int] = None
    reason: Optional[str] = None
    latency_ms: Optional[int] = None
    dwell_time_ms: Optional[int] = None
    clicked_link: Optional[bool] = False
    hovered_link: Optional[bool] = False
    inspected_sender: Optional[bool] = False
    link_hover_count: Optional[int] = 0
    sender_click_count: Optional[int] = 0
    sender_hover_count: Optional[int] = None  # Legacy field
    client_info: Optional[Dict[str, Any]] = None


def safe_int(val, default=0):
    """Safely convert value to integer."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


@router.post("/action/{participant_id}")
async def submit_action(participant_id: str, action: FinalActionRequest):
    """Submit final action on an email."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    now = datetime.now()
    
    # Handle mark_read action separately
    if action.action_type == "mark_read":
        await db.participants.update_one(
            {"_id": ObjectId(participant_id)},
            {"$addToSet": {"read_email_ids": action.email_id}}
        )
        
        await db.logs.insert_one({
            "participant_id": participant_id,
            "email_id": action.email_id,
            "action_type": "mark_read",
            "timestamp": now
        })
        
        return {"status": "marked_read"}
    
    # Safe type conversions
    confidence = safe_int(action.confidence, 5)
    suspicion = safe_int(action.suspicion, 5)
    latency_ms = safe_int(action.latency_ms, 0)
    dwell_time_ms = safe_int(action.dwell_time_ms, 0)
    
    # Support both sender_click_count (new) and sender_hover_count (old)
    sender_count = action.sender_click_count or action.sender_hover_count or 0
    
    # Get email metadata for response record
    email = None
    if ObjectId.is_valid(action.email_id):
        email = await db.emails.find_one({"_id": ObjectId(action.email_id)})
    
    factorial_fields = extract_factorial_fields(email) if email else {}
    is_phishing = email.get("is_phishing", False) if email else False
    correct_response = calculate_correct_response(is_phishing, action.action_type)
    
    # Get session data
    sessions = participant.get("email_sessions", {})
    session = sessions.get(action.email_id, {})
    
    # Build response record for CSV export
    response_record = {
        "participant_id": participant_id,
        "email_id": action.email_id,
        
        # Factorial fields
        **factorial_fields,
        
        # Action
        "action": action.action_type,
        
        # Binary outcomes
        "clicked": 1 if (action.clicked_link or session.get("clicked_link")) else 0,
        "reported": 1 if action.action_type == "report" else 0,
        "ignored": 1 if action.action_type == "ignore" else 0,
        "deleted": 1 if action.action_type == "delete" else 0,
        "marked_safe": 1 if action.action_type == "safe" else 0,
        "correct_response": correct_response,
        
        # Timing
        "response_latency_ms": latency_ms,
        "dwell_time_ms": dwell_time_ms,
        
        # Behavioral indicators
        "hovered_link": 1 if (action.hovered_link or session.get("hovered_link")) else 0,
        "inspected_sender": 1 if (action.inspected_sender or session.get("inspected_sender")) else 0,
        "link_hover_count": action.link_hover_count or session.get("link_hover_count", 0),
        "sender_click_count": sender_count or session.get("sender_click_count", session.get("sender_hover_count", 0)),
        
        # Self-reported
        "confidence_rating": confidence,
        "suspicion_rating": suspicion,
        "reason": action.reason,
        
        # Metadata
        "timestamp": now,
        "client_info": action.client_info
    }
    
    # Store in responses collection
    await db.responses.insert_one(response_record)
    
    # Log the action
    await db.logs.insert_one({
        "participant_id": participant_id,
        "email_id": action.email_id,
        "action_type": action.action_type,
        "timestamp": now,
        "details": {
            "confidence": action.confidence,
            "suspicion": action.suspicion,
            "reason": action.reason,
            "latency_ms": action.latency_ms,
            "dwell_time_ms": action.dwell_time_ms,
            "correct": correct_response
        }
    })
    
    # Close the email session
    await db.participants.update_one(
        {"_id": ObjectId(participant_id)},
        {"$set": {f"email_sessions.{action.email_id}.closed_at": now}}
    )
    
    # Handle delete action
    if action.action_type == "delete":
        await db.participants.update_one(
            {"_id": ObjectId(participant_id)},
            {"$push": {"deleted_email_ids": action.email_id}}
        )
    
    return {"status": "success", "correct": correct_response}


# =============================================================================
# PROGRESS ENDPOINTS
# =============================================================================

@router.post("/complete/{participant_id}")
async def advance_to_next(participant_id: str):
    """Advance participant to the next email."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    result = await db.participants.update_one(
        {"_id": ObjectId(participant_id)},
        {"$inc": {"current_email_order": 1}}
    )
    
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    # Generate timestamp for the NEXT email (which is now current_order + 1, wait logic says incremented)
    # We just incremented current_email_order. So we need the email with that new order.
    
    # Get updated participant to get new order
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    new_order = participant.get("current_email_order", 0)
    
    # Find the email for this new order
    next_email = await db.emails.find_one({"order_id": new_order})
    
    if next_email:
        # Store timestamp for this new email
        await db.participants.update_one(
            {"_id": ObjectId(participant_id)},
            {"$set": {f"email_timestamps.{str(next_email['_id'])}": datetime.now()}}
        )
    
    # Check if study is complete
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    current_order = participant.get("current_email_order", 0)
    
    max_order_email = await db.emails.find_one(sort=[("order_id", -1)])
    max_order_id = max_order_email["order_id"] if max_order_email else 0
    
    if current_order > max_order_id:
        # Mark as completed
        await db.participants.update_one(
            {"_id": ObjectId(participant_id)},
            {"$set": {"completed": True, "completed_at": datetime.now()}}
        )
        
        # Now user can see their bonus
        bonus_total = participant.get("bonus_total", 0)
        return {
            "status": "completed",
            "message": "Study completed! Thank you for your participation.",
            "bonus_total": bonus_total,
            "bonus_history": participant.get("bonus_history", [])
        }
    
    return {"status": "advanced", "current_order": current_order}


@router.get("/progress/{participant_id}")
async def get_progress(participant_id: str):
    """Get participant's current progress."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    max_order_email = await db.emails.find_one(sort=[("order_id", -1)])
    max_order_id = max_order_email["order_id"] if max_order_email else 0
    
    current_order = participant.get("current_email_order", 0)
    
    response = {
        "current_email_order": current_order,
        "total_emails": max_order_id,
        "completed": participant.get("completed", False),
        "started_at": participant.get("started_at"),
        "completed_at": participant.get("completed_at"),
        "comprehension_check_passed": participant.get("comprehension_check_passed", False)
    }
    
    # Only include bonus if completed
    if participant.get("completed", False):
        response["bonus_total"] = participant.get("bonus_total", 0)
    
    return response


# =============================================================================
# DATA EXPORT
# =============================================================================

@router.get("/export/responses")
async def export_responses():
    """Export all email responses for analysis."""
    responses = await db.responses.find({}).to_list(length=100000)

    # Clean up MongoDB ObjectIds
    for r in responses:
        if "_id" in r:
            r["_id"] = str(r["_id"])
        if "timestamp" in r:
            r["timestamp"] = r["timestamp"].isoformat()

    return {"responses": responses, "count": len(responses)}


@router.get("/export/pre-survey")
async def export_pre_survey_responses():
    """Export all pre-experiment survey responses for analysis."""
    responses = await db.pre_survey_responses.find({}).to_list(length=10000)

    # Clean up MongoDB ObjectIds and dates
    for r in responses:
        if "_id" in r:
            r["_id"] = str(r["_id"])
        if "submitted_at" in r and r["submitted_at"]:
            r["submitted_at"] = r["submitted_at"].isoformat()
        if "completed_at" in r and r["completed_at"]:
            r["completed_at"] = r["completed_at"].isoformat()

    return {"pre_survey_responses": responses, "count": len(responses)}


@router.get("/export/post-survey")
async def export_post_survey_responses():
    """Export all post-experiment survey responses for analysis."""
    responses = await db.post_survey_responses.find({}).to_list(length=10000)

    # Clean up MongoDB ObjectIds and dates
    for r in responses:
        if "_id" in r:
            r["_id"] = str(r["_id"])
        if "submitted_at" in r and r["submitted_at"]:
            r["submitted_at"] = r["submitted_at"].isoformat()
        if "study_started_at" in r and r["study_started_at"]:
            r["study_started_at"] = r["study_started_at"].isoformat()
        if "study_completed_at" in r and r["study_completed_at"]:
            r["study_completed_at"] = r["study_completed_at"].isoformat()

    return {"post_survey_responses": responses, "count": len(responses)}


@router.get("/export/participants")
async def export_participants():
    """Export participant data including survey data and bonus totals."""
    participants = await db.participants.find({}).to_list(length=10000)

    for p in participants:
        if "_id" in p:
            p["participant_id"] = str(p["_id"])
            del p["_id"]
        if "started_at" in p and p["started_at"]:
            p["started_at"] = p["started_at"].isoformat()
        if "completed_at" in p and p["completed_at"]:
            p["completed_at"] = p["completed_at"].isoformat()

        # Flatten pre_survey_data into participant record
        pre_survey = p.get("pre_survey_data", {}) or {}
        if pre_survey:
            # Remove nested raw_responses to keep export clean
            pre_survey_flat = {k: v for k, v in pre_survey.items() if k != 'raw_responses'}
            p.update(pre_survey_flat)

        # Flatten post_survey_data into participant record
        post_survey = p.get("post_survey_data", {}) or {}
        if post_survey:
            post_survey_flat = {k: v for k, v in post_survey.items() if k != 'raw_responses'}
            p.update(post_survey_flat)

    return {"participants": participants, "count": len(participants)}


@router.get("/export/participants/csv")
async def export_participants_csv_format():
    """
    Export participant data in CSV-ready format matching phishing_study_participants.csv schema.
    All survey scores are flattened to top-level fields.
    """
    participants = await db.participants.find({}).to_list(length=10000)
    responses = await db.responses.find({}).to_list(length=100000)

    # Group responses by participant
    responses_by_participant = {}
    for r in responses:
        pid = r.get("participant_id")
        if pid not in responses_by_participant:
            responses_by_participant[pid] = []
        responses_by_participant[pid].append(r)

    csv_records = []

    for p in participants:
        pid = str(p["_id"])
        pre_survey = p.get("pre_survey_data", {}) or {}
        post_survey = p.get("post_survey_data", {}) or {}

        # Calculate aggregated outcomes from responses
        participant_responses = responses_by_participant.get(pid, [])
        total_emails = len(participant_responses)

        if total_emails > 0:
            correct_count = sum(1 for r in participant_responses if r.get("correct_response") == 1)
            phishing_responses = [r for r in participant_responses if r.get("ground_truth") == 1]
            legit_responses = [r for r in participant_responses if r.get("ground_truth") == 0]

            phishing_correct = sum(1 for r in phishing_responses if r.get("correct_response") == 1)
            phishing_clicked = sum(1 for r in phishing_responses if r.get("clicked") == 1)
            legit_false_positive = sum(1 for r in legit_responses if r.get("action") in ["report", "delete"])

            overall_accuracy = correct_count / total_emails
            phishing_detection_rate = phishing_correct / len(phishing_responses) if phishing_responses else 0
            phishing_click_rate = phishing_clicked / len(phishing_responses) if phishing_responses else 0
            false_positive_rate = legit_false_positive / len(legit_responses) if legit_responses else 0
            report_rate = sum(1 for r in participant_responses if r.get("action") == "report") / total_emails

            mean_response_latency = sum(r.get("response_latency_ms", 0) for r in participant_responses) / total_emails
            mean_dwell_time = sum(r.get("dwell_time_ms", 0) for r in participant_responses) / total_emails
            latencies = [r.get("response_latency_ms", 0) for r in participant_responses]
            response_latency_sd = (sum((x - mean_response_latency) ** 2 for x in latencies) / total_emails) ** 0.5

            hover_rate = sum(1 for r in participant_responses if r.get("hovered_link") == 1) / total_emails
            sender_inspection_rate = sum(1 for r in participant_responses if r.get("inspected_sender") == 1) / total_emails

            mean_confidence = sum(r.get("confidence_rating", 5) for r in participant_responses) / total_emails
            mean_suspicion_phishing = sum(r.get("suspicion_rating", 5) for r in phishing_responses) / len(phishing_responses) if phishing_responses else 0
            mean_suspicion_legit = sum(r.get("suspicion_rating", 5) for r in legit_responses) / len(legit_responses) if legit_responses else 0
        else:
            overall_accuracy = phishing_detection_rate = phishing_click_rate = 0
            false_positive_rate = report_rate = 0
            mean_response_latency = mean_dwell_time = response_latency_sd = 0
            hover_rate = sender_inspection_rate = 0
            mean_confidence = mean_suspicion_phishing = mean_suspicion_legit = 0

        record = {
            "participant_id": pid,
            # Demographics
            "age": pre_survey.get("age"),
            "gender": pre_survey.get("gender"),
            "education": pre_survey.get("education"),
            "education_numeric": pre_survey.get("education_numeric"),
            "technical_field": pre_survey.get("technical_field"),
            "employment": pre_survey.get("employment"),
            "industry": pre_survey.get("industry"),
            # Cognitive
            "crt_score": pre_survey.get("crt_score"),
            "need_for_cognition": pre_survey.get("need_for_cognition"),
            "working_memory": pre_survey.get("working_memory"),
            # Big Five
            "big5_extraversion": pre_survey.get("big5_extraversion"),
            "big5_agreeableness": pre_survey.get("big5_agreeableness"),
            "big5_conscientiousness": pre_survey.get("big5_conscientiousness"),
            "big5_neuroticism": pre_survey.get("big5_neuroticism"),
            "big5_openness": pre_survey.get("big5_openness"),
            # Personality
            "impulsivity_total": pre_survey.get("impulsivity_total"),
            "sensation_seeking": pre_survey.get("sensation_seeking"),
            "trust_propensity": pre_survey.get("trust_propensity"),
            "risk_taking": pre_survey.get("risk_taking"),
            # State (post-experiment)
            "state_anxiety": post_survey.get("state_anxiety"),
            "current_stress": post_survey.get("current_stress"),
            "fatigue_level": post_survey.get("fatigue_level"),
            # Security attitudes
            "phishing_self_efficacy": pre_survey.get("phishing_self_efficacy"),
            "perceived_risk": pre_survey.get("perceived_risk"),
            "security_attitudes": pre_survey.get("security_attitudes"),
            "privacy_concern": pre_survey.get("privacy_concern"),
            # Knowledge & experience
            "phishing_knowledge": pre_survey.get("phishing_knowledge"),
            "technical_expertise": pre_survey.get("technical_expertise"),
            "prior_victimization": pre_survey.get("prior_victimization"),
            "security_training": pre_survey.get("security_training"),
            "years_email_use": pre_survey.get("years_email_use"),
            # Email habits
            "daily_email_volume": pre_survey.get("daily_email_volume"),
            "email_volume_numeric": pre_survey.get("email_volume_numeric"),
            "email_check_frequency": pre_survey.get("email_check_frequency"),
            "link_click_tendency": pre_survey.get("link_click_tendency"),
            "social_media_usage": pre_survey.get("social_media_usage"),
            # Influence susceptibility
            "authority_susceptibility": pre_survey.get("authority_susceptibility"),
            "urgency_susceptibility": pre_survey.get("urgency_susceptibility"),
            "scarcity_susceptibility": pre_survey.get("scarcity_susceptibility"),
            # Aggregated outcomes
            "overall_accuracy": round(overall_accuracy, 3),
            "phishing_detection_rate": round(phishing_detection_rate, 3),
            "phishing_click_rate": round(phishing_click_rate, 3),
            "false_positive_rate": round(false_positive_rate, 3),
            "report_rate": round(report_rate, 3),
            "mean_response_latency": round(mean_response_latency),
            "mean_dwell_time": round(mean_dwell_time),
            "response_latency_sd": round(response_latency_sd),
            "hover_rate": round(hover_rate, 3),
            "sender_inspection_rate": round(sender_inspection_rate, 3),
            "mean_confidence": round(mean_confidence, 2),
            "mean_suspicion_phishing": round(mean_suspicion_phishing, 2),
            "mean_suspicion_legit": round(mean_suspicion_legit, 2)
        }

        csv_records.append(record)

    return {"participants": csv_records, "count": len(csv_records)}


@router.get("/export/emails")
async def export_emails():
    """Export all email stimuli for analysis."""
    emails = await db.emails.find({}).to_list(length=100)

    for e in emails:
        e = fix_id(e)
        if "timestamp" in e and e["timestamp"]:
            e["timestamp"] = e["timestamp"].isoformat()

    return {"emails": emails, "count": len(emails)}


# =============================================================================
# S3 EXPORT ADMIN ENDPOINTS
# =============================================================================

@router.post("/admin/s3/export-emails")
async def admin_export_emails_to_s3():
    """
    Admin endpoint: Export email stimuli to S3 (run once at experiment setup).
    This creates the static email_stimuli.csv in the S3 bucket.
    """
    result = await s3_exporter.export_email_stimuli(db)
    return result


@router.post("/admin/s3/export-participant/{participant_id}")
async def admin_export_participant_to_s3(participant_id: str):
    """
    Admin endpoint: Manually export a specific participant's data to S3.
    Useful for re-exporting if there was an error.
    """
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")

    result = await s3_exporter.export_participant_completion(participant_id, db)
    return result


@router.post("/admin/s3/export-pre-survey")
async def admin_export_pre_survey_to_s3():
    """
    Admin endpoint: Export all pre-survey responses to S3.
    Creates pre_survey_responses.csv in the S3 bucket.
    """
    result = await s3_exporter.export_pre_survey_responses(db)
    return result


@router.post("/admin/s3/export-post-survey")
async def admin_export_post_survey_to_s3():
    """
    Admin endpoint: Export all post-survey responses to S3.
    Creates post_survey_responses.csv in the S3 bucket.
    """
    result = await s3_exporter.export_post_survey_responses(db)
    return result


@router.get("/admin/s3/status")
async def admin_s3_status():
    """
    Admin endpoint: Check S3 export service status and configuration.
    Returns detailed status including export tracking metadata.
    """
    return s3_exporter.get_status()


@router.post("/admin/s3/reconcile")
async def admin_s3_reconcile(force_all: bool = False):
    """
    Admin endpoint: Batch reconcile MongoDB data with S3.

    This ensures S3 has all completed participant data by:
    1. Finding all completed participants in MongoDB
    2. Exporting any not yet in S3 (incremental)
    3. Creating a snapshot in the archive folder

    Query params:
        force_all: If true, rebuild all CSVs from scratch (full sync)

    Use cases:
        - Initial setup: Run with force_all=true to populate S3
        - Daily sync: Run without force_all to catch any missed exports
        - Recovery: Run with force_all=true after data issues
    """
    result = await s3_exporter.batch_reconcile(db, force_all=force_all)
    return result


@router.post("/admin/s3/archive-snapshot")
async def admin_create_snapshot():
    """
    Admin endpoint: Create a manual snapshot of current CSVs.

    Copies current processed CSVs to archive folder with timestamp.
    Useful before making changes or for periodic backups.
    """
    if not s3_exporter.enabled:
        return {"status": "skipped", "reason": "S3 not configured"}

    try:
        result = await s3_exporter._create_archive_snapshot()
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "reason": str(e)}