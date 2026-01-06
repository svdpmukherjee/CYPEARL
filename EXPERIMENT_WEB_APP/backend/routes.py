"""
CYPEARL Experiment Web App - API Routes (UPDATED v4 - LuxConsultancy Design)

UPDATES IN THIS VERSION:
1. Complete redesign for LuxConsultancy factorial experiment
2. New known domains: luxconsultancy.com, securenebula.com, lockgrid.com,
   superfinance.com, trendyletter.com, greenenvi.com, wattvoltbridge.com
3. Phishing domain detection for spoofed versions:
   - Hyphenation: secure-nebula.com
   - TLD swaps: superfinance.org, lockgrid.net
   - Typosquatting: wattvoltbrdige.com
4. Unknown legitimate domains: cssf.lu, ecb.europa.eu, cnpd.lu, paperjam.lu
5. Unknown phishing domains: eurobank-secure.net, luxprizes-eu.com, etc.
6. Added comprehension check endpoint

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
        "has_aggressive_content": email.get("aggression_level", "low") in ["high", "very_high"],
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
# AUTH ENDPOINTS
# =============================================================================

class LoginRequest(BaseModel):
    prolific_id: str
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None


@router.post("/auth/login")
async def login(body: LoginRequest, request: Request):
    """Create new participant session."""
    # Create initial timestamp for the welcome email (order_id=0)
    welcome_email = await db.emails.find_one({"order_id": 0})
    email_timestamps = {}
    
    if welcome_email:
        email_timestamps[str(welcome_email["_id"])] = datetime.now()

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
        "started_at": datetime.now(),
        "user_agent": body.user_agent or request.headers.get("user-agent"),
        "screen_resolution": body.screen_resolution
    }
    
    result = await db.participants.insert_one(new_participant)
    return {"participant_id": str(result.inserted_id)}


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
    """Export all responses for analysis."""
    responses = await db.responses.find({}).to_list(length=100000)
    
    # Clean up MongoDB ObjectIds
    for r in responses:
        if "_id" in r:
            r["_id"] = str(r["_id"])
        if "timestamp" in r:
            r["timestamp"] = r["timestamp"].isoformat()
    
    return {"responses": responses, "count": len(responses)}


@router.get("/export/participants")
async def export_participants():
    """Export participant data including bonus totals."""
    participants = await db.participants.find({}).to_list(length=10000)
    
    for p in participants:
        if "_id" in p:
            p["id"] = str(p["_id"])
            del p["_id"]
        if "started_at" in p and p["started_at"]:
            p["started_at"] = p["started_at"].isoformat()
        if "completed_at" in p and p["completed_at"]:
            p["completed_at"] = p["completed_at"].isoformat()
    
    return {"participants": participants, "count": len(participants)}


@router.get("/export/emails")
async def export_emails():
    """Export all email stimuli for analysis."""
    emails = await db.emails.find({}).to_list(length=100)
    
    for e in emails:
        e = fix_id(e)
        if "timestamp" in e and e["timestamp"]:
            e["timestamp"] = e["timestamp"].isoformat()
    
    return {"emails": emails, "count": len(emails)}