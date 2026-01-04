"""
CYPEARL Experiment Web App - API Routes (UPDATED v3 - FIXED)

FIXES IN THIS VERSION:
1. Updated domain lists for new factorial design emails 5-8
2. Added new legitimate domains: impots.public.lu, cc.lu
3. Added new phishing domains: domain-renewal-alert.com, eu-business-excellence.com
4. Updated bonus calculation to handle new domain patterns

PREVIOUS CHANGES (v2):
1. sender_hover replaced with sender_click tracking
2. Silent bonus calculation (no feedback to frontend)
3. Bonus revealed only at end via /complete endpoint
4. Fixed duplicate event logging with deduplication
5. All tracking still works, just no visible feedback during study

Updated endpoints:
- POST /bonus/calculate/{participant_id} - Silent bonus calculation (no response data)
- GET /bonus/{participant_id} - Only available at end of study
- POST /session/micro-action - Now handles sender_click
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
    
    sender_map = {
        "known_internal": "known_internal",
        "known_external": "known_external", 
        "unknown_external": "unknown_external",
        "internal": "known_internal",
        "external": "unknown_external"
    }
    sender_familiarity = sender_map.get(
        str(factorial.get("sender", "")).lower(), 
        "unknown_external"
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
# UPDATED v3: Added new domains for fixed factorial design emails 5-8
# =============================================================================

# URL patterns to determine if link is legitimate or phishing
LEGITIMATE_DOMAINS = [
    # Internal company domains (Lux Consultancy)
    'luxconsultancy.com',
    'budget.luxconsultancy.com',
    'training.luxconsultancy.com',
    'employee-portal.luxconsultancy.com',
    # Known external service domains
    'nexuscloud.com',
    'console.nexuscloud.com',
    'techconf.io',
    'secureshield.com',
    'portal.secureshield.com',
    'cybersecuritytoday.com',
    # Unknown external legitimate domains (for emails 6, 8)
    'eurofinance-tax.eu',       # EuroFinance Tax Services (Email #6)
    'luxbusiness-network.eu',   # LuxBusiness Network (Email #8)
]

PHISHING_DOMAINS = [
    # Known internal impersonation
    'secure-payment-gateway.net',
    'hr-rewards-luxconsultancy.com',
    'luxconsultancy-executive.com',
    'luxconsultancy-hr.com',
    'security-update-portal.com',
    'researchconnect-premium-studies.com',
    # Known external impersonation
    'nexuscloud-verify.com',
    'prolink-verify.com',
    # Unknown external phishing
    'cloudvault-secure.net',
    'prof-dev-courses.com',
    # Unknown external phishing domains (for emails 5, 7)
    'domaineguard-alerts.com',     # Fake domain registry (Email #5)
    'globalbiz-excellence.com',    # Fake business awards (Email #7)
]

BONUS_REWARDS = {
    'legitimate': {
        'default': 30,
        'budget': 50,
        'training': 35,
        'portal': 40,
        'techconf': 25,
        'newsletter': 20,
        'tax': 30,      # Tax portal (Email #6)
        'summit': 35,   # Chamber of Commerce event (Email #8)
    },
    'phishing': {
        'default': -80,
        'payment': -100,
        'verify': -90,
        'credential': -85,
        'premium': -70,
        'domain': -90,   # Domain scam (Email #5)
        'award': -85,    # Fake award scam (Email #7)
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
            if 'budget' in link_lower:
                return (BONUS_REWARDS['legitimate']['budget'], 'legitimate')
            elif 'training' in link_lower:
                return (BONUS_REWARDS['legitimate']['training'], 'legitimate')
            elif 'techconf' in link_lower:
                return (BONUS_REWARDS['legitimate']['techconf'], 'legitimate')
            elif 'newsletter' in link_lower or 'cybersecurity' in link_lower:
                return (BONUS_REWARDS['legitimate']['newsletter'], 'legitimate')
            elif 'eurofinance' in link_lower or 'tax' in link_lower or 'declarations' in link_lower:
                return (BONUS_REWARDS['legitimate']['tax'], 'legitimate')
            elif 'luxbusiness' in link_lower or 'summit' in link_lower or 'digital-summit' in link_lower:
                return (BONUS_REWARDS['legitimate']['summit'], 'legitimate')
            elif 'portal' in link_lower or 'employee-portal' in link_lower:
                return (BONUS_REWARDS['legitimate']['portal'], 'legitimate')
            else:
                return (BONUS_REWARDS['legitimate']['default'], 'legitimate')
    
    # Check if phishing
    for domain in PHISHING_DOMAINS:
        if domain in link_lower:
            if 'payment' in link_lower:
                return (BONUS_REWARDS['phishing']['payment'], 'phishing')
            elif 'verify' in link_lower:
                return (BONUS_REWARDS['phishing']['verify'], 'phishing')
            elif 'premium' in link_lower:
                return (BONUS_REWARDS['phishing']['premium'], 'phishing')
            elif 'domaineguard' in link_lower or 'renewal' in link_lower or 'renew' in link_lower:
                return (BONUS_REWARDS['phishing']['domain'], 'phishing')
            elif 'globalbiz' in link_lower or 'excellence' in link_lower or 'nomination' in link_lower:
                return (BONUS_REWARDS['phishing']['award'], 'phishing')
            elif 'password' in link_lower or 'credential' in link_lower:
                return (BONUS_REWARDS['phishing']['credential'], 'phishing')
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
    new_participant = {
        "prolific_id": body.prolific_id,
        "current_email_order": 0,
        "completed": False,
        "email_timestamps": {},
        "read_email_ids": [],
        "deleted_email_ids": [],
        "email_sessions": {},
        "bonus_total": 0,  # Track total bonus (hidden from user)
        "bonus_history": [],  # Track bonus changes
        "started_at": datetime.now(),
        "user_agent": body.user_agent or request.headers.get("user-agent"),
        "screen_resolution": body.screen_resolution
    }
    
    result = await db.participants.insert_one(new_participant)
    return {"participant_id": str(result.inserted_id)}


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
    """
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    # Calculate bonus
    bonus_amount, link_type = calculate_link_bonus(body.link)
    
    if bonus_amount == 0:
        # No change for unknown links
        return {"status": "recorded"}
    
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
    
    # Return minimal response - NO bonus info to frontend
    return {"status": "recorded"}


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
        # Don't reveal bonus during study
        return {"status": "study_in_progress", "message": "Bonus will be revealed at the end"}
    
    return {
        "bonus_total": participant.get("bonus_total", 0),
        "bonus_history": participant.get("bonus_history", [])
    }


# =============================================================================
# EMAIL ENDPOINTS
# =============================================================================

@router.get("/emails/inbox/{participant_id}", response_model=Dict[str, Any])
async def get_inbox(participant_id: str, folder: str = "inbox"):
    """Get participant's inbox with email state. NO bonus info returned."""
    
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")

    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    current_order = participant.get("current_email_order", 0)
    email_timestamps = participant.get("email_timestamps", {})
    read_email_ids = participant.get("read_email_ids", [])
    deleted_email_ids = participant.get("deleted_email_ids", [])
    
    if folder == "inbox":
        query = {"$or": [{"order_id": {"$lte": current_order}}, {"order_id": 0}]}
    elif folder == "deleted":
        query = {"_id": {"$in": [ObjectId(eid) for eid in deleted_email_ids]}}
    else:
        return {"emails": [], "counts": {"unread": 0, "deleted": len(deleted_email_ids)}}

    cursor = db.emails.find(query).sort("order_id", -1)
    emails = await cursor.to_list(length=100)
    
    result_emails = []
    for email in emails:
        email_id = str(email["_id"])
        order_id_str = str(email["order_id"])
        
        if folder == "inbox" and email_id in deleted_email_ids:
            continue
            
        if order_id_str in email_timestamps:
            email["timestamp"] = email_timestamps[order_id_str]
        else:
            now = datetime.now()
            email["timestamp"] = now
            await db.participants.update_one(
                {"_id": ObjectId(participant_id)},
                {"$set": {f"email_timestamps.{order_id_str}": now}}
            )
            
        email["is_read"] = email_id in read_email_ids
        result_emails.append(fix_id(email))
    
    inbox_query = {"$or": [{"order_id": {"$lte": current_order}}, {"order_id": 0}]}
    all_inbox_emails = await db.emails.find(inbox_query).to_list(length=1000)
    
    unread_count = sum(
        1 for e in all_inbox_emails 
        if str(e["_id"]) not in deleted_email_ids and str(e["_id"]) not in read_email_ids
    )
    
    max_order_email = await db.emails.find_one(sort=[("order_id", -1)])
    max_order_id = max_order_email["order_id"] if max_order_email else 0
    is_finished = current_order > max_order_id

    # NO bonus in response - user doesn't see it during study
    return {
        "emails": result_emails,
        "counts": {"unread": unread_count, "deleted": len(deleted_email_ids)},
        "is_finished": is_finished
    }


# =============================================================================
# SESSION TRACKING (with deduplication)
# =============================================================================

class EmailOpenRequest(BaseModel):
    email_id: str


@router.post("/session/open/{participant_id}")
async def open_email_session(participant_id: str, body: EmailOpenRequest):
    """Record when participant opens an email. Prevents duplicate opens."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    email_id = body.email_id
    now = datetime.now()
    
    # Check if session already exists for this email
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if participant:
        existing_sessions = participant.get("email_sessions", {})
        if email_id in existing_sessions:
            # Session already exists, don't create duplicate
            return {"status": "exists", "message": "Session already open for this email"}
    
    session_data = {
        "email_id": email_id,
        "opened_at": now,
        "closed_at": None,
        "clicked_link": False,
        "hovered_link": False,
        "inspected_sender": False,  # Now tracked via sender_click
        "link_hover_count": 0,
        "sender_click_count": 0
    }
    
    await db.participants.update_one(
        {"_id": ObjectId(participant_id)},
        {"$set": {f"email_sessions.{email_id}": session_data}}
    )
    
    # Only log email_open once per email per participant
    existing_open = await db.logs.find_one({
        "participant_id": participant_id,
        "email_id": email_id,
        "action_type": "email_open"
    })
    
    if not existing_open:
        await db.logs.insert_one({
            "participant_id": participant_id,
            "email_id": email_id,
            "action_type": "email_open",
            "timestamp": now
        })
    
    return {"status": "success", "session_started": now.isoformat()}


class MicroActionRequest(BaseModel):
    email_id: str
    action_type: str
    details: Optional[Dict[str, Any]] = None


@router.post("/session/micro-action/{participant_id}")
async def record_micro_action(participant_id: str, body: MicroActionRequest):
    """Record micro-interactions within an email. With deduplication."""
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    email_id = body.email_id
    action_type = body.action_type
    details = body.details or {}
    now = datetime.now()
    
    # Deduplication: Check if same action was logged in last 500ms
    cutoff_time = now - timedelta(milliseconds=500)
    recent_action = await db.logs.find_one({
        "participant_id": participant_id,
        "email_id": email_id,
        "action_type": action_type,
        "timestamp": {"$gte": cutoff_time}
    })
    
    if recent_action:
        # Skip duplicate
        return {"status": "skipped", "reason": "duplicate_action"}
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    sessions = participant.get("email_sessions", {}) if participant else {}
    session = sessions.get(email_id, {})
    
    updates = {}
    
    if action_type == "link_hover":
        duration = details.get("duration_ms", details.get("duration", 0))
        updates[f"email_sessions.{email_id}.hovered_link"] = True
        updates[f"email_sessions.{email_id}.link_hover_count"] = session.get("link_hover_count", 0) + 1
        
    elif action_type == "link_click":
        updates[f"email_sessions.{email_id}.clicked_link"] = True
        
    elif action_type == "sender_click":
        # NEW: Track sender click instead of hover
        updates[f"email_sessions.{email_id}.inspected_sender"] = True
        updates[f"email_sessions.{email_id}.sender_click_count"] = session.get("sender_click_count", 0) + 1
    
    if updates:
        await db.participants.update_one({"_id": ObjectId(participant_id)}, {"$set": updates})
    
    await db.logs.insert_one({
        "participant_id": participant_id,
        "email_id": email_id,
        "action_type": action_type,
        "timestamp": now,
        "details": details
    })
    
    return {"status": "success", "action_recorded": action_type}


# =============================================================================
# ACTION SUBMISSION
# =============================================================================

class ActionSubmission(BaseModel):
    email_id: str
    action_type: str
    reason: Optional[str] = None
    confidence: Optional[Any] = None  # Accept any type, convert later
    suspicion: Optional[Any] = None   # Accept any type, convert later
    latency_ms: Optional[Any] = None
    dwell_time_ms: Optional[Any] = None
    hover_data: Optional[Dict[str, Any]] = None
    client_info: Optional[Dict[str, Any]] = None
    clicked_link: Optional[bool] = None
    hovered_link: Optional[bool] = None
    inspected_sender: Optional[bool] = None
    link_hover_count: Optional[int] = None
    sender_click_count: Optional[int] = None  # New name
    sender_hover_count: Optional[int] = None  # Backward compatible
    
    class Config:
        extra = "ignore"  # Ignore any extra fields not defined


def safe_int(value, default=0):
    """Safely convert value to int, handling strings and None."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


@router.post("/actions/{participant_id}")
async def submit_action(participant_id: str, action: ActionSubmission, request: Request):
    """Submit action on an email."""
    now = datetime.now()
    
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")
    
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    # Handle mark_read action
    if action.action_type == "mark_read":
        read_ids = participant.get("read_email_ids", [])
        if action.email_id not in read_ids:
            await db.participants.update_one(
                {"_id": ObjectId(participant_id)},
                {"$push": {"read_email_ids": action.email_id}}
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
        "completed_at": participant.get("completed_at")
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