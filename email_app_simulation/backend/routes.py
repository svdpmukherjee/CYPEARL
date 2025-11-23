from fastapi import APIRouter, HTTPException, Body, Depends, Request
from typing import List, Dict, Any
from models import Email, UserAction, Participant
from database import db
from bson import ObjectId

router = APIRouter()

# Helper to fix ObjectId serialization
def fix_id(doc):
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

from datetime import datetime

from pydantic import BaseModel

class LoginRequest(BaseModel):
    prolific_id: str

@router.post("/auth/login")
async def login(body: LoginRequest):
    # Create new participant
    # We store the prolific_id for reference
    new_participant = {
        "prolific_id": body.prolific_id,
        "current_email_order": 0,
        "completed": False,
        "email_timestamps": {},
        "read_email_ids": [],
        "deleted_email_ids": []
    }
    
    result = await db.participants.insert_one(new_participant)
    return {"participant_id": str(result.inserted_id)}

@router.get("/emails/inbox/{participant_id}", response_model=Dict[str, Any])
async def get_inbox(participant_id: str, folder: str = "inbox"):
    # Validate ObjectId format
    if not ObjectId.is_valid(participant_id):
        raise HTTPException(status_code=404, detail="Invalid participant ID")

    # Find participant
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    current_order = participant.get("current_email_order", 0)
    email_timestamps = participant.get("email_timestamps", {})
    read_email_ids = participant.get("read_email_ids", [])
    deleted_email_ids = participant.get("deleted_email_ids", [])
    
    # Query logic based on folder
    query = {}
    if folder == "inbox":
        query = {
            "$or": [
                {"order_id": {"$lte": current_order}},
                {"order_id": 0}
            ]
        }
    elif folder == "deleted":
        # For deleted folder, we only show emails that are in deleted_email_ids
        # But we still respect the current_order (can't see future emails even if deleted somehow)
        query = {
             "_id": {"$in": [ObjectId(eid) for eid in deleted_email_ids]}
        }
    else:
        # Other folders empty for now
        return {
            "emails": [],
            "counts": {"unread": 0, "deleted": len(deleted_email_ids)}
        }

    cursor = db.emails.find(query).sort("order_id", -1)
    emails = await cursor.to_list(length=100)
    
    result_emails = []
    for email in emails:
        email_id = str(email["_id"])
        order_id_str = str(email["order_id"])
        
        # Filter for Inbox: Exclude deleted emails
        if folder == "inbox" and email_id in deleted_email_ids:
            continue
            
        # Inject Timestamp
        if order_id_str in email_timestamps:
            email["timestamp"] = email_timestamps[order_id_str]
        else:
            # Lazy init timestamp if missing (should be set on completion of previous)
            # For order 0 and 1 (initial), set to now if missing
            now = datetime.now()
            email["timestamp"] = now
            # Update DB so it persists
            await db.participants.update_one(
                {"_id": ObjectId(participant_id)},
                {"$set": {f"email_timestamps.{order_id_str}": now}}
            )
            
        # Inject Read Status
        email["is_read"] = email_id in read_email_ids
        
        result_emails.append(fix_id(email))
        
    # Calculate counts
    # Unread in inbox: emails in inbox (order_id <= current) that are NOT in read_email_ids
    # Deleted: length of deleted_email_ids
    
    # We need to count unread emails in the inbox context
    # This requires querying for all inbox emails first
    inbox_query = {
        "$or": [
            {"order_id": {"$lte": current_order}},
            {"order_id": 0}
        ]
    }
    all_inbox_cursor = db.emails.find(inbox_query)
    all_inbox_emails = await all_inbox_cursor.to_list(length=1000)
    
    unread_count = 0
    for e in all_inbox_emails:
        eid = str(e["_id"])
        if eid not in deleted_email_ids and eid not in read_email_ids:
            unread_count += 1
            
    deleted_count = len(deleted_email_ids)

    # Calculate is_finished
    # Find the max order_id in the emails collection
    max_order_email = await db.emails.find_one(sort=[("order_id", -1)])
    max_order_id = max_order_email["order_id"] if max_order_email else 0
    
    # If current_order is greater than max_order_id, we are done
    # Note: current_email_order is incremented AFTER the last email is completed
    is_finished = current_order > max_order_id

    return {
        "emails": result_emails,
        "counts": {
            "unread": unread_count,
            "deleted": deleted_count
        },
        "is_finished": is_finished
    }

@router.post("/actions/{participant_id}")
async def submit_action(participant_id: str, action: UserAction, request: Request):
    # Log the action
    action_dict = action.dict()
    action_dict["participant_id"] = participant_id
    
    # 1. Enrich with Email Snapshot (Quality Data)
    if action.email_id:
        try:
            email = await db.emails.find_one({"_id": ObjectId(action.email_id)})
            if email:
                action_dict["email_snapshot"] = {
                    "is_phishing": email.get("is_phishing"),
                    "order_id": email.get("order_id"),
                    "sender_email": email.get("sender_email"),
                    "sender_name": email.get("sender_name"),
                    "factorial_category": email.get("factorial_category"),
                    "tactics": email.get("tactics"),
                    "aggression_level": email.get("aggression_level")
                }
        except Exception as e:
            print(f"Error fetching email snapshot: {e}")

    # 2. Enrich with Client Info (User Agent)
    if not action_dict.get("client_info"):
        action_dict["client_info"] = {}
    
    action_dict["client_info"]["user_agent"] = request.headers.get("user-agent")
    action_dict["client_info"]["ip_address"] = request.client.host

    await db.logs.insert_one(action_dict)
    
    # Get current participant state (create if doesn't exist)
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        # Create participant if it doesn't exist (for tracking actions before inbox is loaded)
        try:
            new_id = ObjectId(participant_id)
            await db.participants.insert_one({
                "_id": new_id,
                "current_email_order": 0,
                "completed": False,
                "email_timestamps": {},
                "read_email_ids": [],
                "deleted_email_ids": []
            })
            participant = {
                "_id": new_id,
                "current_email_order": 0,
                "email_timestamps": {},
                "read_email_ids": [],
                "deleted_email_ids": []
            }
        except Exception as e:
            print(f"Error creating participant in actions endpoint: {e}")
            # If we can't create the participant, just log the action and return
            return {"status": "success", "message": "Action logged"}
    
    # Handle specific actions that update state
    updates = {}
    
    if action.action_type == "delete":
        updates["$addToSet"] = {"deleted_email_ids": action.email_id}
        
    elif action.action_type == "mark_read":
        updates["$addToSet"] = {"read_email_ids": action.email_id}
        
    # Note: We do NOT increment current_email_order here anymore.
    # That happens in /complete endpoint.
    
    if updates:
        await db.participants.update_one(
            {"_id": ObjectId(participant_id)},
            updates
        )
    
    return {"status": "success", "message": "Action recorded"}

@router.post("/complete/{participant_id}")
async def complete_email(participant_id: str):
    participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    current_order = participant.get("current_email_order", 1)
    next_order = current_order + 1
    
    # Set timestamp for the NEXT email
    now = datetime.now()
    
    await db.participants.update_one(
        {"_id": ObjectId(participant_id)},
        {
            "$inc": {"current_email_order": 1},
            "$set": {f"email_timestamps.{next_order}": now}
        }
    )
    
    return {"status": "success", "message": "Advanced to next email"}

@router.post("/seed")
async def seed_emails():
    # Helper to seed some dummy emails
    await db.emails.delete_many({})
    await db.participants.delete_many({})
    
    emails = [
        {
            "sender_name": "HR Department",
            "sender_email": "hr@company-update.com", # Phishing indicator
            "subject": "Urgent: Update your payroll info",
            "body": "Please click here to update your details immediately.",
            "is_phishing": True,
            "order_id": 1
        },
        {
            "sender_name": "Team Lead",
            "sender_email": "lead@company.com",
            "subject": "Project Update",
            "body": "Here are the slides for tomorrow's meeting.",
            "is_phishing": False,
            "order_id": 2
        }
    ]
    
    await db.emails.insert_many(emails)
    return {"message": "Database seeded"}
