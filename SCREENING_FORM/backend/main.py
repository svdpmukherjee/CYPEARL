"""
CYPEARL Stage-1 Pre-Screening API

Collects job-role context and email typology data from Prolific participants
to improve ecological validity of Stage-2 phishing simulation emails.

Endpoints:
  POST /users               – register a Prolific ID at consent time
  PUT  /draft/{pid}         – upsert a partial form snapshot
  GET  /draft/{pid}         – retrieve a partial form snapshot
  POST /submit              – finalize a screening response
  GET  /responses           – list all responses (admin)
  GET  /responses/{pid}     – fetch one response by prolific_id
  GET  /health              – liveness check
"""

import os
from datetime import datetime, timezone
from typing import Optional

import certifi
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, model_validator

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise ValueError("MONGO_URL not found in environment variables")

DB_NAME = os.getenv("SCREENING_DB_NAME", "cypearl_screening")


def parse_allowed_origins(raw_origins: Optional[str]) -> list[str]:
    # Browsers send Origin without trailing slash; normalize env entries to match.
    if not raw_origins:
        return [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
    ]

    parsed = []
    for origin in raw_origins.split(","):
        cleaned = origin.strip().rstrip("/")
        if cleaned:
            parsed.append(cleaned)

    return parsed or [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
    ]


ALLOWED_ORIGINS = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))

client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
db = client[DB_NAME]
collection = db["responses"]
users_collection = db["users"]
drafts_collection = db["drafts"]

app = FastAPI(title="CYPEARL Pre-Screening", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────

VALID_FREQUENCIES = {"Daily", "Weekly", "Monthly", "Rarely"}
VALID_SENDER_TYPES = {"internal", "external"}
MANDATORY_EMAILS = 5
BONUS_PER_EMAIL_PENCE = 5
MAX_BONUS_PENCE = 50


class EmailEntry(BaseModel):
    subject: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    frequency: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def check_frequency(self):
        if self.frequency not in VALID_FREQUENCIES:
            raise ValueError(f"frequency must be one of {VALID_FREQUENCIES}")
        return self


class SenderEntry(BaseModel):
    role: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    emails: list[EmailEntry] = Field(..., min_items=1)

    @model_validator(mode="after")
    def check_type(self):
        if self.type not in VALID_SENDER_TYPES:
            raise ValueError(f"type must be one of {VALID_SENDER_TYPES}")
        return self


class GenericEmailEntry(BaseModel):
    sender: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class UserRegistration(BaseModel):
    prolific_id: str = Field(..., min_length=1)


class ScreeningResponse(BaseModel):
    prolific_id: str = Field(..., min_length=1)
    job_cluster: str = Field(..., min_length=1)
    job_title: str = Field(..., min_length=1)
    daily_tasks: str = Field(..., min_length=1)
    email_senders: list[SenderEntry] = Field(..., min_items=1)
    generic_emails: list[GenericEmailEntry] = Field(default_factory=list)
    suspicious_emails: list[str] = Field(default_factory=list)


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/users")
async def register_user(payload: UserRegistration):
    """Called when a participant enters their Prolific ID and ticks consent.
    Idempotent: if the user already exists, only `last_active_at` is updated.
    Refuses participants who have already submitted (returns 409)."""
    now = datetime.now(timezone.utc)

    existing_response = await collection.find_one(
        {"prolific_id": payload.prolific_id}, {"_id": 1}
    )
    if existing_response:
        raise HTTPException(
            status_code=409,
            detail="A response with this Prolific ID has already been submitted.",
        )

    await users_collection.update_one(
        {"prolific_id": payload.prolific_id},
        {
            "$set": {"last_active_at": now, "status": "in_progress"},
            "$setOnInsert": {
                "prolific_id": payload.prolific_id,
                "consented_at": now,
                "status": "in_progress",
            },
        },
        upsert=True,
    )
    return {"status": "ok"}


@app.put("/draft/{prolific_id}")
async def upsert_draft(prolific_id: str, draft: dict):
    """Upserts a partial form snapshot. The draft is opaque JSON — the schema
    is enforced only on final /submit. Refuses if the user already submitted."""
    existing_response = await collection.find_one(
        {"prolific_id": prolific_id}, {"_id": 1}
    )
    if existing_response:
        raise HTTPException(
            status_code=409,
            detail="A response with this Prolific ID has already been submitted.",
        )

    now = datetime.now(timezone.utc)
    await drafts_collection.update_one(
        {"prolific_id": prolific_id},
        {
            "$set": {"draft": draft, "updated_at": now},
            "$setOnInsert": {"prolific_id": prolific_id, "created_at": now},
        },
        upsert=True,
    )
    await users_collection.update_one(
        {"prolific_id": prolific_id},
        {"$set": {"last_active_at": now, "status": "in_progress"}},
        upsert=True,
    )
    return {"status": "ok"}


@app.get("/draft/{prolific_id}")
async def get_draft(prolific_id: str):
    doc = await drafts_collection.find_one({"prolific_id": prolific_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="No draft found")
    return doc


@app.post("/submit")
async def submit_response(data: ScreeningResponse):
    doc = data.model_dump()
    total_email_count = sum(len(sender.emails) for sender in data.email_senders)
    bonus_emails = max(0, total_email_count - MANDATORY_EMAILS)
    bonus_pence_raw = bonus_emails * BONUS_PER_EMAIL_PENCE
    bonus_pence = min(bonus_pence_raw, MAX_BONUS_PENCE)
    now = datetime.now(timezone.utc)
    doc["submitted_at"] = now
    doc["user_agent"] = ""
    doc["total_email_count"] = total_email_count
    doc["bonus_emails"] = bonus_emails
    doc["bonus_pence"] = bonus_pence

    existing = await collection.find_one({"prolific_id": data.prolific_id})
    if existing:
        raise HTTPException(
            status_code=409,
            detail="Response already submitted for this Prolific ID.",
        )

    await collection.insert_one(doc)
    await users_collection.update_one(
        {"prolific_id": data.prolific_id},
        {"$set": {"status": "submitted", "submitted_at": now}},
        upsert=True,
    )
    await drafts_collection.delete_one({"prolific_id": data.prolific_id})
    return {"status": "ok", "message": "Thank you! Your response has been recorded."}


@app.get("/responses")
async def list_responses():
    cursor = collection.find({}, {"_id": 0})
    return await cursor.to_list(length=5000)


@app.get("/responses/{prolific_id}")
async def get_response(prolific_id: str):
    doc = await collection.find_one({"prolific_id": prolific_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return doc


@app.get("/health")
async def health():
    return {"status": "healthy"}
