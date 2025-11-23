from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId

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

class Email(BaseModel):
    id: Optional[str] = None
    sender_name: str
    sender_email: str
    subject: str
    body: str
    is_phishing: bool
    order_id: int
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserAction(BaseModel):
    email_id: str
    action_type: str  # 'delete', 'report', 'click', 'ignore'
    reason: Optional[str] = None
    confidence: Optional[int] = None # 1-10 or similar scale
    timestamp: datetime = Field(default_factory=datetime.now)
    hover_data: Optional[Dict[str, Any]] = None # Store hover durations/counts
    latency_ms: Optional[int] = None
    email_snapshot: Optional[Dict[str, Any]] = None # Snapshot of email metadata (is_phishing, category, etc.)
    client_info: Optional[Dict[str, Any]] = None # Browser/Device info

class Participant(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    current_email_order: int = 0
    completed: bool = False
    email_timestamps: Dict[str, datetime] = Field(default_factory=dict) # Map order_id (str) to delivery time
    read_email_ids: List[str] = Field(default_factory=list)
    deleted_email_ids: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
