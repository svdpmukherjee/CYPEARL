import os
from motor.motor_asyncio import AsyncIOMotorClient

import certifi

from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise ValueError("MONGO_URL not found in environment variables")
DB_NAME = "bizmail_db"

client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
db = client[DB_NAME]

async def get_database():
    return db