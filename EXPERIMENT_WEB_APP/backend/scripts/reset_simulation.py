import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")

async def reset():
    client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    
    print("Resetting simulation data...")
    
    # Clear participants
    result = await db.participants.delete_many({})
    print(f"Deleted {result.deleted_count} participants.")
    
    # Clear logs (optional, but good for fresh start)
    result = await db.logs.delete_many({})
    print(f"Deleted {result.deleted_count} logs.")
    
    print("Simulation reset complete.")

if __name__ == "__main__":
    asyncio.run(reset())
