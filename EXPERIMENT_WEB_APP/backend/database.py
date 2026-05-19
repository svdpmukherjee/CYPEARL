"""
CYPEARL Multi-Scenario Database Configuration.

Database names are loaded from environment variables so they are never
exposed in source control. Set these in your local .env (development) or
in the host's environment-variable panel (Render, etc.) for production.

Required env vars:
    MONGO_URL                  MongoDB connection string
    PHISHING_DB_NAME           Database name for the phishing scenario
    DARKPATTERNS_DB_NAME       Database name for the dark-patterns scenario
    FAKENEWS_DB_NAME           Database name for the fake-news scenario
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional

import certifi

from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise ValueError("MONGO_URL not found in environment variables")

PHISHING_DB_NAME = os.getenv("PHISHING_DB_NAME")
DARKPATTERNS_DB_NAME = os.getenv("DARKPATTERNS_DB_NAME")
FAKENEWS_DB_NAME = os.getenv("FAKENEWS_DB_NAME")

_missing = [
    name for name, value in (
        ("PHISHING_DB_NAME", PHISHING_DB_NAME),
        ("DARKPATTERNS_DB_NAME", DARKPATTERNS_DB_NAME),
        ("FAKENEWS_DB_NAME", FAKENEWS_DB_NAME),
    ) if not value
]
if _missing:
    raise ValueError(
        f"Missing database env vars: {', '.join(_missing)}"
    )

SCENARIO_DATABASES = {
    "phishing": PHISHING_DB_NAME,
    "dark-patterns": DARKPATTERNS_DB_NAME,
    "fake-news": FAKENEWS_DB_NAME,
}

# Default database (phishing for backwards compatibility)
DEFAULT_DB_NAME = PHISHING_DB_NAME

client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())

db = client[DEFAULT_DB_NAME]

databases = {
    scenario: client[db_name]
    for scenario, db_name in SCENARIO_DATABASES.items()
}


async def get_database(scenario: Optional[str] = None):
    if scenario and scenario in databases:
        return databases[scenario]
    return db


def get_database_sync(scenario: Optional[str] = None):
    if scenario and scenario in databases:
        return databases[scenario]
    return db


def get_phishing_db():
    return databases["phishing"]


def get_darkpatterns_db():
    return databases["dark-patterns"]


def get_fakenews_db():
    return databases["fake-news"]
