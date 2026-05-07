"""Clear all collections in the screening MongoDB database.

Usage:
    python clear_db.py            # interactive confirmation
    python clear_db.py --yes      # skip confirmation
"""

import os
import sys

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise SystemExit("MONGO_URL not found in environment variables")

DB_NAME = os.getenv("SCREENING_DB_NAME", "cypearl_screening")


def main():
    client = MongoClient(MONGO_URL, tlsCAFile=certifi.where())
    db = client[DB_NAME]

    collection_names = db.list_collection_names()
    if not collection_names:
        print(f"Database '{DB_NAME}' has no collections. Nothing to clear.")
        return

    print(f"Database: {DB_NAME}")
    print("Collections and current document counts:")
    for name in collection_names:
        print(f"  - {name}: {db[name].count_documents({})}")

    if "--yes" not in sys.argv:
        confirm = input(
            f"\nDelete ALL documents from these collections in '{DB_NAME}'? Type 'yes' to confirm: "
        ).strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    for name in collection_names:
        result = db[name].delete_many({})
        print(f"Cleared '{name}': {result.deleted_count} document(s) deleted.")

    print("Done.")


if __name__ == "__main__":
    main()
