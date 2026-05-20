"""
Reset Simulation Script - Database Cleanup Utility

This script allows selective or complete reset of experiment databases.
Users can choose to reset:
- Phishing experiment data- Dark Patterns experiment data- Fake News experiment data
- All databases at once

Usage:
    python reset_simulation.py

Then follow the interactive prompts to select which database(s) to reset.
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("PHISHING_DB_NAME") or os.getenv("DB_NAME")
DARKPATTERNS_DB_NAME = os.getenv("DARKPATTERNS_DB_NAME")
FAKENEWS_DB_NAME = os.getenv("FAKENEWS_DB_NAME")

_missing = [
    name for name, value in (
        ("MONGO_URL", MONGO_URL),
        ("PHISHING_DB_NAME", DB_NAME),
        ("DARKPATTERNS_DB_NAME", DARKPATTERNS_DB_NAME),
        ("FAKENEWS_DB_NAME", FAKENEWS_DB_NAME),
    ) if not value
]
if _missing:
    raise ValueError(f"Missing env vars: {', '.join(_missing)}")


async def reset_phishing(client):
    """Reset phishing experiment database ."""
    db = client[DB_NAME]
    print("\n--- Resetting Phishing DB  ---")

    result = await db.participants.delete_many({})
    print(f"  Deleted {result.deleted_count} participants.")

    result = await db.responses.delete_many({})
    print(f"  Deleted {result.deleted_count} responses.")

    result = await db.pre_survey_responses.delete_many({})
    print(f"  Deleted {result.deleted_count} pre-survey responses.")

    result = await db.post_survey_responses.delete_many({})
    print(f"  Deleted {result.deleted_count} post-survey responses.")

    result = await db.logs.delete_many({})
    print(f"  Deleted {result.deleted_count} logs.")

    print("  Phishing database reset complete.")


async def reset_dark_patterns(client):
    """Reset dark patterns experiment database ."""
    db = client[DARKPATTERNS_DB_NAME]
    print("\n--- Resetting Dark Patterns DB  ---")

    result = await db.participants.delete_many({})
    print(f"  Deleted {result.deleted_count} participants.")

    result = await db.responses.delete_many({})
    print(f"  Deleted {result.deleted_count} responses.")

    # result = await db.tasks.delete_many({})
    # print(f"  Deleted {result.deleted_count} tasks.")

    result = await db.pre_survey_responses.delete_many({})
    print(f"  Deleted {result.deleted_count} pre-survey responses.")

    result = await db.post_survey_responses.delete_many({})
    print(f"  Deleted {result.deleted_count} post-survey responses.")

    print("  Dark Patterns database reset complete.")


async def reset_fake_news(client):
    """Reset fake news experiment database."""
    db = client[FAKENEWS_DB_NAME]
    print("\n--- Resetting Fake News DB ---")

    result = await db.participants.delete_many({})
    print(f"  Deleted {result.deleted_count} participants.")

    result = await db.responses.delete_many({})
    print(f"  Deleted {result.deleted_count} responses.")

    # result = await db.articles.delete_many({})
    # print(f"  Deleted {result.deleted_count} articles.")

    result = await db.pre_survey_responses.delete_many({})
    print(f"  Deleted {result.deleted_count} pre-survey responses.")

    result = await db.post_survey_responses.delete_many({})
    print(f"  Deleted {result.deleted_count} post-survey responses.")

    print("  Fake News database reset complete.")


async def reset_all(client):
    """Reset all experiment databases."""
    await reset_phishing(client)
    await reset_dark_patterns(client)
    await reset_fake_news(client)


def display_menu():
    """Display the database selection menu."""
    print("\n" + "="*60)
    print("       CYPEARL Database Reset Utility")
    print("="*60)
    print("\nSelect which database(s) to reset:\n")
    print("  1. Phishing Emails      ")
    print("  2. Dark Patterns        ")
    print("  3. Fake News           ")
    print("  4. ALL databases        (reset everything)")
    print("  5. Exit                 (cancel and exit)")
    print("\n" + "-"*60)


def get_confirmation(db_name):
    """Get user confirmation before deleting data."""
    print(f"\n⚠️  WARNING: This will permanently delete all data in {db_name}!")
    response = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
    return response == 'yes'


async def main():
    """Main entry point with interactive database selection."""
    print("\nConnecting to MongoDB...")

    try:
        client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
        # Test connection
        await client.admin.command('ping')
        print("Connected successfully!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return

    while True:
        display_menu()

        try:
            choice = input("Enter your choice (1-5): ").strip()
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break

        if choice == '1':
            if get_confirmation("Phishing "):
                await reset_phishing(client)
                print("\n✅ Phishing database has been reset.")
            else:
                print("\nOperation cancelled.")

        elif choice == '2':
            if get_confirmation("Dark Patterns "):
                await reset_dark_patterns(client)
                print("\n✅ Dark Patterns database has been reset.")
            else:
                print("\nOperation cancelled.")

        elif choice == '3':
            if get_confirmation("Fake News"):
                await reset_fake_news(client)
                print("\n✅ Fake News database has been reset.")
            else:
                print("\nOperation cancelled.")

        elif choice == '4':
            print("\n⚠️  WARNING: This will delete ALL experiment data!")
            print("    - Phishing ")
            print("    - Dark Patterns ")
            print("    - Fake News")
            if get_confirmation("ALL databases"):
                await reset_all(client)
                print("\n✅ All databases have been reset.")
            else:
                print("\nOperation cancelled.")

        elif choice == '5':
            print("\nExiting without changes.")
            break

        else:
            print("\n❌ Invalid choice. Please enter a number between 1 and 5.")

        # Ask if user wants to perform another operation
        another = input("\nWould you like to perform another operation? (yes/no): ").strip().lower()
        if another != 'yes':
            print("\nGoodbye!")
            break

    client.close()


# For backward compatibility - reset all databases
async def reset():
    """Legacy function - resets all databases without prompting."""
    client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())

    print("Resetting all simulation data (legacy mode)...")

    await reset_phishing(client)
    await reset_dark_patterns(client)

    print("\nAll databases reset complete.")
    client.close()


if __name__ == "__main__":
    # Run interactive mode by default
    asyncio.run(main())
