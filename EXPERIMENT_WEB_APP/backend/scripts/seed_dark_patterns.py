"""
Seed Dark Pattern Tasks into the dark-patterns scenario DB (tasks collection).

16 tasks based on 2x2x2x2 factorial design:
- UI Type: Dark Pattern (8) vs Clean UI (8)
- Intensity: Aggressive vs Mild vs None
- Time Pressure: Present vs Absent
- Visual Manipulation: High vs Neutral

Each task includes:
- Metadata (task_id, ui_type, intensity, etc.)
- Scenario information (title, scenario, goal, context)
- UI Content (text, buttons, options for rendering)

Usage:
    python scripts/seed_dark_patterns.py
"""

import asyncio
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import certifi

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DARKPATTERNS_DB_NAME = os.getenv("DARKPATTERNS_DB_NAME")

if not MONGO_URL or not DARKPATTERNS_DB_NAME:
    raise ValueError("MONGO_URL and DARKPATTERNS_DB_NAME must be set in environment")


DARK_PATTERN_TASKS = [
    # ================================================================
    # DARK PATTERN TASKS (8)
    # ================================================================
    {
        "task_id": "dp_01",
        "ui_type": "dark",
        "intensity": "aggressive",
        "time_pressure": True,
        "visual_manipulation": "high",
        "context": "cookie_consent",
        "description": "Cookie banner with 'Accept All' prominent, 'Manage' hidden, countdown timer",
        "dark_patterns_present": ["preselection", "false_hierarchy", "urgency", "hidden_information"],
        "desired_action": "reject_all",
        "manipulated_action": "accept_all",
        "time_limit_seconds": 30,
        "ground_truth": 1,
        # Scenario information
        "scenario_info": {
            "title": "Cookie Consent Decision",
            "scenario": "You're browsing the internet and land on a new website for the first time. A cookie consent popup appears. You generally prefer to limit tracking and only accept essential cookies when possible.",
            "goal": "Decide how to handle cookies on this website based on your privacy preferences.",
            "context": "Imagine this is a website you might visit occasionally but haven't created an account on."
        },
        # UI Content for rendering
        "ui_content": {
            "site_name": "TechNews Daily",
            "headline": "We value your privacy",
            "description": "This website uses cookies to enhance your browsing experience, serve personalized ads or content, and analyze our traffic.",
            "accept_all_button": "Accept All Cookies",
            "reject_all_button": "Reject All",
            "manage_button": "Manage Preferences",
            "fine_print": "By clicking 'Accept All', you consent to the use of ALL cookies including marketing, analytics, and third-party tracking cookies.",
            "urgency_message": "Make your choice quickly to continue browsing",
            # Dark pattern specific
            "accept_button_style": "prominent",  # Large, colorful
            "reject_button_style": "hidden",  # Small, gray, hard to find
            "preselected_options": ["essential", "analytics", "marketing", "third_party"],
        }
    },
    {
        "task_id": "dp_02",
        "ui_type": "dark",
        "intensity": "aggressive",
        "time_pressure": True,
        "visual_manipulation": "neutral",
        "context": "newsletter_unsubscribe",
        "description": "Unsubscribe flow with 7 steps, hidden final button, guilt-trip language",
        "dark_patterns_present": ["obstruction", "confirmshaming", "urgency"],
        "desired_action": "complete_unsubscribe",
        "manipulated_action": "abandon",
        "time_limit_seconds": 60,
        "ground_truth": 1,
        "scenario_info": {
            "title": "Newsletter Unsubscription",
            "scenario": "You've been receiving too many promotional emails from a company's newsletter. You originally signed up months ago but no longer find the content useful. You've decided you want to unsubscribe.",
            "goal": "Successfully unsubscribe from this newsletter.",
            "context": "You don't want to receive any more emails from this sender."
        },
        "ui_content": {
            "company_name": "StyleBox Premium",
            "guilt_trip_messages": [
                "We'll miss you! 💔",
                "Are you sure you want to leave?",
                "You'll miss out on exclusive deals!",
                "This action cannot be undone."
            ],
            "stay_button": "Stay Subscribed",
            "continue_unsubscribe_link": "I still want to unsubscribe",
            "reasons": [
                "Too many emails",
                "Content not relevant",
                "Found better alternatives",
                "Privacy concerns",
                "Other"
            ],
            "final_warning": "You will lose access to: Exclusive member discounts, Early access to sales, Personalized recommendations",
            "hidden_complete_button": "Complete unsubscription",
            "total_steps": 5
        }
    },
    {
        "task_id": "dp_03",
        "ui_type": "dark",
        "intensity": "aggressive",
        "time_pressure": False,
        "visual_manipulation": "high",
        "context": "free_trial",
        "description": "Free trial requiring credit card, auto-renewal in tiny text, annual plan preselected",
        "dark_patterns_present": ["forced_action", "hidden_costs", "preselection", "false_hierarchy"],
        "desired_action": "skip_trial",
        "manipulated_action": "start_trial_with_card",
        "time_limit_seconds": None,
        "ground_truth": 1,
        "scenario_info": {
            "title": "Free Trial Evaluation",
            "scenario": "You're interested in trying out a new subscription service that offers a free trial. However, you're cautious about services that might be difficult to cancel or that charge unexpectedly.",
            "goal": "Decide whether to start the free trial based on the terms presented.",
            "context": "Consider what commitments or risks might be involved."
        },
        "ui_content": {
            "service_name": "StreamMax Pro",
            "headline": "Start Your FREE 7-Day Trial",
            "subheadline": "Unlimited movies, shows, and exclusive content",
            "plans": [
                {"name": "Monthly", "price": "$14.99/month", "preselected": False},
                {"name": "Annual", "price": "$99.99/year (Save 44%)", "preselected": True}
            ],
            "card_required_text": "Credit card required to start trial",
            "hidden_terms": "Your card will be charged automatically when trial ends. Cancel anytime before trial ends to avoid charges. Annual plans are non-refundable.",
            "start_trial_button": "Start Free Trial",
            "skip_button": "Maybe Later",
            "skip_button_style": "tiny_gray_text"
        }
    },
    {
        "task_id": "dp_04",
        "ui_type": "dark",
        "intensity": "aggressive",
        "time_pressure": False,
        "visual_manipulation": "neutral",
        "context": "account_deletion",
        "description": "Account deletion with 7-step process, repeated 'are you sure', hidden confirm button",
        "dark_patterns_present": ["obstruction", "confirmshaming", "nagging"],
        "desired_action": "complete_deletion",
        "manipulated_action": "abandon",
        "time_limit_seconds": None,
        "ground_truth": 1,
        "scenario_info": {
            "title": "Account Deletion Request",
            "scenario": "You have an account on an online service that you no longer use. You've decided you want to permanently delete your account and all associated data for privacy reasons.",
            "goal": "Successfully delete your account from this service.",
            "context": "You're certain you want to delete the account and don't plan to return."
        },
        "ui_content": {
            "service_name": "CloudSync Plus",
            "warning_messages": [
                "Are you sure? Your data will be permanently deleted.",
                "We noticed you have 247 files stored. These will be lost forever.",
                "Your premium features will be deactivated immediately.",
                "This cannot be undone. Are you absolutely certain?"
            ],
            "keep_account_button": "Keep My Account",
            "proceed_delete_link": "Continue to delete",
            "confirmation_steps": 5,
            "final_confirm_button": "Yes, Delete My Account",
            "final_confirm_style": "hidden_small_text"
        }
    },
    {
        "task_id": "dp_05",
        "ui_type": "dark",
        "intensity": "mild",
        "time_pressure": True,
        "visual_manipulation": "high",
        "context": "checkout_addon",
        "description": "Checkout with preselected insurance addon, slightly muted uncheck option",
        "dark_patterns_present": ["preselection", "false_hierarchy"],
        "desired_action": "uncheck_addon",
        "manipulated_action": "keep_addon",
        "time_limit_seconds": 45,
        "ground_truth": 1,
        "scenario_info": {
            "title": "Checkout Review",
            "scenario": "You're purchasing an item online and have reached the checkout page. You want to complete your purchase with only the items you originally intended to buy.",
            "goal": "Complete the checkout with your intended purchase.",
            "context": "Review the order carefully to ensure you're only paying for what you want."
        },
        "ui_content": {
            "store_name": "TechMart",
            "main_product": {"name": "Wireless Bluetooth Headphones", "price": 79.99},
            "addon": {
                "name": "Extended Protection Plan (2 Years)",
                "price": 12.99,
                "preselected": True,
                "description": "Covers accidental damage and defects"
            },
            "subtotal_with_addon": 92.98,
            "subtotal_without_addon": 79.99,
            "checkout_button": "Complete Purchase",
            "addon_checkbox_style": "preselected_muted"
        }
    },
    {
        "task_id": "dp_06",
        "ui_type": "dark",
        "intensity": "mild",
        "time_pressure": True,
        "visual_manipulation": "neutral",
        "context": "shipping_upgrade",
        "description": "Shipping with premium default, standard option available but not highlighted",
        "dark_patterns_present": ["preselection", "hidden_information"],
        "desired_action": "select_standard",
        "manipulated_action": "keep_premium",
        "time_limit_seconds": 30,
        "ground_truth": 1,
        "scenario_info": {
            "title": "Shipping Selection",
            "scenario": "You're completing an online purchase and need to select a shipping option. You're not in a hurry and would prefer the most economical shipping option for this order.",
            "goal": "Select your preferred shipping method based on your needs.",
            "context": "Consider the cost and delivery time that works best for you."
        },
        "ui_content": {
            "shipping_options": [
                {"name": "Express Delivery", "price": 14.99, "days": "1-2 business days", "preselected": True, "style": "highlighted"},
                {"name": "Standard Shipping", "price": 4.99, "days": "5-7 business days", "preselected": False, "style": "muted"},
                {"name": "Economy", "price": 0.00, "days": "7-14 business days", "preselected": False, "style": "hidden"}
            ],
            "continue_button": "Continue to Payment"
        }
    },
    {
        "task_id": "dp_07",
        "ui_type": "dark",
        "intensity": "mild",
        "time_pressure": False,
        "visual_manipulation": "high",
        "context": "privacy_settings",
        "description": "Privacy toggles with confusing double-negatives, sharing ON by default",
        "dark_patterns_present": ["trick_questions", "preselection"],
        "desired_action": "disable_sharing",
        "manipulated_action": "leave_enabled",
        "time_limit_seconds": None,
        "ground_truth": 1,
        "scenario_info": {
            "title": "Privacy Settings Configuration",
            "scenario": "You're reviewing the privacy settings on an app or service you use. You generally prefer to share minimal personal data and limit targeted advertising when possible.",
            "goal": "Configure your privacy settings according to your preferences.",
            "context": "You want more control over your personal data and how it's used."
        },
        "ui_content": {
            "app_name": "SocialConnect",
            "settings": [
                {"label": "Don't prevent sharing my activity with partners", "default": True, "confusing": True},
                {"label": "Opt out of not receiving personalized ads", "default": True, "confusing": True},
                {"label": "Disable the option to hide my online status", "default": True, "confusing": True},
                {"label": "Share my location for better recommendations", "default": True, "confusing": False}
            ],
            "save_button": "Save Preferences",
            "fine_print": "Changes may take up to 30 days to take effect."
        }
    },
    {
        "task_id": "dp_08",
        "ui_type": "dark",
        "intensity": "mild",
        "time_pressure": False,
        "visual_manipulation": "neutral",
        "context": "decline_offer",
        "description": "Popup with 'No thanks, I don't like saving money' as decline option",
        "dark_patterns_present": ["confirmshaming"],
        "desired_action": "decline",
        "manipulated_action": "accept",
        "time_limit_seconds": None,
        "ground_truth": 1,
        "scenario_info": {
            "title": "Special Offer Response",
            "scenario": "While browsing or trying to leave a website, a special offer popup appears. You weren't looking for any additional products or services and want to continue with your original intention.",
            "goal": "Decline the offer and proceed with what you were doing.",
            "context": "You're not interested in the promotion being offered."
        },
        "ui_content": {
            "headline": "Wait! Don't miss out!",
            "offer": "Get 20% OFF your first order!",
            "discount_code": "SAVE20",
            "accept_button": "Yes! I want to save money",
            "decline_button": "No thanks, I prefer to pay full price",
            "decline_button_style": "shameful_gray"
        }
    },

    # ================================================================
    # CLEAN UI TASKS (8) - Control conditions
    # ================================================================
    {
        "task_id": "clean_01",
        "ui_type": "clean",
        "intensity": "none",
        "time_pressure": True,
        "visual_manipulation": "high",
        "context": "cookie_consent",
        "description": "Cookie banner with equal-sized Accept/Reject buttons, clear options",
        "dark_patterns_present": [],
        "desired_action": "reject_all",
        "manipulated_action": None,
        "time_limit_seconds": 30,
        "ground_truth": 0,
        "scenario_info": {
            "title": "Cookie Consent Decision",
            "scenario": "You're browsing the internet and land on a new website for the first time. A cookie consent popup appears. You generally prefer to limit tracking and only accept essential cookies when possible.",
            "goal": "Decide how to handle cookies on this website based on your privacy preferences.",
            "context": "Imagine this is a website you might visit occasionally but haven't created an account on."
        },
        "ui_content": {
            "site_name": "TechNews Daily",
            "headline": "Cookie Preferences",
            "description": "We use cookies to improve your experience. You can choose which cookies to accept.",
            "accept_all_button": "Accept All",
            "reject_all_button": "Reject All",
            "manage_button": "Customize",
            "buttons_style": "equal_size",
            "preselected_options": ["essential"]
        }
    },
    {
        "task_id": "clean_02",
        "ui_type": "clean",
        "intensity": "none",
        "time_pressure": True,
        "visual_manipulation": "neutral",
        "context": "newsletter_unsubscribe",
        "description": "One-click unsubscribe with clear confirmation",
        "dark_patterns_present": [],
        "desired_action": "complete_unsubscribe",
        "manipulated_action": None,
        "time_limit_seconds": 30,
        "ground_truth": 0,
        "scenario_info": {
            "title": "Newsletter Unsubscription",
            "scenario": "You've been receiving too many promotional emails from a company's newsletter. You originally signed up months ago but no longer find the content useful. You've decided you want to unsubscribe.",
            "goal": "Successfully unsubscribe from this newsletter.",
            "context": "You don't want to receive any more emails from this sender."
        },
        "ui_content": {
            "company_name": "StyleBox Premium",
            "headline": "Unsubscribe",
            "message": "Click below to unsubscribe from our newsletter.",
            "unsubscribe_button": "Unsubscribe",
            "cancel_button": "Cancel",
            "total_steps": 1
        }
    },
    {
        "task_id": "clean_03",
        "ui_type": "clean",
        "intensity": "none",
        "time_pressure": False,
        "visual_manipulation": "high",
        "context": "free_trial",
        "description": "Free trial with no card required, clear terms, no preselection",
        "dark_patterns_present": [],
        "desired_action": "start_trial",
        "manipulated_action": None,
        "time_limit_seconds": None,
        "ground_truth": 0,
        "scenario_info": {
            "title": "Free Trial Evaluation",
            "scenario": "You're interested in trying out a new subscription service that offers a free trial. However, you're cautious about services that might be difficult to cancel or that charge unexpectedly.",
            "goal": "Decide whether to start the free trial based on the terms presented.",
            "context": "Consider what commitments or risks might be involved."
        },
        "ui_content": {
            "service_name": "StreamMax Pro",
            "headline": "Start Your Free 7-Day Trial",
            "subheadline": "No credit card required",
            "features": ["Unlimited streaming", "Cancel anytime", "No automatic charges"],
            "terms": "Your trial will end automatically after 7 days. We'll remind you before it ends.",
            "start_trial_button": "Start Free Trial",
            "skip_button": "No Thanks",
            "buttons_style": "equal_prominence"
        }
    },
    {
        "task_id": "clean_04",
        "ui_type": "clean",
        "intensity": "none",
        "time_pressure": False,
        "visual_manipulation": "neutral",
        "context": "account_deletion",
        "description": "2-step deletion with clear confirm button",
        "dark_patterns_present": [],
        "desired_action": "complete_deletion",
        "manipulated_action": None,
        "time_limit_seconds": None,
        "ground_truth": 0,
        "scenario_info": {
            "title": "Account Deletion Request",
            "scenario": "You have an account on an online service that you no longer use. You've decided you want to permanently delete your account and all associated data for privacy reasons.",
            "goal": "Successfully delete your account from this service.",
            "context": "You're certain you want to delete the account and don't plan to return."
        },
        "ui_content": {
            "service_name": "CloudSync Plus",
            "headline": "Delete Account",
            "message": "This will permanently delete your account and all data.",
            "confirm_button": "Delete My Account",
            "cancel_button": "Cancel",
            "confirmation_steps": 2
        }
    },
    {
        "task_id": "clean_05",
        "ui_type": "clean",
        "intensity": "none",
        "time_pressure": True,
        "visual_manipulation": "high",
        "context": "checkout_addon",
        "description": "Checkout with addon unchecked by default, clear opt-in",
        "dark_patterns_present": [],
        "desired_action": "proceed_without_addon",
        "manipulated_action": None,
        "time_limit_seconds": 45,
        "ground_truth": 0,
        "scenario_info": {
            "title": "Checkout Review",
            "scenario": "You're purchasing an item online and have reached the checkout page. You want to complete your purchase with only the items you originally intended to buy.",
            "goal": "Complete the checkout with your intended purchase.",
            "context": "Review the order carefully to ensure you're only paying for what you want."
        },
        "ui_content": {
            "store_name": "TechMart",
            "main_product": {"name": "Wireless Bluetooth Headphones", "price": 79.99},
            "addon": {
                "name": "Extended Protection Plan (2 Years)",
                "price": 12.99,
                "preselected": False,
                "description": "Optional: Covers accidental damage"
            },
            "checkout_button": "Complete Purchase"
        }
    },
    {
        "task_id": "clean_06",
        "ui_type": "clean",
        "intensity": "none",
        "time_pressure": True,
        "visual_manipulation": "neutral",
        "context": "shipping_upgrade",
        "description": "Shipping with standard as default, premium as clear upgrade option",
        "dark_patterns_present": [],
        "desired_action": "keep_standard",
        "manipulated_action": None,
        "time_limit_seconds": 30,
        "ground_truth": 0,
        "scenario_info": {
            "title": "Shipping Selection",
            "scenario": "You're completing an online purchase and need to select a shipping option. You're not in a hurry and would prefer the most economical shipping option for this order.",
            "goal": "Select your preferred shipping method based on your needs.",
            "context": "Consider the cost and delivery time that works best for you."
        },
        "ui_content": {
            "shipping_options": [
                {"name": "Standard Shipping", "price": 4.99, "days": "5-7 business days", "preselected": True},
                {"name": "Express Delivery", "price": 14.99, "days": "1-2 business days", "preselected": False},
                {"name": "Free Economy", "price": 0.00, "days": "7-14 business days", "preselected": False}
            ],
            "continue_button": "Continue to Payment"
        }
    },
    {
        "task_id": "clean_07",
        "ui_type": "clean",
        "intensity": "none",
        "time_pressure": False,
        "visual_manipulation": "high",
        "context": "privacy_settings",
        "description": "Clear privacy toggles, sharing OFF by default, plain language",
        "dark_patterns_present": [],
        "desired_action": "keep_disabled",
        "manipulated_action": None,
        "time_limit_seconds": None,
        "ground_truth": 0,
        "scenario_info": {
            "title": "Privacy Settings Configuration",
            "scenario": "You're reviewing the privacy settings on an app or service you use. You generally prefer to share minimal personal data and limit targeted advertising when possible.",
            "goal": "Configure your privacy settings according to your preferences.",
            "context": "You want more control over your personal data and how it's used."
        },
        "ui_content": {
            "app_name": "SocialConnect",
            "settings": [
                {"label": "Share activity with partners", "default": False, "confusing": False},
                {"label": "Receive personalized ads", "default": False, "confusing": False},
                {"label": "Show my online status", "default": False, "confusing": False},
                {"label": "Share location for recommendations", "default": False, "confusing": False}
            ],
            "save_button": "Save Preferences"
        }
    },
    {
        "task_id": "clean_08",
        "ui_type": "clean",
        "intensity": "none",
        "time_pressure": False,
        "visual_manipulation": "neutral",
        "context": "decline_offer",
        "description": "Popup with neutral 'No thanks' decline option",
        "dark_patterns_present": [],
        "desired_action": "decline",
        "manipulated_action": None,
        "time_limit_seconds": None,
        "ground_truth": 0,
        "scenario_info": {
            "title": "Special Offer Response",
            "scenario": "While browsing or trying to leave a website, a special offer popup appears. You weren't looking for any additional products or services and want to continue with your original intention.",
            "goal": "Decline the offer and proceed with what you were doing.",
            "context": "You're not interested in the promotion being offered."
        },
        "ui_content": {
            "headline": "Special Offer",
            "offer": "Get 20% OFF your first order!",
            "discount_code": "SAVE20",
            "accept_button": "Apply Discount",
            "decline_button": "No Thanks",
            "buttons_style": "equal_size"
        }
    },
]


async def seed():
    client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
    db = client[DARKPATTERNS_DB_NAME]

    print("Clearing existing dark pattern tasks...")
    await db.tasks.delete_many({})

    print(f"Seeding {len(DARK_PATTERN_TASKS)} dark pattern tasks...")
    result = await db.tasks.insert_many(DARK_PATTERN_TASKS)
    print(f"Inserted {len(result.inserted_ids)} tasks.")

    # Create indexes
    await db.tasks.create_index("task_id", unique=True)
    await db.tasks.create_index("ui_type")
    await db.tasks.create_index("context")
    print("Created indexes on tasks collection.")

    # Ensure other collections exist with indexes
    await db.participants.create_index("participant_id", unique=True)
    await db.participants.create_index("prolific_id", unique=True)
    await db.responses.create_index("participant_id")
    await db.responses.create_index("task_id")
    await db.pre_survey_responses.create_index("participant_id", unique=True)
    await db.post_survey_responses.create_index("participant_id", unique=True)
    print("Created indexes on participants, responses, pre_survey_responses, post_survey_responses.")

    # Verify
    count = await db.tasks.count_documents({})
    print(f"\nVerification: {count} tasks in tasks collection")

    dark_count = await db.tasks.count_documents({"ui_type": "dark"})
    clean_count = await db.tasks.count_documents({"ui_type": "clean"})
    print(f"  Dark pattern tasks: {dark_count}")
    print(f"  Clean UI tasks:     {clean_count}")

    # Show sample of what's stored
    sample = await db.tasks.find_one({"task_id": "dp_01"})
    if sample:
        print(f"\nSample task (dp_01):")
        print(f"  - Scenario: {sample.get('scenario_info', {}).get('title', 'N/A')}")
        print(f"  - UI Content Keys: {list(sample.get('ui_content', {}).keys())}")

    collections = await db.list_collection_names()
    print(f"\nCollections in dark-patterns DB: {sorted(collections)}")

    print("\nDark patterns seed complete.")
    print("\nNOTE: Run this script whenever you need to update task content.")
    print("      The frontend will fetch this content from the database.")


if __name__ == "__main__":
    asyncio.run(seed())
