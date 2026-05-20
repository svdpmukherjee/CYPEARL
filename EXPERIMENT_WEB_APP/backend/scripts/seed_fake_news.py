"""
CYPEARL Fake News Experiment - Seed Script (Twitter/X Style)

Seeds the fake-news scenario DB with 16 factorial-designed news items as tweets.

Factorial Design (2x2x2x2 = 16 items):
- Veracity: Fake (8) vs Real (8)
- Political Lean: Liberal vs Conservative (balanced)
- Source Credibility: High (mainstream) vs Low (unknown)
- Emotional Valence: High vs Neutral

Run with: python scripts/seed_fake_news.py
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient
import certifi

MONGO_URL = os.getenv("MONGO_URL")
FAKENEWS_DB_NAME = os.getenv("FAKENEWS_DB_NAME")
if not MONGO_URL or not FAKENEWS_DB_NAME:
    raise ValueError("MONGO_URL and FAKENEWS_DB_NAME must be set in environment")

# 16 Factorial-Designed News Items (Twitter/X Style)
NEWS_ITEMS = [
    # ============================================================================
    # FAKE NEWS (8 items)
    # ============================================================================

    # 1) Fake – Conservative – High credibility – High emotion
    # IMPERSONATOR: Fake Warren Buffett account (misspelled handle)
    {
        "item_id": "fake_01",
        "veracity": "fake",
        "political_lean": "conservative",
        "source_credibility": "high",
        "emotional_valence": "high",
        "display_name": "Warren Buffett",
        "handle": "@WarrenBuffet",  # Missing one 't' - impersonator
        "badge_type": "blue",  # Blue badge for impersonator
        "source_logo": "/fakenews/logos/buffet.png",
        "thumbnail": "/fakenews/thumbnails/image.jpg",
        "thumbnail_type": "image",
        "tweet_content": "I've been warning about this for years. Leaked briefing notes show foreign partners preparing massive new tariffs on American farmers next year, while the current administration 'chooses not to respond' 😡 This is economic sabotage. Rural communities are being sacrificed. Share if you agree.\n#TradeWar #Farmers #Economy",
        "topic": "politics",
        "engagement_counts": {"replies": 8900, "retweets": 45000, "likes": 189000, "views": 2400000},
        "fact_check_status": "false",
        "manipulation_techniques": ["impersonation", "emotional_language", "vague_attribution", "call_to_action"]
    },

    # 2) Fake – Conservative – High credibility – Neutral
    # IMPERSONATOR: Fake Jordan Peterson account (extra underscore in handle)
    {
        "item_id": "fake_02",
        "veracity": "fake",
        "political_lean": "conservative",
        "source_credibility": "high",
        "emotional_valence": "neutral",
        "display_name": "Dr Jordan B Peterson",
        "handle": "@jordanbpeterson_",  # Extra underscore - impersonator
        "badge_type": "blue",  # Blue badge for impersonator
        "source_logo": "/fakenews/logos/Peterson.png",
        "thumbnail": "/fakenews/thumbnails/image.jpg",
        "thumbnail_type": "image",
        "tweet_content": "An unverified draft of an immigration proposal circulating among staffers appears to grant automatic residency to all recent border crossers within six months. The document lacks key legal details. This is precisely the kind of policy drift I've been discussing.\n#Immigration #Policy #Order",
        "topic": "politics",
        "engagement_counts": {"replies": 2300, "retweets": 8900, "likes": 34000, "views": 890000},
        "fact_check_status": "false",
        "manipulation_techniques": ["impersonation", "unverified_claims", "hedging_language"]
    },

    # 3) Fake – Conservative – Low credibility – High emotion
    {
        "item_id": "fake_03",
        "veracity": "fake",
        "political_lean": "conservative",
        "source_credibility": "low",
        "emotional_valence": "high",
        "display_name": "Freedom Guard Tribune",
        "handle": "@FreedomGuardUSA",
        "badge_type": "grey",
        "source_logo": "/fakenews/logos/freedom.png",
        "thumbnail": "/fakenews/thumbnails/video.mp4",
        "thumbnail_type": "video",
        "tweet_content": "BREAKING 🚨: Global elites just signed a secret pact to replace MILLIONS of local workers with cheap foreign labor by 2030. They think ordinary citizens are too distracted to notice. This is economic sabotage, not policy. Retweet before they bury this! 😤🇺🇸\n#Jobs #Globalism #WakeUp",
        "topic": "politics",
        "engagement_counts": {"replies": 8900, "retweets": 34000, "likes": 89000, "views": 1200000},
        "fact_check_status": "false",
        "manipulation_techniques": ["conspiracy_framing", "urgency_language", "emotional_appeal", "call_to_action"]
    },

    # 4) Fake – Conservative – Low credibility – Neutral
    {
        "item_id": "fake_04",
        "veracity": "fake",
        "political_lean": "conservative",
        "source_credibility": "low",
        "emotional_valence": "neutral",
        "display_name": "Campus Watchdog Blog",
        "handle": "@CampusWatchBlog",
        "badge_type": "gold",
        "source_logo": "/fakenews/logos/free_speech.png",
        "thumbnail": None,
        "thumbnail_type": None,
        "tweet_content": "A widely shared online 'survey' claims that more than 80% of professors at major universities quietly downgrade students for expressing conservative views. The report lists no sponsoring institution, and the methodology is not available for review.\n#Campus #FreeSpeech",
        "topic": "education",
        "engagement_counts": {"replies": 340, "retweets": 890, "likes": 2100, "views": 78000},
        "fact_check_status": "false",
        "manipulation_techniques": ["false_statistics", "unverifiable_source"]
    },

    # 5) Fake – Liberal – High credibility – High emotion
    # IMPERSONATOR: Fake Robert Reich account (misspelled handle)
    {
        "item_id": "fake_05",
        "veracity": "fake",
        "political_lean": "liberal",
        "source_credibility": "high",
        "emotional_valence": "high",
        "display_name": "Robert Reich",
        "handle": "@RobertReiich",  # Double 'i' - impersonator
        "badge_type": "blue",  # Blue badge for impersonator
        "source_logo": "/fakenews/logos/robert.png",
        "thumbnail": "/fakenews/thumbnails/image.jpg",
        "thumbnail_type": "image",
        "tweet_content": "If this internal memo is genuine, executives at a major hospital chain knowingly refused life‑saving treatment to uninsured patients to protect quarterly profits 💔 Doctors describe patients being turned away from ERs. 'This is not medicine, it's business.' This is corporate greed killing people.\n#Healthcare #PatientsOverProfits #CorporateGreed",
        "topic": "health",
        "engagement_counts": {"replies": 12000, "retweets": 67000, "likes": 198000, "views": 3200000},
        "fact_check_status": "false",
        "manipulation_techniques": ["impersonation", "unverified_document", "emotional_language", "hedging_language"]
    },

    # 6) Fake – Liberal – High credibility – Neutral
    # IMPERSONATOR: Fake Neil deGrasse Tyson account (misspelled handle)
    {
        "item_id": "fake_06",
        "veracity": "fake",
        "political_lean": "liberal",
        "source_credibility": "high",
        "emotional_valence": "neutral",
        "display_name": "Neil deGrasse Tyson",
        "handle": "@nabordeGrasse",  # Wrong prefix - impersonator (real is @neiltyson)
        "badge_type": "blue",  # Blue badge for impersonator
        "source_logo": "/fakenews/logos/tyson.png",
        "thumbnail": "/fakenews/thumbnails/video.mp4",
        "thumbnail_type": "video",
        "tweet_content": "A document circulating online claims that a single environmental regulation caused a 70% drop in industrial emissions last year. The text does not identify authors or data sources, and the claim has not appeared in any peer‑reviewed research. Science requires verifiable data.\n#ClimatePolicy #Science #Data",
        "topic": "science",
        "engagement_counts": {"replies": 3400, "retweets": 12000, "likes": 56000, "views": 1200000},
        "fact_check_status": "false",
        "manipulation_techniques": ["impersonation", "false_statistics", "unverifiable_claims"]
    },

    # 7) Fake – Liberal – Low credibility – High emotion
    {
        "item_id": "fake_07",
        "veracity": "fake",
        "political_lean": "liberal",
        "source_credibility": "low",
        "emotional_valence": "high",
        "display_name": "People's Climate Ledger",
        "handle": "@PeoplesClimateLG",
        "badge_type": "grey",
        "source_logo": "/fakenews/logos/climate_watch.png",
        "thumbnail": "/fakenews/thumbnails/video.mp4",
        "thumbnail_type": "video",
        "tweet_content": "EXPOSED 😡: 'Leaked' files allegedly show an energy corporation paying off ministers in a small coastal nation to block renewable projects and keep burning fossil fuels. Entire communities are being sold out so executives can squeeze out a few more years of profit.\n#Corruption #ClimateJustice #FossilFuels",
        "topic": "politics",
        "engagement_counts": {"replies": 5600, "retweets": 23000, "likes": 67000, "views": 1100000},
        "fact_check_status": "false",
        "manipulation_techniques": ["unverified_leaks", "emotional_language", "vague_attribution"]
    },

    # 8) Fake – Liberal – Low credibility – Neutral
    {
        "item_id": "fake_08",
        "veracity": "fake",
        "political_lean": "liberal",
        "source_credibility": "low",
        "emotional_valence": "neutral",
        "display_name": "City Equity Now",
        "handle": "@CityEquityNow",
        "badge_type": "grey",
        "source_logo": "/fakenews/logos/city_equity.png",
        "thumbnail": None,
        "thumbnail_type": None,
        "tweet_content": "A viral graphic claims that cities which doubled their social housing budgets cut homelessness by 90% in just one year. The post lists no underlying data, and researchers contacted by our page said they were unable to locate any study matching those figures.\n#Housing #Homelessness",
        "topic": "social",
        "engagement_counts": {"replies": 450, "retweets": 1200, "likes": 3400, "views": 89000},
        "fact_check_status": "false",
        "manipulation_techniques": ["false_statistics", "unverifiable_source"]
    },

    # ============================================================================
    # REAL NEWS (8 items)
    # ============================================================================

    # 9) Real – Conservative – High credibility – High emotion
    {
        "item_id": "real_01",
        "veracity": "real",
        "political_lean": "conservative",
        "source_credibility": "high",
        "emotional_valence": "high",
        "display_name": "National Agriculture Desk",
        "handle": "@NatAgDesk",
        "badge_type": "blue",
        "source_logo": "/fakenews/logos/agriculture.png",
        "thumbnail": "/fakenews/thumbnails/image.jpg",
        "thumbnail_type": "image",
        "tweet_content": "Farmers in several border regions report being squeezed by new tariffs and rising transport costs, warning that another season like this could force small family farms to close 😟 Producer groups say relief has been slow, even as officials defend the trade measures as 'necessary leverage'.\n#Agriculture #Tariffs #Rural",
        "topic": "economics",
        "engagement_counts": {"replies": 3400, "retweets": 8900, "likes": 23000, "views": 670000},
        "fact_check_status": "true",
        "manipulation_techniques": []
    },

    # 10) Real – Conservative – High credibility – Neutral
    {
        "item_id": "real_02",
        "veracity": "real",
        "political_lean": "conservative",
        "source_credibility": "high",
        "emotional_valence": "neutral",
        "display_name": "Global Markets Service",
        "handle": "@GlobalMktsSvc",
        "badge_type": "blue",
        "source_logo": "/fakenews/logos/economy.png",
        "thumbnail": "/fakenews/thumbnails/video.mp4",
        "thumbnail_type": "video",
        "tweet_content": "The central bank announced a modest interest rate increase today, citing stable employment and gradual inflation as key factors. Market analysts expect the decision to cool borrowing in certain sectors while leaving overall consumer demand mostly unchanged in the near term.\n#InterestRates #Economy",
        "topic": "economics",
        "engagement_counts": {"replies": 234, "retweets": 890, "likes": 2100, "views": 145000},
        "fact_check_status": "true",
        "manipulation_techniques": []
    },

    # 11) Real – Conservative – Low credibility – High emotion
    {
        "item_id": "real_03",
        "veracity": "real",
        "political_lean": "conservative",
        "source_credibility": "low",
        "emotional_valence": "high",
        "display_name": "Small Business Pulse",
        "handle": "@SmallBizPulse",
        "badge_type": "blue",
        "source_logo": "/fakenews/logos/support_local.png",
        "thumbnail": "/fakenews/thumbnails/video.mp4",
        "thumbnail_type": "video",
        "tweet_content": "Another wave of new rules is hitting local shops hard 😡 Owners tell us they're drowning in compliance paperwork while large corporations hire teams of lawyers. Many say that if costs keep rising, their family‑run businesses may not survive another year.\n#SmallBusiness #Regulations #SupportLocal",
        "topic": "business",
        "engagement_counts": {"replies": 890, "retweets": 3400, "likes": 8900, "views": 234000},
        "fact_check_status": "true",
        "manipulation_techniques": []
    },

    # 12) Real – Conservative – Low credibility – Neutral
    {
        "item_id": "real_04",
        "veracity": "real",
        "political_lean": "conservative",
        "source_credibility": "low",
        "emotional_valence": "neutral",
        "display_name": "Digital Parenting Forum",
        "handle": "@DigParentForum",
        "badge_type": "blue",
        "source_logo": "/fakenews/logos/parenting.png",
        "thumbnail": None,
        "thumbnail_type": None,
        "tweet_content": "In a recent national poll, a majority of parents reported concern about social media's impact on teens' sleep, attention, and mood. Researchers note that outcomes vary widely depending on how platforms are used and the level of guidance and monitoring at home.\n#Parenting #SocialMedia",
        "topic": "social",
        "engagement_counts": {"replies": 450, "retweets": 1800, "likes": 4500, "views": 123000},
        "fact_check_status": "true",
        "manipulation_techniques": []
    },

    # 13) Real – Liberal – High credibility – High emotion
    {
        "item_id": "real_05",
        "veracity": "real",
        "political_lean": "liberal",
        "source_credibility": "high",
        "emotional_valence": "high",
        "display_name": "International Relief Monitor",
        "handle": "@ReliefMonitor",
        "badge_type": "blue",
        "source_logo": "/fakenews/logos/human_rights.png",
        "thumbnail": "/fakenews/thumbnails/image.jpg",
        "thumbnail_type": "image",
        "tweet_content": "Field teams describe families waiting hours at checkpoints, only to find hospitals without beds or basic supplies 💔 Children with treatable conditions face dangerous delays. Local doctors are urging the creation of safe humanitarian corridors so patients can reach care.\n#HumanRights #Healthcare #HumanitarianAid",
        "topic": "health",
        "engagement_counts": {"replies": 5600, "retweets": 23000, "likes": 56000, "views": 890000},
        "fact_check_status": "true",
        "manipulation_techniques": []
    },

    # 14) Real – Liberal – High credibility – Neutral
    {
        "item_id": "real_06",
        "veracity": "real",
        "political_lean": "liberal",
        "source_credibility": "high",
        "emotional_valence": "neutral",
        "display_name": "Urban Demographics Institute",
        "handle": "@UrbanDemoInst",
        "badge_type": "blue",
        "source_logo": "/fakenews/logos/urban.png",
        "thumbnail": None,
        "thumbnail_type": None,
        "tweet_content": "New census data show urban populations have grown faster than rural areas for several consecutive years. Demographers link the trend to job availability, access to services, and younger adults relocating to metropolitan regions for education and work opportunities.\n#Urbanization #Demographics",
        "topic": "demographics",
        "engagement_counts": {"replies": 178, "retweets": 560, "likes": 1200, "views": 67000},
        "fact_check_status": "true",
        "manipulation_techniques": []
    },

    # 15) Real – Liberal – Low credibility – High emotion
    {
        "item_id": "real_07",
        "veracity": "real",
        "political_lean": "liberal",
        "source_credibility": "low",
        "emotional_valence": "high",
        "display_name": "Climate Frontline Voices",
        "handle": "@ClimateFrontline",
        "badge_type": "blue",
        "source_logo": "/fakenews/logos/climate.png",
        "thumbnail": "/fakenews/thumbnails/video.mp4",
        "thumbnail_type": "video",
        "tweet_content": "Residents in low‑lying coastal towns are already seeing homes flooded more often and streets swallowed by high tides 🚨 Families who have lived there for generations are being forced to move. Local activists say they feel abandoned in the face of rising seas. 😢\n#ClimateCrisis #SeaLevelRise #FrontlineCommunities",
        "topic": "science",
        "engagement_counts": {"replies": 1200, "retweets": 5600, "likes": 12000, "views": 345000},
        "fact_check_status": "true",
        "manipulation_techniques": []
    },

    # 16) Real – Liberal – Low credibility – Neutral
    {
        "item_id": "real_08",
        "veracity": "real",
        "political_lean": "liberal",
        "source_credibility": "low",
        "emotional_valence": "neutral",
        "display_name": "Fair Wage Observatory",
        "handle": "@FairWageObs",
        "badge_type": "blue",
        "source_logo": "/fakenews/logos/fair_wage.png",
        "thumbnail": "/fakenews/thumbnails/video.mp4",
        "thumbnail_type": "video",
        "tweet_content": "An economic study of several regions finds that areas which raised the minimum wage saw modest reductions in measured poverty rates over multiple years. The authors caution that wage policy interacts with housing costs, taxation, and social benefits when it comes to overall living standards.\n#MinimumWage #Poverty #Economics",
        "topic": "economics",
        "engagement_counts": {"replies": 340, "retweets": 1100, "likes": 2800, "views": 89000},
        "fact_check_status": "true",
        "manipulation_techniques": []
    }
]


async def seed_news_items():
    """Seed the database with news items."""
    client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
    db = client[FAKENEWS_DB_NAME]

    # Clear existing items
    await db.news_items.delete_many({})
    print("Cleared existing news items")

    # Insert new items
    result = await db.news_items.insert_many(NEWS_ITEMS)
    print(f"Inserted {len(result.inserted_ids)} news items")

    # Verify the factorial design
    items = await db.news_items.find().to_list(length=100)

    # Count by veracity
    fake_count = sum(1 for i in items if i["veracity"] == "fake")
    real_count = sum(1 for i in items if i["veracity"] == "real")
    print(f"\nVeracity: Fake={fake_count}, Real={real_count}")

    # Count by political lean
    lib_count = sum(1 for i in items if i["political_lean"] == "liberal")
    con_count = sum(1 for i in items if i["political_lean"] == "conservative")
    print(f"Political Lean: Liberal={lib_count}, Conservative={con_count}")

    # Count by source credibility
    high_count = sum(1 for i in items if i["source_credibility"] == "high")
    low_count = sum(1 for i in items if i["source_credibility"] == "low")
    print(f"Source Credibility: High={high_count}, Low={low_count}")

    # Count by emotional valence
    emo_high = sum(1 for i in items if i["emotional_valence"] == "high")
    emo_neutral = sum(1 for i in items if i["emotional_valence"] == "neutral")
    print(f"Emotional Valence: High={emo_high}, Neutral={emo_neutral}")

    # Count media types
    with_image = sum(1 for i in items if i.get("thumbnail_type") == "image")
    with_video = sum(1 for i in items if i.get("thumbnail_type") == "video")
    text_only = sum(1 for i in items if i.get("thumbnail") is None)
    print(f"\nMedia Distribution: Images={with_image}, Videos={with_video}, Text-only={text_only}")

    # Verify full factorial crossing
    print("\n--- Factorial Design Verification ---")
    for veracity in ["fake", "real"]:
        for political in ["liberal", "conservative"]:
            for credibility in ["high", "low"]:
                for emotion in ["high", "neutral"]:
                    count = sum(
                        1 for i in items
                        if i["veracity"] == veracity
                        and i["political_lean"] == political
                        and i["source_credibility"] == credibility
                        and i["emotional_valence"] == emotion
                    )
                    print(f"{veracity:4} | {political:12} | {credibility:4} | {emotion:7} | Count: {count}")

    client.close()
    print("\nSeeding complete!")


if __name__ == "__main__":
    asyncio.run(seed_news_items())
