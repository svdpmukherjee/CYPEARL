"""
CYPEARL Fake News / Misinformation Experiment API Routes

Endpoints for the fake news susceptibility experiment.
All data is stored in the fake-news scenario database.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
import random
import uuid
import math

from database import get_fakenews_db
from models.fake_news import (
    NewsItem,
    NewsEvaluationSession,
    FakeNewsParticipant,
    FakeNewsPreSurvey,
    FakeNewsPostSurvey,
    NewsItemActionRequest,
    ParticipantProgressResponse,
    calculate_congruence,
    assign_question_order,
)

router = APIRouter(prefix="/fake-news", tags=["fake-news"])


# ============================================================================
# Helper Functions
# ============================================================================

async def get_news_item_from_db(item_id: str):
    """Fetch a news item from the database by item_id."""
    db = get_fakenews_db()
    item = await db.news_items.find_one({"item_id": item_id})
    return item


async def get_all_news_items_from_db():
    """Fetch all news items from the database."""
    db = get_fakenews_db()
    items = await db.news_items.find().to_list(length=100)
    return items


async def get_all_item_ids():
    """Get list of all news item IDs from database."""
    db = get_fakenews_db()
    items = await db.news_items.find({}, {"item_id": 1}).to_list(length=100)
    return [i["item_id"] for i in items]


# ============================================================================
# Authentication & Session Management
# ============================================================================

from pydantic import BaseModel


class FakeNewsLoginRequest(BaseModel):
    prolific_id: str
    session_id: Optional[str] = None
    scenario: Optional[str] = None
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None


@router.post("/auth/login")
async def login(body: FakeNewsLoginRequest):
    """
    Create or retrieve participant session for fake news experiment.
    """
    db = get_fakenews_db()

    # Check if participant already exists
    existing = await db.participants.find_one({"prolific_id": body.prolific_id})
    if existing:
        return {
            "participant_id": existing["participant_id"],
            "is_new": False,
            "current_item_index": existing.get("current_item_index", 0),
            "is_finished": existing.get("is_finished", False),
            "question_order": existing.get("question_order", "accuracy_first")
        }

    # Create new participant
    participant_id = str(uuid.uuid4())

    # Randomize item order for this participant
    item_ids = await get_all_item_ids()
    if not item_ids:
        raise HTTPException(
            status_code=500,
            detail="No news items found in database. Run seed_fake_news.py first."
        )
    random.shuffle(item_ids)

    # Assign question order for counterbalancing
    question_order = assign_question_order(participant_id)

    participant = FakeNewsParticipant(
        participant_id=participant_id,
        prolific_id=body.prolific_id,
        session_id=body.session_id or "",
        item_order=item_ids,
        question_order=question_order,
        user_agent=body.user_agent,
        screen_resolution=body.screen_resolution
    )

    await db.participants.insert_one(participant.model_dump())

    return {
        "participant_id": participant_id,
        "is_new": True,
        "current_item_index": 0,
        "is_finished": False,
        "question_order": question_order
    }


# ============================================================================
# Political Ideology Update (after pre-survey)
# ============================================================================

class PoliticalIdeologyUpdate(BaseModel):
    political_ideology: int  # 1-7 scale
    party_id: Optional[str] = None


@router.post("/ideology/{participant_id}")
async def update_political_ideology(participant_id: str, data: PoliticalIdeologyUpdate):
    """
    Update participant's political ideology after pre-survey.
    This is needed for calculating congruence for each news item.
    """
    db = get_fakenews_db()

    result = await db.participants.update_one(
        {"participant_id": participant_id},
        {
            "$set": {
                "political_ideology": data.political_ideology,
                "party_id": data.party_id,
                "updated_at": datetime.utcnow()
            }
        }
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Participant not found")

    return {"status": "success", "message": "Political ideology updated"}


# ============================================================================
# Survey Endpoints
# ============================================================================

@router.post("/survey/pre")
async def submit_pre_survey(survey_data: dict):
    """Submit pre-experiment survey data."""
    db = get_fakenews_db()

    participant_id = survey_data.get("participant_id")
    if not participant_id:
        raise HTTPException(status_code=400, detail="participant_id required")

    # Add metadata
    survey_data["submitted_at"] = datetime.utcnow()

    # Store in pre_survey_responses collection
    await db.pre_survey_responses.update_one(
        {"participant_id": participant_id},
        {"$set": survey_data},
        upsert=True
    )

    # Update participant record with political ideology if present
    update_fields = {
        "pre_survey_completed": True,
        "updated_at": datetime.utcnow()
    }

    if "political_ideology" in survey_data:
        update_fields["political_ideology"] = survey_data["political_ideology"]
    if "party_id" in survey_data:
        update_fields["party_id"] = survey_data["party_id"]

    await db.participants.update_one(
        {"participant_id": participant_id},
        {"$set": update_fields}
    )

    return {"status": "success", "message": "Pre-survey saved"}


@router.post("/survey/post/{participant_id}")
async def submit_post_survey(participant_id: str, survey_data: dict):
    """Submit post-experiment survey data."""
    db = get_fakenews_db()

    # Add metadata
    survey_data["participant_id"] = participant_id
    survey_data["submitted_at"] = datetime.utcnow()

    # Store in post_survey_responses collection
    await db.post_survey_responses.update_one(
        {"participant_id": participant_id},
        {"$set": survey_data},
        upsert=True
    )

    # Update participant record
    await db.participants.update_one(
        {"participant_id": participant_id},
        {"$set": {"post_survey_completed": True, "updated_at": datetime.utcnow()}}
    )

    return {"status": "success", "message": "Post-survey saved"}


# ============================================================================
# News Item Management
# ============================================================================

@router.get("/items")
async def get_all_items():
    """Get all news item definitions from database (for admin/debugging)."""
    items = await get_all_news_items_from_db()
    for item in items:
        item.pop("_id", None)
    return {"items": items}


@router.get("/items/current/{participant_id}")
async def get_current_item(participant_id: str):
    """Get the current news item for a participant, with congruence calculated."""
    db = get_fakenews_db()

    participant = await db.participants.find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    item_order = participant.get("item_order", [])
    total_items = len(item_order)

    if participant.get("is_finished", False):
        return {
            "is_finished": True,
            "item": None,
            "progress": {
                "current": participant.get("items_completed", 0),
                "total": total_items
            }
        }

    current_index = participant.get("current_item_index", 0)

    if current_index >= total_items:
        return {
            "is_finished": True,
            "item": None,
            "progress": {
                "current": total_items,
                "total": total_items
            }
        }

    current_item_id = item_order[current_index]

    # Fetch item from database
    item = await get_news_item_from_db(current_item_id)

    if item:
        item.pop("_id", None)

        # Calculate congruence based on participant's political ideology
        political_ideology = participant.get("political_ideology")
        if political_ideology:
            congruence = calculate_congruence(
                political_ideology,
                item.get("political_lean", "neutral")
            )
            item["congruence"] = congruence
        else:
            item["congruence"] = "neutral"

    return {
        "is_finished": False,
        "item": item,
        "progress": {
            "current": current_index,
            "total": total_items
        },
        "question_order": participant.get("question_order", "accuracy_first")
    }


@router.get("/items/{item_id}")
async def get_item(item_id: str):
    """Get a specific news item definition from database."""
    item = await get_news_item_from_db(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    item.pop("_id", None)
    return item


# ============================================================================
# Action Recording
# ============================================================================

@router.post("/action/{participant_id}")
async def submit_item_action(participant_id: str, action_data: NewsItemActionRequest):
    """Submit evaluation and metrics for a completed news item."""
    db = get_fakenews_db()

    # Get participant
    participant = await db.participants.find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    # Get news item from database
    item = await get_news_item_from_db(action_data.item_id)
    if not item:
        raise HTTPException(status_code=404, detail="News item not found")

    # Calculate congruence
    political_ideology = participant.get("political_ideology") or 4  # Default to moderate if None
    congruence = calculate_congruence(
        political_ideology,
        item.get("political_lean", "neutral")
    )

    # Determine ground truth and outcomes
    veracity = item.get("veracity", "fake")
    ground_truth = 1 if veracity == "real" else 0

    # Calculate derived metrics (handle None for demo/testing)
    believed = 1 if action_data.accuracy_rating is not None and action_data.accuracy_rating > 4 else 0
    would_share = 1 if action_data.sharing_intention is not None and action_data.sharing_intention > 4 else 0

    # Correct judgment: for fake news, rating <= 4 is correct; for real news, rating > 4 is correct
    if action_data.accuracy_rating is None:
        correct_judgment = 0
    elif veracity == "fake":
        correct_judgment = 1 if action_data.accuracy_rating <= 4 else 0
    else:
        correct_judgment = 1 if action_data.accuracy_rating > 4 else 0

    # Create session record
    session_data = {
        "item_id": action_data.item_id,
        "participant_id": participant_id,
        "veracity": veracity,
        "ground_truth": ground_truth,
        "political_lean": item.get("political_lean"),
        "congruence": congruence,
        "source_credibility": item.get("source_credibility"),
        "emotional_valence": item.get("emotional_valence"),
        "topic": item.get("topic"),
        "question_order": participant.get("question_order", "accuracy_first"),
        "item_displayed_at": datetime.utcnow(),
        "judgment_completed_at": datetime.utcnow(),
        "reading_time_ms": action_data.reading_time_ms,
        "time_to_accuracy_judgment_ms": action_data.time_to_accuracy_judgment_ms,
        "time_to_sharing_judgment_ms": action_data.time_to_sharing_judgment_ms,
        "accuracy_rating": action_data.accuracy_rating,
        "sharing_intention": action_data.sharing_intention,
        "seen_before": action_data.seen_before,
        "confidence": action_data.confidence,
        # Qualitative responses (for chain-of-thought LLM conditioning)
        "reason": action_data.reason,
        "cues_noticed": action_data.cues_noticed,
        "evaluation_process": action_data.evaluation_process,
        "influencing_factors": action_data.influencing_factors,
        "uncertainty_points": action_data.uncertainty_points,
        # Derived metrics
        "believed": believed,
        "would_share": would_share,
        "correct_judgment": correct_judgment,
        "source_hover": action_data.source_hover,
        "source_hover_time_ms": action_data.source_hover_time_ms,
        "source_click": action_data.source_click,
        "headline_reread": action_data.headline_reread,
        "engagement_hover": action_data.engagement_hover,
        "scroll_depth": action_data.scroll_depth,
        "scroll_events": action_data.scroll_events,
        "hover_events": action_data.hover_events,
    }

    # Save to responses collection
    await db.responses.insert_one(session_data)

    return {
        "status": "success",
        "believed": believed,
        "would_share": would_share,
        "correct_judgment": correct_judgment
    }


@router.post("/complete/{participant_id}")
async def complete_item(participant_id: str):
    """Mark current item as complete and advance to next."""
    db = get_fakenews_db()

    participant = await db.participants.find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    current_index = participant.get("current_item_index", 0)
    items_completed = participant.get("items_completed", 0)
    item_order = participant.get("item_order", [])

    new_index = current_index + 1
    new_completed = items_completed + 1
    is_finished = new_index >= len(item_order)

    # Update participant
    await db.participants.update_one(
        {"participant_id": participant_id},
        {
            "$set": {
                "current_item_index": new_index,
                "items_completed": new_completed,
                "is_finished": is_finished,
                "updated_at": datetime.utcnow()
            }
        }
    )

    # If finished, calculate aggregate metrics
    if is_finished:
        await calculate_aggregate_metrics(participant_id)

    return {
        "status": "success",
        "current_item_index": new_index,
        "items_completed": new_completed,
        "is_finished": is_finished
    }


# ============================================================================
# Progress & Metrics
# ============================================================================

@router.get("/progress/{participant_id}")
async def get_progress(participant_id: str) -> ParticipantProgressResponse:
    """Get participant progress."""
    db = get_fakenews_db()

    participant = await db.participants.find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    item_order = participant.get("item_order", [])
    current_index = participant.get("current_item_index", 0)
    next_item_id = item_order[current_index] if current_index < len(item_order) else None

    return ParticipantProgressResponse(
        participant_id=participant_id,
        current_item_index=current_index,
        items_completed=participant.get("items_completed", 0),
        total_items=len(item_order),
        is_finished=participant.get("is_finished", False),
        next_item_id=next_item_id
    )


async def calculate_aggregate_metrics(participant_id: str):
    """Calculate and store aggregate outcome metrics for a participant."""
    db = get_fakenews_db()

    # Get all responses for this participant
    responses = await db.responses.find({"participant_id": participant_id}).to_list(length=100)

    if not responses:
        return

    # Separate by veracity
    fake_responses = [r for r in responses if r.get("veracity") == "fake"]
    real_responses = [r for r in responses if r.get("veracity") == "real"]

    # Calculate belief rates
    fake_belief_rate = None
    real_belief_rate = None

    if fake_responses:
        fake_believed = sum(1 for r in fake_responses if r.get("believed", 0) == 1)
        fake_belief_rate = fake_believed / len(fake_responses)

    if real_responses:
        real_believed = sum(1 for r in real_responses if r.get("believed", 0) == 1)
        real_belief_rate = real_believed / len(real_responses)

    # Accuracy discernment
    accuracy_discernment = None
    if fake_belief_rate is not None and real_belief_rate is not None:
        accuracy_discernment = real_belief_rate - fake_belief_rate

    # Calculate share rates
    fake_share_rate = None
    real_share_rate = None

    if fake_responses:
        fake_shared = sum(1 for r in fake_responses if r.get("would_share", 0) == 1)
        fake_share_rate = fake_shared / len(fake_responses)

    if real_responses:
        real_shared = sum(1 for r in real_responses if r.get("would_share", 0) == 1)
        real_share_rate = real_shared / len(real_responses)

    # Sharing discernment
    sharing_discernment = None
    if fake_share_rate is not None and real_share_rate is not None:
        sharing_discernment = real_share_rate - fake_share_rate

    # Signal Detection Theory metrics
    d_prime = None
    criterion = None

    if real_belief_rate is not None and fake_belief_rate is not None:
        # Hit rate: correctly believing real news
        hit_rate = max(0.01, min(0.99, real_belief_rate))
        # False alarm rate: incorrectly believing fake news
        fa_rate = max(0.01, min(0.99, fake_belief_rate))

        # Calculate z-scores (inverse normal CDF approximation)
        def norm_ppf(p):
            """Approximation of inverse normal CDF"""
            if p <= 0:
                return -3.0
            if p >= 1:
                return 3.0
            # Rational approximation
            a = [
                -3.969683028665376e+01, 2.209460984245205e+02,
                -2.759285104469687e+02, 1.383577518672690e+02,
                -3.066479806614716e+01, 2.506628277459239e+00
            ]
            b = [
                -5.447609879822406e+01, 1.615858368580409e+02,
                -1.556989798598866e+02, 6.680131188771972e+01,
                -1.328068155288572e+01
            ]
            c = [
                -7.784894002430293e-03, -3.223964580411365e-01,
                -2.400758277161838e+00, -2.549732539343734e+00,
                4.374664141464968e+00, 2.938163982698783e+00
            ]
            d = [
                7.784695709041462e-03, 3.224671290700398e-01,
                2.445134137142996e+00, 3.754408661907416e+00
            ]
            p_low = 0.02425
            p_high = 1 - p_low

            if p < p_low:
                q = math.sqrt(-2 * math.log(p))
                return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                       ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
            elif p <= p_high:
                q = p - 0.5
                r = q * q
                return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                       (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
            else:
                q = math.sqrt(-2 * math.log(1 - p))
                return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

        z_hit = norm_ppf(hit_rate)
        z_fa = norm_ppf(fa_rate)

        # d' (discrimination)
        d_prime = z_hit - z_fa

        # c (criterion/bias)
        criterion = -0.5 * (z_hit + z_fa)

    # Partisan bias (d' congruent - d' incongruent)
    congruence_bias = None
    congruent_responses = [r for r in responses if r.get("congruence") == "congruent"]
    incongruent_responses = [r for r in responses if r.get("congruence") == "incongruent"]

    if len(congruent_responses) >= 4 and len(incongruent_responses) >= 4:
        # Calculate d' for congruent items
        cong_real = [r for r in congruent_responses if r.get("veracity") == "real"]
        cong_fake = [r for r in congruent_responses if r.get("veracity") == "fake"]

        if cong_real and cong_fake:
            cong_hit = sum(1 for r in cong_real if r.get("believed", 0) == 1) / len(cong_real)
            cong_fa = sum(1 for r in cong_fake if r.get("believed", 0) == 1) / len(cong_fake)
            cong_hit = max(0.01, min(0.99, cong_hit))
            cong_fa = max(0.01, min(0.99, cong_fa))

            # Calculate d' for incongruent items
            incong_real = [r for r in incongruent_responses if r.get("veracity") == "real"]
            incong_fake = [r for r in incongruent_responses if r.get("veracity") == "fake"]

            if incong_real and incong_fake:
                incong_hit = sum(1 for r in incong_real if r.get("believed", 0) == 1) / len(incong_real)
                incong_fa = sum(1 for r in incong_fake if r.get("believed", 0) == 1) / len(incong_fake)
                incong_hit = max(0.01, min(0.99, incong_hit))
                incong_fa = max(0.01, min(0.99, incong_fa))

                # Using the same norm_ppf function
                def norm_ppf(p):
                    if p <= 0:
                        return -3.0
                    if p >= 1:
                        return 3.0
                    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
                    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
                    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
                    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
                    p_low = 0.02425
                    p_high = 1 - p_low
                    if p < p_low:
                        q = math.sqrt(-2 * math.log(p))
                        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
                    elif p <= p_high:
                        q = p - 0.5
                        r = q * q
                        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
                    else:
                        q = math.sqrt(-2 * math.log(1 - p))
                        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

                d_prime_cong = norm_ppf(cong_hit) - norm_ppf(cong_fa)
                d_prime_incong = norm_ppf(incong_hit) - norm_ppf(incong_fa)
                congruence_bias = d_prime_cong - d_prime_incong

    # Process metrics
    reading_times = [r.get("reading_time_ms", 0) for r in responses if r.get("reading_time_ms")]
    mean_reading_time_ms = sum(reading_times) / len(reading_times) if reading_times else None

    source_inspections = sum(1 for r in responses if r.get("source_hover", False) or r.get("source_click", False))
    source_inspection_rate = source_inspections / len(responses) if responses else None

    # Update participant with metrics
    await db.participants.update_one(
        {"participant_id": participant_id},
        {
            "$set": {
                "fake_belief_rate": fake_belief_rate,
                "real_belief_rate": real_belief_rate,
                "accuracy_discernment": accuracy_discernment,
                "fake_share_rate": fake_share_rate,
                "real_share_rate": real_share_rate,
                "sharing_discernment": sharing_discernment,
                "d_prime": d_prime,
                "criterion": criterion,
                "congruence_bias": congruence_bias,
                "mean_reading_time_ms": mean_reading_time_ms,
                "source_inspection_rate": source_inspection_rate,
                "updated_at": datetime.utcnow()
            }
        }
    )


# ============================================================================
# Export Endpoints
# ============================================================================

@router.get("/export/participants")
async def export_participants():
    """Export all participant data."""
    db = get_fakenews_db()

    participants = await db.participants.find().to_list(length=10000)

    # Join with pre-survey data
    for p in participants:
        pre_survey = await db.pre_survey_responses.find_one(
            {"participant_id": p["participant_id"]}
        )
        if pre_survey:
            for key, value in pre_survey.items():
                if key not in ["_id", "participant_id", "submitted_at"]:
                    p[key] = value

    # Remove MongoDB _id
    for p in participants:
        p.pop("_id", None)

    return {"participants": participants}


@router.get("/export/responses")
async def export_responses():
    """Export all response data."""
    db = get_fakenews_db()

    responses = await db.responses.find().to_list(length=100000)

    # Remove MongoDB _id
    for r in responses:
        r.pop("_id", None)

    return {"responses": responses}


@router.get("/export/participants/csv")
async def export_participants_csv():
    """Export participant data in CSV-ready format."""
    db = get_fakenews_db()

    participants = await db.participants.find().to_list(length=10000)
    pre_surveys = await db.pre_survey_responses.find().to_list(length=10000)

    # Create lookup for pre-surveys
    pre_survey_lookup = {ps["participant_id"]: ps for ps in pre_surveys}

    # Flatten data
    csv_data = []
    for p in participants:
        row = {
            "participant_id": p.get("participant_id"),
            "prolific_id": p.get("prolific_id"),
            "political_ideology": p.get("political_ideology"),
            "party_id": p.get("party_id"),
            "question_order": p.get("question_order"),
            "fake_belief_rate": p.get("fake_belief_rate"),
            "real_belief_rate": p.get("real_belief_rate"),
            "accuracy_discernment": p.get("accuracy_discernment"),
            "fake_share_rate": p.get("fake_share_rate"),
            "real_share_rate": p.get("real_share_rate"),
            "sharing_discernment": p.get("sharing_discernment"),
            "d_prime": p.get("d_prime"),
            "criterion": p.get("criterion"),
            "congruence_bias": p.get("congruence_bias"),
            "mean_reading_time_ms": p.get("mean_reading_time_ms"),
            "source_inspection_rate": p.get("source_inspection_rate"),
        }

        # Add pre-survey data
        pre_survey = pre_survey_lookup.get(p["participant_id"], {})
        row.update({
            "age": pre_survey.get("age"),
            "gender": pre_survey.get("gender"),
            "education": pre_survey.get("education"),
            "technical_field": pre_survey.get("technical_field"),
            "crt_score": pre_survey.get("crt_score"),
            "need_for_cognition": pre_survey.get("need_for_cognition"),
            "working_memory": pre_survey.get("working_memory"),
            "big5_extraversion": pre_survey.get("big5_extraversion"),
            "big5_agreeableness": pre_survey.get("big5_agreeableness"),
            "big5_conscientiousness": pre_survey.get("big5_conscientiousness"),
            "big5_neuroticism": pre_survey.get("big5_neuroticism"),
            "big5_openness": pre_survey.get("big5_openness"),
            "impulsivity_total": pre_survey.get("impulsivity_total"),
            "sensation_seeking": pre_survey.get("sensation_seeking"),
            "trust_propensity": pre_survey.get("trust_propensity"),
            "risk_taking": pre_survey.get("risk_taking"),
            "authority_susceptibility": pre_survey.get("authority_susceptibility"),
            "urgency_susceptibility": pre_survey.get("urgency_susceptibility"),
            "scarcity_susceptibility": pre_survey.get("scarcity_susceptibility"),
            "conspiracy_mentality": pre_survey.get("conspiracy_mentality"),
            "news_media_literacy": pre_survey.get("news_media_literacy"),
            "bullshit_receptivity": pre_survey.get("bullshit_receptivity"),
            "trust_in_media": pre_survey.get("trust_in_media"),
            "fomo": pre_survey.get("fomo"),
            "fake_news_detection_efficacy": pre_survey.get("fake_news_detection_efficacy"),
            "perceived_misinfo_harm": pre_survey.get("perceived_misinfo_harm"),
        })

        csv_data.append(row)

    return {"data": csv_data}


@router.get("/export/responses/csv")
async def export_responses_csv():
    """Export response data in CSV-ready format."""
    db = get_fakenews_db()

    responses = await db.responses.find().to_list(length=100000)

    csv_data = []
    for r in responses:
        row = {
            "participant_id": r.get("participant_id"),
            "item_id": r.get("item_id"),
            "veracity": r.get("veracity"),
            "ground_truth": r.get("ground_truth"),
            "political_lean": r.get("political_lean"),
            "congruence": r.get("congruence"),
            "source_credibility": r.get("source_credibility"),
            "emotional_valence": r.get("emotional_valence"),
            "topic": r.get("topic"),
            "question_order": r.get("question_order"),
            "accuracy_rating": r.get("accuracy_rating"),
            "sharing_intention": r.get("sharing_intention"),
            "believed": r.get("believed"),
            "would_share": r.get("would_share"),
            "correct_judgment": r.get("correct_judgment"),
            "seen_before": r.get("seen_before"),
            "confidence": r.get("confidence"),
            # Qualitative responses
            "reason": r.get("reason"),
            "cues_noticed": r.get("cues_noticed"),
            "evaluation_process": r.get("evaluation_process"),
            "influencing_factors": r.get("influencing_factors"),
            "uncertainty_points": r.get("uncertainty_points"),
            # Timing and behavioral metrics
            "reading_time_ms": r.get("reading_time_ms"),
            "time_to_accuracy_ms": r.get("time_to_accuracy_judgment_ms"),
            "time_to_sharing_ms": r.get("time_to_sharing_judgment_ms"),
            "source_hover": r.get("source_hover"),
            "source_hover_time_ms": r.get("source_hover_time_ms"),
            "source_click": r.get("source_click"),
            "headline_reread": r.get("headline_reread"),
            "engagement_hover": r.get("engagement_hover"),
            "scroll_depth": r.get("scroll_depth"),
        }
        csv_data.append(row)

    return {"data": csv_data}
