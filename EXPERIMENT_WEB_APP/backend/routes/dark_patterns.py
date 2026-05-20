"""
CYPEARL Dark Patterns Experiment API Routes

Endpoints for the dark patterns UI/UX experiment.
All data is stored in the dark-patterns scenario database.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
import random
import uuid

from database import get_darkpatterns_db
from models.dark_patterns import (
    DarkPatternTask,
    DarkPatternSession,
    DarkPatternParticipant,
    DarkPatternsPreSurvey,
    DarkPatternsPostSurvey,
    TaskActionRequest,
    ParticipantProgressResponse,
    UIType,
    Intensity,
    VisualManipulation,
    TaskContext,
)

router = APIRouter(prefix="/dark-patterns", tags=["dark-patterns"])

# ============================================================================
# Task Library - Now fetched from database (seeded by seed_dark_patterns.py)
# ============================================================================

# Helper function to get task from database
async def get_task_from_db(task_id: str):
    """Fetch a task from the database by task_id."""
    db = get_darkpatterns_db()
    task = await db.tasks.find_one({"task_id": task_id})
    return task

# Helper function to get all tasks from database
async def get_all_tasks_from_db():
    """Fetch all tasks from the database."""
    db = get_darkpatterns_db()
    tasks = await db.tasks.find().to_list(length=100)
    return tasks

# Helper function to get all task IDs
async def get_all_task_ids():
    """Get list of all task IDs from database."""
    db = get_darkpatterns_db()
    tasks = await db.tasks.find({}, {"task_id": 1}).to_list(length=100)
    return [t["task_id"] for t in tasks]


# ============================================================================
# Authentication & Session Management
# ============================================================================

from pydantic import BaseModel

class DarkPatternsLoginRequest(BaseModel):
    prolific_id: str
    session_id: Optional[str] = None
    scenario: Optional[str] = None
    user_agent: Optional[str] = None
    screen_resolution: Optional[str] = None

@router.post("/auth/login")
async def login(body: DarkPatternsLoginRequest):
    """
    Create or retrieve participant session for dark patterns experiment.
    """
    db = get_darkpatterns_db()

    # Check if participant already exists
    existing = await db.participants.find_one({"prolific_id": body.prolific_id})
    if existing:
        return {
            "participant_id": existing["participant_id"],
            "is_new": False,
            "current_task_index": existing.get("current_task_index", 0),
            "is_finished": existing.get("is_finished", False)
        }

    # Create new participant
    participant_id = str(uuid.uuid4())

    # Randomize task order for this participant - fetch from database
    task_ids = await get_all_task_ids()
    if not task_ids:
        raise HTTPException(status_code=500, detail="No tasks found in database. Run seed_dark_patterns.py first.")
    random.shuffle(task_ids)

    participant = DarkPatternParticipant(
        participant_id=participant_id,
        prolific_id=body.prolific_id,
        session_id=body.session_id or "",
        task_order=task_ids,
        user_agent=body.user_agent,
        screen_resolution=body.screen_resolution
    )

    await db.participants.insert_one(participant.model_dump())

    return {
        "participant_id": participant_id,
        "is_new": True,
        "current_task_index": 0,
        "is_finished": False
    }


# ============================================================================
# Survey Endpoints
# ============================================================================

@router.post("/survey/pre")
async def submit_pre_survey(survey_data: dict):
    """Submit pre-experiment survey data."""
    db = get_darkpatterns_db()

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

    # Update participant record
    await db.participants.update_one(
        {"participant_id": participant_id},
        {"$set": {"pre_survey_completed": True, "updated_at": datetime.utcnow()}}
    )

    return {"status": "success", "message": "Pre-survey saved"}


@router.post("/survey/post/{participant_id}")
async def submit_post_survey(participant_id: str, survey_data: dict):
    """Submit post-experiment survey data."""
    db = get_darkpatterns_db()

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
# Task Management
# ============================================================================

@router.get("/tasks")
async def get_all_tasks():
    """Get all task definitions from database (for admin/debugging)."""
    tasks = await get_all_tasks_from_db()
    # Remove MongoDB _id field
    for task in tasks:
        task.pop("_id", None)
    return {"tasks": tasks}


@router.get("/tasks/current/{participant_id}")
async def get_current_task(participant_id: str):
    """Get the current task for a participant, including scenario_info and ui_content."""
    db = get_darkpatterns_db()

    participant = await db.participants.find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    task_order = participant.get("task_order", [])
    total_tasks = len(task_order)

    if participant.get("is_finished", False):
        return {
            "is_finished": True,
            "task": None,
            "progress": {
                "current": participant.get("tasks_completed", 0),
                "total": total_tasks
            }
        }

    current_index = participant.get("current_task_index", 0)

    if current_index >= total_tasks:
        return {
            "is_finished": True,
            "task": None,
            "progress": {
                "current": total_tasks,
                "total": total_tasks
            }
        }

    current_task_id = task_order[current_index]

    # Fetch task from database (includes scenario_info and ui_content)
    task = await get_task_from_db(current_task_id)

    if task:
        # Remove MongoDB _id field
        task.pop("_id", None)

    return {
        "is_finished": False,
        "task": task,
        "progress": {
            "current": current_index,
            "total": total_tasks
        }
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get a specific task definition from database."""
    task = await get_task_from_db(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    task.pop("_id", None)
    return task


# ============================================================================
# Action Recording
# ============================================================================

@router.post("/action/{participant_id}")
async def submit_task_action(participant_id: str, action_data: TaskActionRequest):
    """Submit action and metrics for a completed task."""
    db = get_darkpatterns_db()

    # Get participant
    participant = await db.participants.find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    # Get task definition from database
    task = await get_task_from_db(action_data.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Determine outcomes using task data from database
    is_dark_pattern = task.get("ui_type") == "dark"
    manipulated = False
    resisted = False
    abandoned = action_data.final_action == "abandon"

    if is_dark_pattern:
        manipulated = action_data.final_action == task.get("manipulated_action")
        resisted = action_data.final_action == task.get("desired_action")
    else:
        resisted = action_data.final_action == task.get("desired_action")

    # Create session record - store as dict for flexibility
    session_data = {
        "task_id": action_data.task_id,
        "participant_id": participant_id,
        "ui_type": task.get("ui_type"),
        "intensity": task.get("intensity"),
        "time_pressure": task.get("time_pressure"),
        "visual_manipulation": task.get("visual_manipulation"),
        "context": task.get("context"),
        "ground_truth": task.get("ground_truth", 1 if is_dark_pattern else 0),
        "task_started_at": datetime.utcnow(),
        "task_completed_at": datetime.utcnow(),
        "task_completion_time_ms": action_data.task_completion_time_ms,
        "time_to_first_action_ms": action_data.time_to_first_action_ms,
        "final_action": action_data.final_action,
        "manipulated": manipulated,
        "resisted": resisted,
        "abandoned": abandoned,
        "scroll_depth": action_data.scroll_depth,
        "scroll_to_target_ms": action_data.scroll_to_target_ms,
        "fine_print_expand_count": action_data.fine_print_expand_count,
        "fine_print_hover_time_ms": action_data.fine_print_hover_time_ms,
        "option_hover_count": action_data.option_hover_count,
        "correct_option_hover": action_data.correct_option_hover,
        "correct_option_hover_time_ms": action_data.correct_option_hover_time_ms,
        "backtrack_count": action_data.backtrack_count,
        "click_path": action_data.click_path,
        "click_path_length": len(action_data.click_path) if action_data.click_path else 0,
        "steps_completed": action_data.steps_completed,
        "steps_total": action_data.steps_total,
        "perceived_difficulty": action_data.perceived_difficulty,
        "noticed_unusual": action_data.noticed_unusual,
        "confidence_rating": action_data.confidence_rating,
        # Qualitative feedback fields
        "reason": getattr(action_data, 'reason', None),
        "details_noticed": getattr(action_data, 'details_noticed', None),
        "decision_reason": getattr(action_data, 'decision_reason', None),
        "what_influenced": getattr(action_data, 'what_influenced', None),
        "frustration_points": getattr(action_data, 'frustration_points', None),
    }

    # Save to responses collection
    await db.responses.insert_one(session_data)

    return {
        "status": "success",
        "manipulated": manipulated,
        "resisted": resisted
    }


@router.post("/complete/{participant_id}")
async def complete_task(participant_id: str):
    """Mark current task as complete and advance to next."""
    db = get_darkpatterns_db()

    participant = await db.participants.find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    current_index = participant.get("current_task_index", 0)
    tasks_completed = participant.get("tasks_completed", 0)
    task_order = participant.get("task_order", [])

    new_index = current_index + 1
    new_completed = tasks_completed + 1
    is_finished = new_index >= len(task_order)

    # Update participant
    await db.participants.update_one(
        {"participant_id": participant_id},
        {
            "$set": {
                "current_task_index": new_index,
                "tasks_completed": new_completed,
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
        "current_task_index": new_index,
        "tasks_completed": new_completed,
        "is_finished": is_finished
    }


# ============================================================================
# Progress & Metrics
# ============================================================================

@router.get("/progress/{participant_id}")
async def get_progress(participant_id: str) -> ParticipantProgressResponse:
    """Get participant progress."""
    db = get_darkpatterns_db()

    participant = await db.participants.find_one({"participant_id": participant_id})
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    task_order = participant.get("task_order", [])
    current_index = participant.get("current_task_index", 0)
    next_task_id = task_order[current_index] if current_index < len(task_order) else None

    return ParticipantProgressResponse(
        participant_id=participant_id,
        current_task_index=current_index,
        tasks_completed=participant.get("tasks_completed", 0),
        total_tasks=len(task_order),
        is_finished=participant.get("is_finished", False),
        next_task_id=next_task_id
    )


async def calculate_aggregate_metrics(participant_id: str):
    """Calculate and store aggregate outcome metrics for a participant."""
    db = get_darkpatterns_db()

    # Get all responses for this participant
    responses = await db.responses.find({"participant_id": participant_id}).to_list(length=100)

    if not responses:
        return

    # Calculate metrics
    dark_responses = [r for r in responses if r.get("ground_truth") == 1]
    clean_responses = [r for r in responses if r.get("ground_truth") == 0]

    # Manipulation rate (% of dark pattern tasks where manipulated)
    manipulation_rate = None
    if dark_responses:
        manipulated_count = sum(1 for r in dark_responses if r.get("manipulated", False))
        manipulation_rate = manipulated_count / len(dark_responses)

    # Resistance rate (% of all tasks where user achieved desired outcome)
    resistance_count = sum(1 for r in responses if r.get("resisted", False))
    resistance_rate = resistance_count / len(responses) if responses else None

    # Detection rate (% where user noticed unusual - for dark patterns)
    detection_rate = None
    if dark_responses:
        noticed_count = sum(1 for r in dark_responses if r.get("noticed_unusual", False))
        detection_rate = noticed_count / len(dark_responses)

    # False alarm rate (% of clean UIs flagged as unusual)
    false_alarm_rate = None
    if clean_responses:
        false_alarm_count = sum(1 for r in clean_responses if r.get("noticed_unusual", False))
        false_alarm_rate = false_alarm_count / len(clean_responses)

    # Mean task time
    task_times = [r.get("task_completion_time_ms", 0) for r in responses if r.get("task_completion_time_ms")]
    mean_task_time_ms = sum(task_times) / len(task_times) if task_times else None

    # Mean scroll depth
    scroll_depths = [r.get("scroll_depth", 0) for r in responses]
    mean_scroll_depth = sum(scroll_depths) / len(scroll_depths) if scroll_depths else None

    # Fine print inspection rate
    fine_print_count = sum(1 for r in responses if r.get("fine_print_expand_count", 0) > 0)
    fine_print_inspection_rate = fine_print_count / len(responses) if responses else None

    # Abandonment rate
    abandoned_count = sum(1 for r in responses if r.get("abandoned", False))
    abandonment_rate = abandoned_count / len(responses) if responses else None

    # Update participant with metrics
    await db.participants.update_one(
        {"participant_id": participant_id},
        {
            "$set": {
                "manipulation_rate": manipulation_rate,
                "resistance_rate": resistance_rate,
                "detection_rate": detection_rate,
                "false_alarm_rate": false_alarm_rate,
                "mean_task_time_ms": mean_task_time_ms,
                "mean_scroll_depth": mean_scroll_depth,
                "fine_print_inspection_rate": fine_print_inspection_rate,
                "abandonment_rate": abandonment_rate,
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
    db = get_darkpatterns_db()

    participants = await db.participants.find().to_list(length=10000)

    # Join with pre-survey data
    for p in participants:
        pre_survey = await db.pre_survey_responses.find_one(
            {"participant_id": p["participant_id"]}
        )
        if pre_survey:
            # Merge pre-survey fields
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
    db = get_darkpatterns_db()

    responses = await db.responses.find().to_list(length=100000)

    # Remove MongoDB _id and convert enums to strings
    for r in responses:
        r.pop("_id", None)
        # Convert enums to strings
        for key in ["ui_type", "intensity", "visual_manipulation", "context"]:
            if key in r and hasattr(r[key], "value"):
                r[key] = r[key].value

    return {"responses": responses}


@router.get("/export/participants/csv")
async def export_participants_csv():
    """Export participant data in CSV-ready format."""
    db = get_darkpatterns_db()

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
            "manipulation_rate": p.get("manipulation_rate"),
            "resistance_rate": p.get("resistance_rate"),
            "detection_rate": p.get("detection_rate"),
            "false_alarm_rate": p.get("false_alarm_rate"),
            "mean_task_time_ms": p.get("mean_task_time_ms"),
            "mean_scroll_depth": p.get("mean_scroll_depth"),
            "fine_print_inspection_rate": p.get("fine_print_inspection_rate"),
            "abandonment_rate": p.get("abandonment_rate"),
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
            "digital_literacy": pre_survey.get("digital_literacy"),
            "impulse_buying": pre_survey.get("impulse_buying"),
            "shopping_frequency": pre_survey.get("shopping_frequency"),
            "years_online_shopping": pre_survey.get("years_online_shopping"),
            "dp_awareness_heard_term": pre_survey.get("dp_awareness_heard_term"),
            "manipulation_detection_efficacy": pre_survey.get("manipulation_detection_efficacy"),
            "perceived_online_risk": pre_survey.get("perceived_online_risk"),
        })

        csv_data.append(row)

    return {"data": csv_data}


@router.get("/export/responses/csv")
async def export_responses_csv():
    """Export response data in CSV-ready format."""
    db = get_darkpatterns_db()

    responses = await db.responses.find().to_list(length=100000)

    csv_data = []
    for r in responses:
        row = {
            "participant_id": r.get("participant_id"),
            "task_id": r.get("task_id"),
            "ui_type": r.get("ui_type"),
            "intensity": r.get("intensity"),
            "time_pressure": r.get("time_pressure"),
            "visual_manipulation": r.get("visual_manipulation"),
            "context": r.get("context"),
            "ground_truth": r.get("ground_truth"),
            "final_action": r.get("final_action"),
            "manipulated": r.get("manipulated"),
            "resisted": r.get("resisted"),
            "abandoned": r.get("abandoned"),
            "task_completion_time_ms": r.get("task_completion_time_ms"),
            "time_to_first_action_ms": r.get("time_to_first_action_ms"),
            "scroll_depth": r.get("scroll_depth"),
            "fine_print_expand_count": r.get("fine_print_expand_count"),
            "option_hover_count": r.get("option_hover_count"),
            "correct_option_hover": r.get("correct_option_hover"),
            "backtrack_count": r.get("backtrack_count"),
            "click_path_length": r.get("click_path_length"),
            "steps_completed": r.get("steps_completed"),
            "steps_total": r.get("steps_total"),
            "perceived_difficulty": r.get("perceived_difficulty"),
            "noticed_unusual": r.get("noticed_unusual"),
            "confidence_rating": r.get("confidence_rating"),
        }
        csv_data.append(row)

    return {"data": csv_data}
