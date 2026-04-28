from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from models.meeting import UtteranceType, Priority, ExtractedTask
from core.logging import get_logger
from services.notion import push_tasks as notion_push
from services.linear import push_tasks as linear_push

router = APIRouter(prefix="/tasks", tags=["tasks"])
logger = get_logger(__name__)

# Simple in-memory feedback store — persist to DB in production
_feedback: list[dict] = []


class FeedbackPayload(BaseModel):
    meeting_id: str
    utterance_text: str
    original_type: UtteranceType
    corrected_type: UtteranceType
    corrected_owner: Optional[str] = None
    corrected_deadline: Optional[str] = None


class TaskCorrection(BaseModel):
    meeting_id: str
    task_title: str
    corrected_owner: Optional[str] = None
    corrected_deadline: Optional[str] = None
    corrected_priority: Optional[Priority] = None
    is_valid_task: bool = True          # set False to remove spurious tasks


@router.post("/feedback")
async def submit_feedback(payload: FeedbackPayload):
    """
    Correction endpoint. Stores classifier errors.
    Use this data to evaluate and improve the classifier over time.
    """
    record = payload.model_dump()
    _feedback.append(record)
    logger.info(
        f"Feedback: '{payload.utterance_text[:60]}' "
        f"{payload.original_type} → {payload.corrected_type}"
    )
    return {"status": "recorded", "total_feedback": len(_feedback)}


@router.post("/correct")
async def correct_task(payload: TaskCorrection):
    """Correct extracted task fields."""
    logger.info(f"Task correction for: {payload.task_title}")
    return {"status": "recorded"}


@router.get("/feedback/export")
async def export_feedback():
    """Export all feedback for classifier evaluation."""
    return {
        "total": len(_feedback),
        "records": _feedback,
    }


@router.get("/feedback/stats")
async def feedback_stats():
    """Quick accuracy stats from feedback."""
    if not _feedback:
        return {"message": "No feedback yet"}

    total = len(_feedback)
    correct = sum(1 for f in _feedback if f["original_type"] == f["corrected_type"])
    wrong_by_type: dict[str, int] = {}

    for f in _feedback:
        if f["original_type"] != f["corrected_type"]:
            key = f"{f['original_type']} → {f['corrected_type']}"
            wrong_by_type[key] = wrong_by_type.get(key, 0) + 1

    return {
        "total_feedback": total,
        "classifier_accuracy": round(correct / total, 3),
        "most_common_errors": sorted(wrong_by_type.items(), key=lambda x: -x[1])[:5],
    }


@router.post("/notion")
async def push_to_notion(task: ExtractedTask):
    """Push a single extracted task to Notion."""
    logger.info(f"On-demand push to Notion for task: {task.title}")
    updated = await notion_push([task])
    return {"status": "success", "task": updated[0]}


@router.post("/linear")
async def push_to_linear(task: ExtractedTask):
    """Push a single extracted task to Linear."""
    logger.info(f"On-demand push to Linear for task: {task.title}")
    updated = await linear_push([task])
    return {"status": "success", "task": updated[0]}
