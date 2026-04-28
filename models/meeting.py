from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class UtteranceType(str, Enum):
    DECISION = "decision"
    COMMITMENT = "commitment"
    DISCUSSION = "discussion"
    OPEN_QUESTION = "open_question"


class ContentType(str, Enum):
    MEETING = "meeting"
    LECTURE = "lecture"
    PODCAST = "podcast"
    INTERVIEW = "interview"
    OTHER = "other"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Transcript models ────────────────────────────

class Utterance(BaseModel):
    speaker: str
    text: str
    start_time: float
    end_time: float
    utterance_type: Optional[UtteranceType] = None
    confidence: float = 1.0


class ClassifiedUtterance(Utterance):
    utterance_type: UtteranceType
    classification_confidence: float
    reasoning: str


# ── Task models ──────────────────────────────────

class ExtractedTask(BaseModel):
    title: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    context_quote: str
    source_speaker: Optional[str] = None
    notion_page_id: Optional[str] = None
    linear_issue_id: Optional[str] = None


# ── Lecture-specific models ──────────────────────

class KeyTakeaway(BaseModel):
    title: str
    explanation: str
    importance: str = "medium"  # high, medium, low

class StudyNote(BaseModel):
    topic: str
    content: str
    key_terms: list[str] = []


# ── Video analysis models ────────────────────────

class SlideContent(BaseModel):
    timestamp: float
    title: str = ""
    content: str = ""
    description: str = ""


# ── Meeting models ───────────────────────────────

class MeetingStatus(str, Enum):
    UPLOADING = "uploading"
    TRANSCRIBING = "transcribing"
    DETECTING = "detecting"
    ANALYZING_VIDEO = "analyzing_video"
    CLASSIFYING = "classifying"
    EXTRACTING = "extracting"
    PUSHING = "pushing"
    DONE = "done"
    FAILED = "failed"


class MeetingCreate(BaseModel):
    title: str
    participants: list[str] = Field(default_factory=list)
    push_notion: bool = False
    push_slack: bool = False
    push_linear: bool = False
    source_url: Optional[str] = None


class MeetingResult(BaseModel):
    meeting_id: str
    title: str
    status: MeetingStatus
    content_type: ContentType = ContentType.MEETING
    duration_seconds: Optional[float] = None
    # Meeting-specific
    decisions: list[ClassifiedUtterance] = []
    commitments: list[ClassifiedUtterance] = []
    discussions: list[ClassifiedUtterance] = []
    open_questions: list[ClassifiedUtterance] = []
    tasks: list[ExtractedTask] = []
    # Lecture-specific
    key_takeaways: list[KeyTakeaway] = []
    study_notes: list[StudyNote] = []
    # Video analysis
    slide_contents: list[SlideContent] = []
    scene_description: str = ""
    # Shared
    utterances: list[ClassifiedUtterance] = []
    summary: Optional[str] = None
    source_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MeetingStatusResponse(BaseModel):
    meeting_id: str
    status: MeetingStatus
    progress_message: str
