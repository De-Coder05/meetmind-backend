import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from core.config import get_settings
from core.logging import get_logger
from db.database import get_db, AsyncSessionLocal
from db.schemas import Meeting
from models.meeting import (
    MeetingCreate, MeetingResult, MeetingStatus, ContentType,
    MeetingStatusResponse, UtteranceType
)
from services.transcription import transcribe
from services.downloader import download_audio
from agents.classifier import classify_utterances
from agents.content_detector import detect_content_type
from agents.action_extractor import (
    extract_tasks, generate_summary,
    extract_takeaways, extract_study_notes,
)
from services.notion import push_tasks, push_meeting_summary
from services.slack import send_digest
from services.linear import push_tasks as push_tasks_linear

router = APIRouter(prefix="/meetings", tags=["meetings"])
logger = get_logger(__name__)
settings = get_settings()


# ── Request model for URL uploads ────────────────

class UrlUploadRequest(BaseModel):
    url: str
    title: str = "Untitled"
    participants: str = ""
    push_notion: bool = False
    push_slack: bool = False
    push_linear: bool = False


# ── Endpoints ────────────────────────────────────

@router.post("/upload", response_model=MeetingStatusResponse)
async def upload_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    participants: str = Form(default=""),
    push_notion: bool = Form(default=False),
    push_slack: bool = Form(default=False),
    push_linear: bool = Form(default=False),
    db: AsyncSession = Depends(get_db)
):
    """Upload an audio file and kick off the full MeetMind pipeline."""
    meeting_id = str(uuid.uuid4())

    allowed = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".webm"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(exist_ok=True)
    audio_path = upload_dir / f"{meeting_id}{suffix}"

    async with aiofiles.open(audio_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    config = MeetingCreate(
        title=title,
        participants=[p.strip() for p in participants.split(",") if p.strip()],
        push_notion=push_notion,
        push_slack=push_slack,
        push_linear=push_linear,
    )

    new_meeting = Meeting(id=meeting_id, title=title, status=MeetingStatus.TRANSCRIBING)
    db.add(new_meeting)
    await db.commit()
    # If uploaded file is a video, also use it for video analysis
    video_formats = {".mp4", ".webm"}
    video_path = audio_path if suffix in video_formats else None

    background_tasks.add_task(_run_pipeline, meeting_id, audio_path, config, video_path=video_path)

    return MeetingStatusResponse(
        meeting_id=meeting_id,
        status=MeetingStatus.TRANSCRIBING,
        progress_message="Pipeline started. Audio uploaded and transcription beginning.",
    )


@router.post("/upload-url", response_model=MeetingStatusResponse)
async def upload_from_url(
    background_tasks: BackgroundTasks,
    body: UrlUploadRequest,
    db: AsyncSession = Depends(get_db),
):
    """Download audio from a URL (YouTube, etc.) and process it."""
    meeting_id = str(uuid.uuid4())

    config = MeetingCreate(
        title=body.title,
        participants=[p.strip() for p in body.participants.split(",") if p.strip()],
        push_notion=body.push_notion,
        push_slack=body.push_slack,
        push_linear=body.push_linear,
        source_url=body.url,
    )

    new_meeting = Meeting(id=meeting_id, title=body.title, status=MeetingStatus.UPLOADING)
    db.add(new_meeting)
    await db.commit()

    background_tasks.add_task(_run_url_pipeline, meeting_id, body.url, config)

    return MeetingStatusResponse(
        meeting_id=meeting_id,
        status=MeetingStatus.UPLOADING,
        progress_message="Downloading audio from URL...",
    )


@router.get("/{meeting_id}/status", response_model=MeetingStatusResponse)
async def get_status(meeting_id: str, db: AsyncSession = Depends(get_db)):
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(404, "Meeting not found")

    messages = {
        MeetingStatus.UPLOADING: "Downloading audio from URL...",
        MeetingStatus.ANALYZING_VIDEO: "Analyzing video frames...",
        MeetingStatus.TRANSCRIBING: "Transcribing audio...",
        MeetingStatus.DETECTING: "Detecting content type...",
        MeetingStatus.CLASSIFYING: "Classifying utterances...",
        MeetingStatus.EXTRACTING: "Extracting insights...",
        MeetingStatus.PUSHING: "Pushing to integrations...",
        MeetingStatus.DONE: "Pipeline complete.",
        MeetingStatus.FAILED: "Pipeline failed. Check logs.",
    }

    return MeetingStatusResponse(
        meeting_id=meeting_id,
        status=MeetingStatus(meeting.status),
        progress_message=messages.get(MeetingStatus(meeting.status), "Processing..."),
    )


@router.get("/{meeting_id}", response_model=MeetingResult)
async def get_result(meeting_id: str, db: AsyncSession = Depends(get_db)):
    meeting = await db.get(Meeting, meeting_id)
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    if meeting.status not in (MeetingStatus.DONE, MeetingStatus.FAILED):
        raise HTTPException(202, "Meeting still processing")

    if meeting.result_json:
        return MeetingResult(**meeting.result_json)

    return MeetingResult(
        meeting_id=meeting_id,
        title=meeting.title,
        status=MeetingStatus(meeting.status)
    )


@router.get("/", response_model=list[MeetingResult])
async def list_meetings(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Meeting).order_by(Meeting.created_at.desc()))
    meetings = result.scalars().all()

    out = []
    for m in meetings:
        if m.result_json:
            out.append(MeetingResult(**m.result_json))
        else:
            out.append(MeetingResult(
                meeting_id=m.id,
                title=m.title,
                status=MeetingStatus(m.status)
            ))
    return out


# ── URL Pipeline ─────────────────────────────────

async def _run_url_pipeline(meeting_id: str, url: str, config: MeetingCreate):
    """Download audio+video from URL, then run the pipeline."""
    async with AsyncSessionLocal() as db:
        meeting = await db.get(Meeting, meeting_id)
        if not meeting:
            return

        try:
            meeting.status = MeetingStatus.UPLOADING
            await db.commit()

            dl_result = await download_audio(url)
            logger.info(
                f"[{meeting_id}] Downloaded: audio={dl_result.audio_path.name}, "
                f"video={'yes' if dl_result.video_path else 'no'}"
            )

            await _run_pipeline(
                meeting_id, dl_result.audio_path, config,
                video_path=dl_result.video_path,
            )

        except Exception as e:
            logger.error(f"[{meeting_id}] URL download failed: {e}", exc_info=True)
            meeting.status = MeetingStatus.FAILED
            result = MeetingResult(
                meeting_id=meeting_id,
                title=config.title,
                status=MeetingStatus.FAILED,
                source_url=url,
            )
            meeting.result_json = result.model_dump(mode='json')
            await db.commit()


# ── Main Pipeline ────────────────────────────────

async def _run_pipeline(
    meeting_id: str,
    audio_path: Path,
    config: MeetingCreate,
    video_path: Optional[Path] = None,
):
    async with AsyncSessionLocal() as db:
        meeting = await db.get(Meeting, meeting_id)
        if not meeting:
            return

        result = MeetingResult(
            meeting_id=meeting_id,
            title=config.title,
            status=MeetingStatus.TRANSCRIBING,
            source_url=config.source_url,
        )

        frames_dir = None
        visual_context_str = None

        try:
            # ── Stage 1: Video analysis (if video available) ──
            if video_path and video_path.exists():
                meeting.status = MeetingStatus.ANALYZING_VIDEO
                await db.commit()

                try:
                    from services.frame_extractor import extract_frames, cleanup_frames
                    from agents.video_analyzer import analyze_video_frames, format_visual_context_for_diarization

                    frames = await extract_frames(video_path)
                    frames_dir = video_path.parent / f"{video_path.stem}_frames"

                    if frames:
                        visual_data = await analyze_video_frames(frames)

                        visual_context_str = format_visual_context_for_diarization(visual_data)

                        # Store slide content and scene
                        from models.meeting import SlideContent
                        result.slide_contents = [
                            SlideContent(
                                timestamp=s.get("timestamp", 0),
                                title=s.get("title", ""),
                                content=s.get("content", ""),
                            )
                            for s in visual_data.get("slides", [])
                        ]
                        result.scene_description = visual_data.get("scene", "")

                        logger.info(
                            f"[{meeting_id}] Video analysis: {len(result.slide_contents)} slides, "
                            f"scene='{result.scene_description[:50]}'"
                        )

                except Exception as e:
                    logger.warning(f"[{meeting_id}] Video analysis failed (non-fatal): {e}")

                meeting.result_json = result.model_dump(mode='json')
                await db.commit()

            # ── Stage 2: Transcribe ──────────────────
            meeting.status = MeetingStatus.TRANSCRIBING
            await db.commit()

            utterances = await transcribe(
                audio_path,
                participants=config.participants,
                visual_context=visual_context_str,
            )
            result.duration_seconds = max((u.end_time for u in utterances), default=0)
            logger.info(f"[{meeting_id}] Transcription: {len(utterances)} utterances, {result.duration_seconds:.0f}s")

            meeting.result_json = result.model_dump(mode='json')
            await db.commit()

            # ── Stage 3: Detect content type ─────────
            meeting.status = MeetingStatus.DETECTING
            await db.commit()

            detection = await detect_content_type(utterances)
            content_type = detection["content_type"]
            result.content_type = ContentType(content_type)
            logger.info(f"[{meeting_id}] Content type: {content_type} ({detection.get('confidence', 0):.0%})")

            meeting.result_json = result.model_dump(mode='json')
            await db.commit()

            # ── Stage 4: Classify ────────────────────
            meeting.status = MeetingStatus.CLASSIFYING
            await db.commit()

            classified = await classify_utterances(utterances, content_type=content_type)

            result.decisions = [c for c in classified if c.utterance_type == UtteranceType.DECISION]
            result.commitments = [c for c in classified if c.utterance_type == UtteranceType.COMMITMENT]
            result.discussions = [c for c in classified if c.utterance_type == UtteranceType.DISCUSSION]
            result.open_questions = [c for c in classified if c.utterance_type == UtteranceType.OPEN_QUESTION]
            result.utterances = classified

            logger.info(
                f"[{meeting_id}] Classification: "
                f"{len(result.decisions)} decisions, {len(result.commitments)} commitments, "
                f"{len(result.discussions)} discussions, {len(result.open_questions)} open questions"
            )

            meeting.result_json = result.model_dump(mode='json')
            await db.commit()

            # ── Stage 5: Extract insights ────────────
            meeting.status = MeetingStatus.EXTRACTING
            await db.commit()

            if content_type == "meeting":
                result.tasks = await extract_tasks(classified)
                logger.info(f"[{meeting_id}] Extracted {len(result.tasks)} tasks")
            else:
                result.key_takeaways = await extract_takeaways(classified, config.title)
                result.study_notes = await extract_study_notes(classified, config.title)
                logger.info(
                    f"[{meeting_id}] Extracted {len(result.key_takeaways)} takeaways, "
                    f"{len(result.study_notes)} study notes"
                )

            result.summary = await generate_summary(classified, config.title, content_type)

            meeting.result_json = result.model_dump(mode='json')
            await db.commit()

            # ── Stage 6: Push to integrations ────────
            meeting.status = MeetingStatus.PUSHING
            await db.commit()

            if config.push_notion and getattr(settings, "notion_api_key", None):
                try:
                    if content_type == "meeting":
                        result.tasks = await push_tasks(result.tasks, getattr(settings, "notion_database_id", None))
                    await push_meeting_summary(result, getattr(settings, "notion_database_id", None))
                    logger.info(f"[{meeting_id}] Notion push succeeded")
                except Exception as e:
                    logger.error(f"[{meeting_id}] Notion push failed (non-fatal): {e}")

            if config.push_slack and getattr(settings, "slack_bot_token", None):
                try:
                    await send_digest(result, getattr(settings, "slack_default_channel", None))
                    logger.info(f"[{meeting_id}] Slack push succeeded")
                except Exception as e:
                    logger.error(f"[{meeting_id}] Slack push failed (non-fatal): {e}")

            if config.push_linear and getattr(settings, "linear_api_key", None) and content_type == "meeting":
                try:
                    result.tasks = await push_tasks_linear(result.tasks)
                    logger.info(f"[{meeting_id}] Linear push succeeded")
                except Exception as e:
                    logger.error(f"[{meeting_id}] Linear push failed (non-fatal): {e}")

            # ── Done ─────────────────────────────────
            meeting.status = MeetingStatus.DONE
            result.status = MeetingStatus.DONE
            meeting.result_json = result.model_dump(mode='json')
            await db.commit()

            logger.info(f"[{meeting_id}] Pipeline complete ✓ ({content_type})")

        except Exception as e:
            logger.error(f"[{meeting_id}] Pipeline failed: {e}", exc_info=True)
            meeting.status = MeetingStatus.FAILED
            result.status = MeetingStatus.FAILED
            meeting.result_json = result.model_dump(mode='json')
            await db.commit()

        finally:
            # Clean up all temporary files
            for path in [audio_path, video_path]:
                try:
                    if path and path.exists():
                        path.unlink(missing_ok=True)
                except Exception:
                    pass
            if frames_dir:
                try:
                    import shutil
                    shutil.rmtree(frames_dir, ignore_errors=True)
                except Exception:
                    pass

