"""
Transcription service with smart segmentation and Gemini-based speaker diarization.
"""
from __future__ import annotations

import whisper
from pathlib import Path
from typing import Optional
from core.config import get_settings
from core.logging import get_logger
from core.gemini_client import gemini_generate
from services.audio import preprocess_audio
from models.meeting import Utterance

logger = get_logger(__name__)
settings = get_settings()

_whisper_model = None

# ── Constants ─────────────────────────────────────
MERGE_GAP_THRESHOLD = 1.5   # seconds — merge segments closer than this
PAUSE_BREAK_GAP = 3.0       # seconds — break into separate utterances at this gap
MIN_UTTERANCE_CHARS = 10    # skip tiny fragments below this length
DIARIZATION_BATCH = 40      # max utterances per Gemini diarization call


def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {settings.whisper_model}")
        _whisper_model = whisper.load_model(settings.whisper_model)
        logger.info("Whisper model loaded")
    return _whisper_model


async def transcribe(
    audio_path: Path,
    participants: Optional[list[str]] = None,
    visual_context: Optional[str] = None,
) -> list[Utterance]:
    """
    Full transcription pipeline:
    1. Preprocess audio (normalize, convert to 16kHz WAV)
    2. Transcribe with Whisper
    3. Merge short segments into coherent utterances (no speaker assignment)
    4. Use Gemini to diarize — group utterances by speaker using conversational + visual context
    """
    # 1. Preprocess
    processed_path = await preprocess_audio(audio_path)
    model = load_whisper()
    logger.info(f"Transcribing: {audio_path.name}")

    # 2. Whisper transcription
    result = model.transcribe(
        str(processed_path),
        word_timestamps=True,
        verbose=False,
        language="en",
    )

    raw_segments = result.get("segments", [])
    logger.info(f"Whisper returned {len(raw_segments)} raw segments")

    if not raw_segments:
        logger.warning("No segments returned from Whisper")
        return []

    # 3. Merge into coherent utterances (speaker-agnostic)
    utterances = _merge_segments(raw_segments)
    logger.info(f"After merge: {len(utterances)} utterances")

    # 4. Gemini-based speaker diarization (with optional visual cues)
    utterances = await _diarize_with_gemini(utterances, participants, visual_context)

    # Clean up
    if processed_path != audio_path:
        try:
            processed_path.unlink(missing_ok=True)
        except Exception:
            pass

    return utterances


def _merge_segments(raw_segments: list[dict]) -> list[Utterance]:
    """
    Merge Whisper's micro-segments into coherent utterances based on time gaps.
    Does NOT assign speakers — that's done by Gemini later.
    All utterances get a placeholder "Unknown" speaker.
    """
    if not raw_segments:
        return []

    merged = []
    current_texts = []
    current_start = raw_segments[0]["start"]
    current_end = raw_segments[0]["end"]

    for i, seg in enumerate(raw_segments):
        text = seg["text"].strip()
        if not text:
            continue

        if i == 0:
            current_texts.append(text)
            current_end = seg["end"]
            continue

        gap = seg["start"] - current_end

        if gap > PAUSE_BREAK_GAP:
            # Big pause — flush current utterance and start new one
            _flush(merged, current_texts, current_start, current_end)
            current_texts = [text]
            current_start = seg["start"]
            current_end = seg["end"]
        elif gap <= MERGE_GAP_THRESHOLD:
            # Small gap — merge into current utterance
            current_texts.append(text)
            current_end = seg["end"]
        else:
            # Medium gap — flush but start fresh
            _flush(merged, current_texts, current_start, current_end)
            current_texts = [text]
            current_start = seg["start"]
            current_end = seg["end"]

    _flush(merged, current_texts, current_start, current_end)
    return merged


def _flush(merged: list[Utterance], texts: list[str], start: float, end: float):
    """Flush accumulated text into an Utterance."""
    if not texts:
        return
    combined = " ".join(texts).strip()
    if len(combined) < MIN_UTTERANCE_CHARS:
        return
    merged.append(Utterance(
        speaker="Unknown",
        text=combined,
        start_time=round(start, 2),
        end_time=round(end, 2),
    ))


async def _diarize_with_gemini(
    utterances: list[Utterance],
    participants: Optional[list[str]] = None,
    visual_context: Optional[str] = None,
) -> list[Utterance]:
    """
    Use Gemini to assign speakers to utterances by analyzing conversational context
    and optional visual cues from video frames.
    """
    if not utterances:
        return utterances

    # Build participant hint
    participant_hint = ""
    if participants and any(p.strip() for p in participants):
        names = [p.strip() for p in participants if p.strip()]
        participant_hint = f"\n\nKnown participants ({len(names)} people): {', '.join(names)}."

    # Build visual context hint
    visual_hint = ""
    if visual_context:
        visual_hint = f"\n\nVISUAL CUES FROM VIDEO (use these to help identify speakers):\n{visual_context}"
    
    # Process in batches with overlap for context continuity
    all_results = []

    for batch_start in range(0, len(utterances), DIARIZATION_BATCH):
        batch_end = min(batch_start + DIARIZATION_BATCH, len(utterances))
        batch = utterances[batch_start:batch_end]

        context_size = min(5, batch_start)
        context_before = utterances[batch_start - context_size:batch_start] if context_size > 0 else []

        prev_speakers = {}
        if all_results:
            for u in all_results[-10:]:
                if u.speaker != "Unknown":
                    prev_speakers[u.speaker] = u.text[:50]

        diarized = await _diarize_batch(
            batch, batch_start, context_before,
            participant_hint + visual_hint, prev_speakers
        )
        all_results.extend(diarized)

    unique = set(u.speaker for u in all_results)
    logger.info(f"Diarization complete: {len(unique)} unique speakers across {len(all_results)} utterances")

    return all_results


async def _diarize_batch(
    batch: list[Utterance],
    batch_offset: int,
    context_before: list[Utterance],
    participant_hint: str,
    prev_speakers: dict,
) -> list[Utterance]:
    """Diarize a single batch of utterances using Gemini."""

    # Build the transcript for analysis
    lines = []
    for i, u in enumerate(batch):
        timestamp = f"[{u.start_time:.0f}s–{u.end_time:.0f}s]"
        lines.append(f"#{batch_offset + i}: {timestamp} \"{u.text}\"")

    transcript = "\n".join(lines)

    # Show context from previous batch if available
    context_str = ""
    if context_before:
        ctx_lines = [f"  [{u.speaker}]: \"{u.text[:80]}\"" for u in context_before]
        context_str = f"\n\nPrevious context (already assigned speakers):\n" + "\n".join(ctx_lines)

    prev_speaker_str = ""
    if prev_speakers:
        prev_speaker_str = f"\n\nSpeakers identified so far: {', '.join(prev_speakers.keys())}. Reuse these same names if recognizing the same person."

    prompt = f"""You are an expert at speaker diarization — identifying WHO is saying WHAT in a multi-person conversation.

Below is a transcript of a meeting/conversation where each line is a separate utterance. Your job is to assign a speaker to each utterance.

RULES:
1. Identify speakers by their REAL NAMES when possible (from introductions, being addressed by name, or known participants).
2. When you can't determine a real name, use consistent labels like "Speaker 1", "Speaker 2", etc.
3. The SAME person talking in consecutive utterances should get the SAME speaker label.
4. Look for cues:
   - "Hi, I'm Sarah" → this utterance is Sarah
   - "Thanks, John" → previous or next utterance might be John
   - Same topic continuation → likely same speaker
   - "I agree with what Sarah said" → NOT Sarah, someone else
   - Questions followed by answers → likely different speakers
5. Be CONSERVATIVE with speaker count — a typical meeting has 3-10 people, not 20+.
6. If two utterances close in time sound like the same train of thought, they're likely the same speaker.
{participant_hint}{context_str}{prev_speaker_str}

Transcript to diarize:
{transcript}

Return a JSON array with one object per utterance (in order), each with:
- "index": the # number from the transcript
- "speaker": the assigned speaker name

Return ONLY valid JSON. No markdown. No explanation."""

    try:
        result = await gemini_generate(prompt, max_retries=3, parse_json=True)

        if not isinstance(result, list):
            logger.warning(f"Diarization returned non-list: {type(result)}")
            return batch

        # Map results back to utterances
        speaker_map = {}
        for item in result:
            idx = item.get("index", -1)
            speaker = item.get("speaker", "Unknown")
            if isinstance(idx, int):
                speaker_map[idx] = speaker

        for i, u in enumerate(batch):
            global_idx = batch_offset + i
            if global_idx in speaker_map:
                u.speaker = speaker_map[global_idx]

        return batch

    except Exception as e:
        logger.error(f"Diarization batch failed: {e}")
        return batch
