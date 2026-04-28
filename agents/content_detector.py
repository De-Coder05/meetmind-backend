"""
Content type detector — classifies audio content as meeting, lecture, podcast, interview, or other.
"""
from __future__ import annotations

from core.logging import get_logger
from core.gemini_client import gemini_generate
from models.meeting import Utterance

logger = get_logger(__name__)


async def detect_content_type(utterances: list[Utterance]) -> dict:
    """
    Analyze the first ~2 minutes of transcript to classify content type.
    Returns {"content_type": "meeting"|"lecture"|"podcast"|"interview"|"other", "confidence": float}
    """
    if not utterances:
        return {"content_type": "meeting", "confidence": 0.5}

    # Take first 2 minutes of transcript
    intro = [u for u in utterances if u.start_time <= 120.0]
    if not intro:
        intro = utterances[:10]

    transcript = "\n".join(f"[{u.speaker}]: {u.text}" for u in intro)

    prompt = f"""Classify this audio transcript into one of these content types:

- **meeting**: Multiple participants discussing, making decisions, assigning tasks. Key signals: back-and-forth dialogue, "let's discuss", "action items", multiple distinct speakers.
- **lecture**: A single speaker teaching/presenting educational content. Key signals: one dominant speaker, explaining concepts, academic tone, slides references.
- **podcast**: Casual conversation or interview-style with hosts/guests. Key signals: informal tone, "welcome to the show", entertainment focus, audience references.
- **interview**: Structured Q&A between interviewer and interviewee. Key signals: alternating questions and answers, formal introductions, "tell me about".
- **other**: Anything that doesn't fit the above categories.

Transcript (first ~2 minutes):
{transcript}

Return a JSON object:
{{"content_type": "meeting"|"lecture"|"podcast"|"interview"|"other", "confidence": 0.0-1.0, "reasoning": "one sentence"}}

Return ONLY valid JSON."""

    try:
        result = await gemini_generate(prompt, max_retries=2, parse_json=True)

        content_type = result.get("content_type", "meeting")
        if content_type not in ("meeting", "lecture", "podcast", "interview", "other"):
            content_type = "meeting"

        logger.info(f"Content type detected: {content_type} (confidence: {result.get('confidence', 0):.2f})")
        return {
            "content_type": content_type,
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", ""),
        }

    except Exception as e:
        logger.error(f"Content detection failed: {e}, defaulting to 'meeting'")
        return {"content_type": "meeting", "confidence": 0.5}
