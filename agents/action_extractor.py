"""
Action extractor — pulls tasks from classified utterances and generates summaries.
Adapts to content type: meetings get tasks, lectures get takeaways and study notes.
"""
from __future__ import annotations

import json
from core.logging import get_logger
from core.gemini_client import gemini_generate
from models.meeting import (
    ClassifiedUtterance, ExtractedTask, Priority, UtteranceType,
    KeyTakeaway, StudyNote,
)

logger = get_logger(__name__)

EXTRACTION_PROMPT = """You are extracting actionable tasks from meeting commitments and open questions.

For each input utterance, extract a task object with:
- "title": concise action title (verb + object, max 10 words)
- "owner": person responsible (null if unclear)
- "deadline": when it's due in natural language (null if not mentioned)
- "priority": "high", "medium", or "low" — infer from context (urgency words, blocking nature)
- "context_quote": the exact original utterance text
- "source_speaker": who said it

Priority heuristics:
- HIGH: blocking other work, client-facing, explicit urgency ("ASAP", "urgent", "before launch")
- LOW: vague timeline ("sometime", "eventually", "when you get a chance")
- MEDIUM: everything else

Return ONLY a valid JSON array. No markdown. No explanation."""


async def extract_tasks(
    classified: list[ClassifiedUtterance],
) -> list[ExtractedTask]:
    """Extract tasks from commitments and open questions only."""
    actionable = [
        c for c in classified
        if c.utterance_type in (UtteranceType.COMMITMENT, UtteranceType.OPEN_QUESTION)
    ]

    if not actionable:
        logger.info("No actionable utterances found")
        return []

    input_json = json.dumps([
        {"speaker": u.speaker, "text": u.text, "type": u.utterance_type.value}
        for u in actionable
    ], indent=2)

    prompt = f"{EXTRACTION_PROMPT}\n\nInput:\n{input_json}"

    try:
        tasks_data = await gemini_generate(prompt, max_retries=3, parse_json=True)
        if not isinstance(tasks_data, list):
            return []

        tasks = []
        for t in tasks_data:
            try:
                tasks.append(ExtractedTask(
                    title=t.get("title", "Untitled task"),
                    owner=t.get("owner"),
                    deadline=t.get("deadline"),
                    priority=Priority(t.get("priority", "medium")),
                    context_quote=t.get("context_quote", ""),
                    source_speaker=t.get("source_speaker"),
                ))
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping malformed task: {e}")

        logger.info(f"Extracted {len(tasks)} tasks from {len(actionable)} actionable utterances")
        return tasks
    except Exception as e:
        logger.error(f"Task extraction error: {e}")
        return []


# ── Lecture-specific extraction ──────────────────

async def extract_takeaways(
    classified: list[ClassifiedUtterance],
    title: str,
) -> list[KeyTakeaway]:
    """Extract key takeaways from lecture/podcast content."""
    transcript = "\n".join(f"[{c.speaker}]: {c.text}" for c in classified[:50])

    prompt = f"""You are analyzing a lecture/educational content titled "{title}".
Extract the 5-8 most important takeaways from this transcript.

Transcript:
{transcript}

Return a JSON array where each object has:
- "title": concise takeaway (max 10 words)
- "explanation": 1-2 sentence explanation of the concept
- "importance": "high", "medium", or "low"

Return ONLY valid JSON."""

    try:
        data = await gemini_generate(prompt, max_retries=3, parse_json=True)
        if not isinstance(data, list):
            return []

        return [
            KeyTakeaway(
                title=t.get("title", ""),
                explanation=t.get("explanation", ""),
                importance=t.get("importance", "medium"),
            )
            for t in data if t.get("title")
        ]
    except Exception as e:
        logger.error(f"Takeaway extraction failed: {e}")
        return []


async def extract_study_notes(
    classified: list[ClassifiedUtterance],
    title: str,
) -> list[StudyNote]:
    """Generate structured study notes from lecture content."""
    transcript = "\n".join(f"[{c.speaker}]: {c.text}" for c in classified[:50])

    prompt = f"""You are creating study notes from a lecture titled "{title}".
Organize the content into 3-6 topic sections.

Transcript:
{transcript}

Return a JSON array where each object has:
- "topic": section heading
- "content": 2-4 sentence summary of that topic
- "key_terms": array of important terms/concepts mentioned

Return ONLY valid JSON."""

    try:
        data = await gemini_generate(prompt, max_retries=3, parse_json=True)
        if not isinstance(data, list):
            return []

        return [
            StudyNote(
                topic=n.get("topic", ""),
                content=n.get("content", ""),
                key_terms=n.get("key_terms", []),
            )
            for n in data if n.get("topic")
        ]
    except Exception as e:
        logger.error(f"Study notes extraction failed: {e}")
        return []


# ── Summary generation ───────────────────────────

async def generate_summary(
    classified: list[ClassifiedUtterance],
    title: str,
    content_type: str = "meeting",
) -> str:
    """Generate a summary adapted to content type."""
    if content_type == "meeting":
        return await _meeting_summary(classified, title)
    else:
        return await _general_summary(classified, title, content_type)


async def _meeting_summary(classified: list[ClassifiedUtterance], title: str) -> str:
    decisions = [c for c in classified if c.utterance_type == UtteranceType.DECISION]
    commitments = [c for c in classified if c.utterance_type == UtteranceType.COMMITMENT]
    open_qs = [c for c in classified if c.utterance_type == UtteranceType.OPEN_QUESTION]
    discussions = [c for c in classified if c.utterance_type == UtteranceType.DISCUSSION]

    context = f"""Meeting: {title}
Total utterances: {len(classified)}
Decisions made ({len(decisions)}): {[f'{d.speaker}: {d.text}' for d in decisions[:5]]}
Commitments ({len(commitments)}): {[f'{c.speaker}: {c.text}' for c in commitments[:5]]}
Open questions ({len(open_qs)}): {[f'{q.speaker}: {q.text}' for q in open_qs[:5]]}
Key discussion points: {[d.text for d in discussions[:3]]}"""

    prompt = f"""Write a 3-5 sentence executive summary of this meeting.
Be direct and specific. Mention key decisions and who owns what.
Include speaker names when attributing actions or decisions.
Do not use bullet points.

{context}"""

    try:
        return await gemini_generate(prompt, max_retries=3, parse_json=False)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"Meeting '{title}' contained {len(classified)} utterances."


async def _general_summary(
    classified: list[ClassifiedUtterance],
    title: str,
    content_type: str,
) -> str:
    transcript = "\n".join(f"[{c.speaker}]: {c.text}" for c in classified[:30])

    prompt = f"""Write a 3-5 sentence summary of this {content_type} titled "{title}".
Capture the key themes and main points discussed.
Be concise and informative. Do not use bullet points.

Transcript excerpt:
{transcript}"""

    try:
        return await gemini_generate(prompt, max_retries=3, parse_json=False)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"This {content_type} titled '{title}' contained {len(classified)} segments."
