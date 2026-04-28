"""
Context-aware utterance classifier using Gemini.
Classifies utterances into: decision, commitment, discussion, open_question.
Uses sliding context windows and second-pass verification for robustness.
"""
from __future__ import annotations

import json
from typing import Optional
from core.logging import get_logger
from core.gemini_client import gemini_generate
from models.meeting import Utterance, ClassifiedUtterance, UtteranceType

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert meeting analyst. Your job is to classify each meeting utterance into exactly one of four categories:

DECISION — Something that was definitively agreed upon. Characteristics:
- Past or present tense agreement ("we decided", "we're going with X", "agreed", "let's do X")
- No hedging language
- Group consensus implied, or a final call being made
- Example: "Okay we're going with the React frontend, that's decided."
- Example: "Let's schedule the release for next Monday."

COMMITMENT — A specific promise to do something, made by a named person with an implicit or explicit deadline. Characteristics:
- Named owner ("I will", "Ravi will", "the design team will")
- Specific action, not vague intention
- Example: "I'll send the updated proposal to the client by Thursday."
- NOT a commitment: "We should probably look into pricing sometime." (this is discussion)

DISCUSSION — Something explored or debated but NOT resolved. Characteristics:
- Hedged language ("maybe", "we could", "one option is", "I think we should consider")
- Multiple perspectives offered
- No conclusion reached
- General statements, greetings, status updates
- Example: "We could either go with AWS or GCP, both have tradeoffs."
- Example: "Good morning everyone, let's get started."

OPEN_QUESTION — An unanswered question that needs follow-up. Characteristics:
- Explicit question with no answer in the same utterance
- Or a problem raised with no solution identified
- Example: "But who handles the QA for the new features?"
- Example: "Does anyone know the current status of the API integration?"

IMPORTANT RULES:
1. Use the surrounding context to understand multi-turn exchanges. An utterance after a question might be a decision ("Yes, let's do that.").
2. If someone says "I'll do X" or "I can handle that", classify as COMMITMENT even if informal.
3. Greetings, pleasantries, and pure status updates are DISCUSSION.
4. When in doubt between decision and discussion, check if there's finality language.

Return a JSON array. Each object must have:
- "text": the original utterance text (unchanged)
- "speaker": the speaker name (unchanged)
- "utterance_type": one of "decision", "commitment", "discussion", "open_question"
- "confidence": float 0.0-1.0
- "reasoning": one sentence explaining why you chose this category

Only return valid JSON. No markdown. No explanation outside the JSON."""

FEW_SHOT_EXAMPLES = """
Input utterances:
[
  {"speaker": "Priya", "text": "Okay so we are definitely launching on April 15th, that's final."},
  {"speaker": "Rohan", "text": "I will set up the staging environment by end of day tomorrow."},
  {"speaker": "Priya", "text": "We could use Redis or Memcached for caching, not sure which."},
  {"speaker": "Ankit", "text": "But who is actually responsible for writing the test cases?"},
  {"speaker": "Priya", "text": "Let's go with Ankit on that. Ankit, you own the test suite."},
  {"speaker": "Ankit", "text": "Sure, I'll have the test suite ready by Wednesday."}
]

Output:
[
  {"text": "Okay so we are definitely launching on April 15th, that's final.", "speaker": "Priya", "utterance_type": "decision", "confidence": 0.97, "reasoning": "Definitive agreement with explicit finality marker and a specific date."},
  {"text": "I will set up the staging environment by end of day tomorrow.", "speaker": "Rohan", "utterance_type": "commitment", "confidence": 0.96, "reasoning": "Named owner (I = Rohan), specific action, explicit deadline (end of day tomorrow)."},
  {"text": "We could use Redis or Memcached for caching, not sure which.", "speaker": "Priya", "utterance_type": "discussion", "confidence": 0.91, "reasoning": "Two options presented with hedged language, no conclusion reached."},
  {"text": "But who is actually responsible for writing the test cases?", "speaker": "Ankit", "utterance_type": "open_question", "confidence": 0.95, "reasoning": "Direct unanswered question about ownership with no resolution in the utterance."},
  {"text": "Let's go with Ankit on that. Ankit, you own the test suite.", "speaker": "Priya", "utterance_type": "decision", "confidence": 0.93, "reasoning": "Assignment of ownership with finality — resolves the previous open question."},
  {"text": "Sure, I'll have the test suite ready by Wednesday.", "speaker": "Ankit", "utterance_type": "commitment", "confidence": 0.95, "reasoning": "Explicit acceptance of task with specific deadline (Wednesday)."}
]
"""

CONTEXT_WINDOW = 3  # number of surrounding utterances to include as context


async def classify_utterances(
    utterances: list[Utterance],
    batch_size: int = 15,
    content_type: str = "meeting",
) -> list[ClassifiedUtterance]:
    """
    Classify utterances into 4 buckets using Gemini.
    For non-meeting content (lectures, podcasts), uses simplified classification.
    Uses sliding context windows for multi-turn understanding.
    Runs a second pass on low-confidence items.
    """
    if not utterances:
        return []

    # For non-meeting content, use simplified classification
    if content_type != "meeting":
        return await _classify_non_meeting(utterances, content_type)

    all_classified = []

    for i in range(0, len(utterances), batch_size):
        batch = utterances[i:i + batch_size]

        # Gather context: utterances before and after this batch
        context_before = utterances[max(0, i - CONTEXT_WINDOW):i]
        context_after = utterances[i + batch_size:i + batch_size + CONTEXT_WINDOW]

        classified = await _classify_batch(batch, context_before, context_after)
        all_classified.extend(classified)
        logger.info(f"Classified batch {i // batch_size + 1}: {len(classified)} utterances")

    # Second pass: re-examine low-confidence items with more context
    low_confidence = [(idx, c) for idx, c in enumerate(all_classified) if c.classification_confidence < 0.7]

    if low_confidence and len(low_confidence) <= 10:
        logger.info(f"Re-examining {len(low_confidence)} low-confidence classifications")
        all_classified = await _second_pass(all_classified, low_confidence)

    return all_classified


async def _classify_non_meeting(
    utterances: list[Utterance],
    content_type: str,
) -> list[ClassifiedUtterance]:
    """Simplified classification for lectures, podcasts, interviews."""
    classified = []
    for u in utterances:
        # Simple heuristic: questions → open_question, everything else → discussion
        text_lower = u.text.lower().strip()
        is_question = text_lower.endswith("?") or any(
            text_lower.startswith(w) for w in ["who ", "what ", "how ", "why ", "when ", "where ", "does ", "is ", "are ", "can ", "could "]
        )

        classified.append(ClassifiedUtterance(
            speaker=u.speaker,
            text=u.text,
            start_time=u.start_time,
            end_time=u.end_time,
            utterance_type=UtteranceType.OPEN_QUESTION if is_question else UtteranceType.DISCUSSION,
            classification_confidence=0.85,
            reasoning=f"Auto-classified for {content_type} content",
        ))

    logger.info(f"Non-meeting classification ({content_type}): {len(classified)} utterances")
    return classified


async def _classify_batch(
    batch: list[Utterance],
    context_before: list[Utterance],
    context_after: list[Utterance],
) -> list[ClassifiedUtterance]:
    """Classify a batch with surrounding context."""

    # Build context strings
    context_str = ""
    if context_before:
        before_json = json.dumps(
            [{"speaker": u.speaker, "text": u.text} for u in context_before],
            indent=2,
        )
        context_str += f"\n\nPreceding context (for reference only, do NOT classify these):\n{before_json}\n"

    if context_after:
        after_json = json.dumps(
            [{"speaker": u.speaker, "text": u.text} for u in context_after],
            indent=2,
        )
        context_str += f"\nFollowing context (for reference only, do NOT classify these):\n{after_json}\n"

    input_json = json.dumps(
        [{"speaker": u.speaker, "text": u.text} for u in batch],
        indent=2,
    )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Examples:\n{FEW_SHOT_EXAMPLES}\n"
        f"{context_str}\n"
        f"Now classify ONLY these {len(batch)} utterances:\n{input_json}"
    )

    try:
        classified_data = await gemini_generate(prompt, max_retries=3, parse_json=True)

        if not isinstance(classified_data, list):
            raise ValueError(f"Expected list, got {type(classified_data)}")

        result = []
        for idx, orig in enumerate(batch):
            if idx < len(classified_data):
                cls = classified_data[idx]
                try:
                    result.append(ClassifiedUtterance(
                        speaker=orig.speaker,
                        text=orig.text,
                        start_time=orig.start_time,
                        end_time=orig.end_time,
                        utterance_type=UtteranceType(cls.get("utterance_type", "discussion")),
                        classification_confidence=float(cls.get("confidence", 0.5)),
                        reasoning=cls.get("reasoning", ""),
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Bad classification for utterance {idx}: {e}")
                    result.append(_fallback_classify(orig))
            else:
                logger.warning(f"Missing classification for utterance {idx}, using fallback")
                result.append(_fallback_classify(orig))

        return result

    except Exception as e:
        logger.error(f"Classification failed for batch: {e}")
        return [_fallback_classify(u) for u in batch]


async def _second_pass(
    all_classified: list[ClassifiedUtterance],
    low_confidence: list[tuple[int, ClassifiedUtterance]],
) -> list[ClassifiedUtterance]:
    """Re-examine low-confidence classifications with expanded context."""

    for idx, item in low_confidence:
        # Get 5 utterances of context on each side
        context_start = max(0, idx - 5)
        context_end = min(len(all_classified), idx + 6)
        context_items = all_classified[context_start:context_end]

        context_text = "\n".join(
            f"[{c.utterance_type.value.upper()}] [{c.speaker}]: {c.text}"
            for c in context_items
        )

        prompt = f"""Re-examine this single utterance classification. It was previously classified with low confidence.

Full conversation context (already classified):
{context_text}

The utterance to re-examine:
Speaker: {item.speaker}
Text: "{item.text}"
Previous classification: {item.utterance_type.value} (confidence: {item.classification_confidence})
Previous reasoning: {item.reasoning}

Given the surrounding context, what is the correct classification?
Return a JSON object with: "utterance_type", "confidence", "reasoning"
utterance_type must be one of: "decision", "commitment", "discussion", "open_question"

Return ONLY valid JSON."""

        try:
            result = await gemini_generate(prompt, max_retries=2, parse_json=True)

            new_type = UtteranceType(result.get("utterance_type", item.utterance_type.value))
            new_confidence = float(result.get("confidence", item.classification_confidence))

            # Only update if the new classification is more confident
            if new_confidence > item.classification_confidence:
                all_classified[idx] = ClassifiedUtterance(
                    speaker=item.speaker,
                    text=item.text,
                    start_time=item.start_time,
                    end_time=item.end_time,
                    utterance_type=new_type,
                    classification_confidence=new_confidence,
                    reasoning=result.get("reasoning", item.reasoning) + " [re-examined]",
                )
                logger.info(
                    f"Re-classified utterance {idx}: "
                    f"{item.utterance_type.value} → {new_type.value} "
                    f"(confidence: {item.classification_confidence:.2f} → {new_confidence:.2f})"
                )

        except Exception as e:
            logger.warning(f"Second-pass failed for utterance {idx}: {e}")

    return all_classified


def _fallback_classify(utterance: Utterance) -> ClassifiedUtterance:
    """Fallback classification when Gemini fails."""
    return ClassifiedUtterance(
        speaker=utterance.speaker,
        text=utterance.text,
        start_time=utterance.start_time,
        end_time=utterance.end_time,
        utterance_type=UtteranceType.DISCUSSION,
        classification_confidence=0.0,
        reasoning="Classification failed — defaulted to discussion",
    )
