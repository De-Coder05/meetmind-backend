"""
Centralized Gemini client with exponential backoff retries,
robust JSON parsing, and rate-limit handling.
"""
from __future__ import annotations

import json
import re
import asyncio
import google.generativeai as genai
from typing import Optional, Any
from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

genai.configure(api_key=settings.gemini_api_key)
_model = None


def _get_model():
    """Lazy model init so .env changes take effect on reload."""
    global _model
    if _model is None:
        _model = genai.GenerativeModel(settings.gemini_model)
        logger.info(f"Gemini model initialized: {settings.gemini_model}")
    return _model


def parse_json_response(raw: str) -> Any:
    """
    Bulletproof JSON parser that handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    - Trailing commas (common LLM mistake)
    - Partial/truncated JSON
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        # Extract content between first ``` and last ```
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            # Remove language identifier (e.g., "json")
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()
        else:
            # Fallback: strip first line and last ```
            lines = text.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

    # Remove trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first JSON array or object
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        # Find matching closing bracket
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
            if depth == 0:
                candidate = text[start_idx:i+1]
                candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    raise json.JSONDecodeError("Could not extract valid JSON from response", text, 0)


async def gemini_generate(
    prompt: str,
    max_retries: int = 3,
    parse_json: bool = True,
    temperature: Optional[float] = None,
) -> Any:
    """
    Call Gemini with exponential backoff retries.

    Args:
        prompt: The prompt to send.
        max_retries: Number of retry attempts.
        parse_json: If True, parse the response as JSON.
        temperature: Optional temperature override.

    Returns:
        Parsed JSON (dict/list) if parse_json=True, else raw text string.

    Raises:
        Exception: If all retries are exhausted.
    """
    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature

    last_error = None

    for attempt in range(max_retries):
        try:
            response = _get_model().generate_content(
                prompt,
                generation_config=generation_config if generation_config else None,
            )

            raw_text = response.text.strip()

            if not raw_text:
                raise ValueError("Empty response from Gemini")

            if parse_json:
                return parse_json_response(raw_text)
            else:
                return raw_text

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                f"Gemini JSON parse failed (attempt {attempt+1}/{max_retries}): {e}"
            )
            # Don't retry JSON errors immediately — the model might need a nudge
            if attempt < max_retries - 1:
                # Append a reminder to the prompt to enforce JSON
                prompt = prompt + "\n\nIMPORTANT: You MUST return ONLY valid JSON. No markdown, no explanation."
                await asyncio.sleep(1)

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Rate limit handling
            if "429" in error_str or "resource exhausted" in error_str:
                wait = min(2 ** (attempt + 2), 30)  # 4s, 8s, 16s
                logger.warning(f"Gemini rate limited, waiting {wait}s (attempt {attempt+1})")
                await asyncio.sleep(wait)
            elif "500" in error_str or "503" in error_str:
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                logger.warning(f"Gemini server error, retrying in {wait}s (attempt {attempt+1})")
                await asyncio.sleep(wait)
            else:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Gemini call failed: {e} — retrying in {wait}s (attempt {attempt+1})")
                await asyncio.sleep(wait)

    # All retries exhausted
    logger.error(f"Gemini call failed after {max_retries} attempts: {last_error}")
    raise last_error


async def gemini_generate_multimodal(
    prompt: str,
    image_paths: list[Path | str],
    max_retries: int = 3,
    parse_json: bool = True,
    temperature: Optional[float] = None,
) -> Any:
    """
    Call Gemini with text + images (multimodal).

    Args:
        prompt: The text prompt.
        image_paths: List of paths to image files.
        max_retries: Number of retry attempts.
        parse_json: If True, parse the response as JSON.
        temperature: Optional temperature override.

    Returns:
        Parsed JSON (dict/list) if parse_json=True, else raw text string.
    """
    from PIL import Image
    from pathlib import Path as P

    # Load images
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(P(img_path))
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")

    if not images:
        logger.warning("No valid images loaded, falling back to text-only")
        return await gemini_generate(prompt, max_retries, parse_json, temperature)

    # Build multimodal content: [image1, image2, ..., text_prompt]
    content = images + [prompt]

    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature

    last_error = None

    for attempt in range(max_retries):
        try:
            response = _get_model().generate_content(
                content,
                generation_config=generation_config if generation_config else None,
            )

            raw_text = response.text.strip()

            if not raw_text:
                raise ValueError("Empty response from Gemini")

            if parse_json:
                return parse_json_response(raw_text)
            else:
                return raw_text

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"Gemini multimodal JSON parse failed (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON."
                content = images + [prompt]
                await asyncio.sleep(1)

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in error_str or "resource exhausted" in error_str:
                wait = min(2 ** (attempt + 2), 30)
                logger.warning(f"Gemini rate limited, waiting {wait}s")
                await asyncio.sleep(wait)
            else:
                wait = 2 ** attempt
                logger.warning(f"Gemini multimodal failed: {e} — retrying in {wait}s")
                await asyncio.sleep(wait)

    logger.error(f"Gemini multimodal failed after {max_retries} attempts: {last_error}")
    raise last_error
