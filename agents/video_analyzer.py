"""
Video analyzer — uses Gemini Vision to extract visual context from video frames.
Identifies speakers, slides, screen content, and scene descriptions.
"""
from __future__ import annotations

from pathlib import Path
from core.logging import get_logger
from core.gemini_client import gemini_generate_multimodal

logger = get_logger(__name__)


async def analyze_video_frames(
    frames: list[tuple[float, Path]],
    transcript_excerpt: str = "",
) -> dict:
    """
    Analyze video frames with Gemini Vision to extract:
    - Speaker visual cues (faces, positions, name tags)
    - Slide/presentation content
    - Scene descriptions
    """
    if not frames:
        return {"speaker_cues": [], "slides": [], "scene": ""}

    all_speaker_cues = []
    all_slides = []
    scene_description = ""

    image_paths = [p for _, p in frames]
    timestamps = [t for t, _ in frames]
    timestamp_list = ", ".join(f"{t:.0f}s" for t in timestamps)

    prompt = f"""Analyze these {len(frames)} video frames from a meeting/lecture recording.
The frames are taken at timestamps (in order): {timestamp_list}

For each frame, identify:

1. **People visible**: Describe anyone on camera — appearance, position, Zoom/Teams name banners, name tags. If a person appears to be speaking (mouth open, gesturing, highlighted in video call UI), note that.
2. **Slides/Screen content**: If there's a presentation slide, shared screen, or whiteboard visible, extract the key text and describe the content.
3. **Scene**: Briefly describe the setting (conference room, Zoom call, lecture hall, etc.)

{f"Transcript context: {transcript_excerpt[:500]}" if transcript_excerpt else ""}

Return a JSON object:
{{
  "speaker_cues": [
    {{"frame_index": 0, "timestamp": {timestamps[0]}, "description": "...", "possible_name": "..." or null}}
  ],
  "slides": [
    {{"frame_index": 0, "timestamp": {timestamps[0]}, "title": "...", "content": "extracted text from slide"}}
  ],
  "scene": "brief scene description"
}}

Rules:
- Only include speaker_cues entries where you can actually see a person.
- Only include slides entries where there is actual slide/screen content visible.
- For possible_name, only fill in if you can read a clear identifier — otherwise null.
- YOU MUST format your response as ONLY pure valid JSON."""

    try:
        result = await gemini_generate_multimodal(
            prompt=prompt,
            image_paths=image_paths,
            max_retries=3,
            parse_json=True,
        )

        if isinstance(result, dict):
            # Adjust timestamps from frame_index safely
            for cue in result.get("speaker_cues", []):
                fi = cue.get("frame_index", 0)
                if 0 <= fi < len(timestamps):
                    cue["timestamp"] = timestamps[fi]
                all_speaker_cues.append(cue)

            for slide in result.get("slides", []):
                fi = slide.get("frame_index", 0)
                if 0 <= fi < len(timestamps):
                    slide["timestamp"] = timestamps[fi]
                all_slides.append(slide)

            if result.get("scene"):
                scene_description = result["scene"]

    except Exception as e:
        logger.warning(f"Video analysis failed: {e}")

    logger.info(
        f"Video analysis: {len(all_speaker_cues)} speaker cues, "
        f"{len(all_slides)} slides detected"
    )

    return {
        "speaker_cues": all_speaker_cues,
        "slides": all_slides,
        "scene": scene_description,
    }


def format_visual_context_for_diarization(visual_context: dict) -> str:
    """
    Format visual analysis results into a compact, deduplicated string
    for the diarization prompt.

    Raw video analysis can produce 200+ cues (one per person per frame).
    This function collapses them into a unique participant roster with
    one description each, keeping the context short enough (~1-2KB) to
    actually help the diarization model rather than overwhelming it.
    """
    lines = []

    if visual_context.get("scene"):
        lines.append(f"Setting: {visual_context['scene']}")

    cues = visual_context.get("speaker_cues", [])
    if not cues:
        if not lines:
            return ""
        return "\n".join(lines)

    # ── Deduplicate by name ─────────────────────────
    # Keep the first (longest) description per identified name
    named: dict[str, dict] = {}   # name -> best cue
    unnamed: list[dict] = []

    for c in cues:
        name = c.get("possible_name")
        if name and name.lower() not in ("null", "none", "unknown", ""):
            key = name.strip()
            existing = named.get(key)
            desc = c.get("description", "")
            if not existing or len(desc) > len(existing.get("description", "")):
                named[key] = c
        else:
            # Only keep unnamed if we haven't already seen many
            if len(unnamed) < 3:
                unnamed.append(c)

    # ── Build concise participant roster ────────────
    if named:
        lines.append(f"\nIdentified participants from video ({len(named)} people):")
        for name, c in named.items():
            desc = c.get("description", "")
            # Truncate very long descriptions
            if len(desc) > 120:
                desc = desc[:117] + "..."
            lines.append(f"  - {name}: {desc}")

    if unnamed:
        lines.append("\nUnidentified speakers:")
        for c in unnamed:
            ts = c.get("timestamp", "?")
            desc = c.get("description", "")
            if len(desc) > 100:
                desc = desc[:97] + "..."
            lines.append(f"  - At {ts}s: {desc}")

    return "\n".join(lines)
