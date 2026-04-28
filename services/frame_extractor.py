"""
Video frame extractor using ffmpeg.
Extracts keyframes at regular intervals for visual analysis.
"""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from core.logging import get_logger

logger = get_logger(__name__)

FRAME_INTERVAL = 15  # extract one frame every N seconds
MAX_FRAMES = 10      # cap frames — for video calls, 10 is plenty
FRAME_QUALITY = 5    # JPEG quality (2=best, 31=worst), 5 is good balance


async def extract_frames(
    video_path: Path,
    output_dir: Path | None = None,
    interval: int = FRAME_INTERVAL,
    max_frames: int = MAX_FRAMES,
) -> list[tuple[float, Path]]:
    """
    Extract keyframes from a video at regular intervals.
    
    Returns:
        List of (timestamp_seconds, image_path) tuples.
    """
    if not video_path.exists():
        logger.warning(f"Video file not found: {video_path}")
        return []

    # Ensure ffmpeg is available
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg not found — cannot extract frames")
        return []

    # Create output directory
    frames_dir = output_dir or video_path.parent / f"{video_path.stem}_frames"
    frames_dir.mkdir(exist_ok=True)

    # Get video duration first
    duration = _get_duration(video_path)
    if duration <= 0:
        logger.warning("Could not determine video duration")
        return []

    # Calculate actual interval to stay within max_frames limit
    total_possible = int(duration / interval)
    if total_possible > max_frames:
        interval = int(duration / max_frames)
        logger.info(f"Adjusted frame interval to {interval}s to stay within {max_frames} frame limit")

    # Extract frames with ffmpeg
    output_pattern = str(frames_dir / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps=1/{interval},scale=960:-1",  # 1 frame per interval, max 960px wide
        "-q:v", str(FRAME_QUALITY),
        "-y",                    # overwrite
        output_pattern,
    ]

    logger.info(f"Extracting frames every {interval}s from {video_path.name} ({duration:.0f}s total)")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg frame extraction failed: {result.stderr[:300]}")
            return []

    except subprocess.TimeoutExpired:
        logger.error("Frame extraction timed out")
        return []

    # Collect extracted frames with timestamps
    frames = []
    for frame_file in sorted(frames_dir.glob("frame_*.jpg")):
        # Frame number from filename (1-indexed)
        frame_num = int(frame_file.stem.split("_")[1])
        timestamp = (frame_num - 1) * interval
        frames.append((timestamp, frame_file))

    # Cap at max_frames
    frames = frames[:max_frames]

    logger.info(f"Extracted {len(frames)} frames from {video_path.name}")
    return frames


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired):
        return 0.0


async def cleanup_frames(frames_dir: Path):
    """Remove extracted frames directory."""
    try:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up frames: {e}")
