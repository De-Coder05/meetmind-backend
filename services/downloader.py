"""
Audio downloader for YouTube, Vimeo, and direct URLs.
Uses yt-dlp for platform video extraction and falls back to HTTP for direct links.
Returns both audio and video paths when video is available.
"""
from __future__ import annotations

import uuid
import subprocess
import httpx
from pathlib import Path
from typing import Optional
from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DownloadResult:
    """Holds both audio and optional video paths."""
    def __init__(self, audio_path: Path, video_path: Optional[Path] = None):
        self.audio_path = audio_path
        self.video_path = video_path


async def download_audio(url: str, output_dir: Optional[Path] = None) -> DownloadResult:
    """
    Download audio (and video when available) from a URL.
    Returns a DownloadResult with audio_path and optional video_path.
    """
    out_dir = output_dir or Path(settings.upload_dir)
    out_dir.mkdir(exist_ok=True)
    file_id = str(uuid.uuid4())

    # Try yt-dlp first
    try:
        return await _download_with_ytdlp(url, out_dir, file_id)
    except Exception as e:
        logger.warning(f"yt-dlp failed: {e}. Trying direct HTTP download...")

    # Fallback: direct HTTP download
    try:
        return await _download_direct(url, out_dir, file_id)
    except Exception as e:
        logger.error(f"All download methods failed for URL: {url}")
        raise ValueError(f"Could not download audio from URL: {url}. Error: {e}")


async def _download_with_ytdlp(url: str, out_dir: Path, file_id: str) -> DownloadResult:
    """Download audio + video using yt-dlp with multi-strategy fallback."""
    audio_path = None
    video_path = None

    # ── Strategy list: try each until one works ──
    strategies = [
        # 1. android_vr client (was reliable, may still work)
        ["--extractor-args", "youtube:player_client=android_vr"],
        # 2. Try browser cookies (Chrome, then Firefox)
        ["--cookies-from-browser", "chrome"],
        ["--cookies-from-browser", "firefox"],
        # 3. Bare default — sometimes works for non-age-restricted content
        [],
    ]

    # Step 1: Download video (for frame extraction)
    video_template = str(out_dir / f"{file_id}_video.%(ext)s")

    for strategy in strategies:
        video_cmd = [
            "yt-dlp",
            *strategy,
            "-f", "best[height<=720]/best",
            "--no-playlist",
            "--output", video_template,
            "--no-check-certificates",
            "--quiet",
            "--no-warnings",
            url,
        ]

        strategy_name = " ".join(strategy) if strategy else "default"
        logger.info(f"yt-dlp video attempt ({strategy_name}): {url}")

        try:
            result = subprocess.run(video_cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                for f in out_dir.glob(f"{file_id}_video.*"):
                    video_path = f
                    logger.info(f"Video downloaded ({strategy_name}): {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
                    break
                if video_path:
                    break
            else:
                err = result.stderr[:200] if result.stderr else "unknown error"
                logger.warning(f"yt-dlp video ({strategy_name}) failed: {err}")
        except subprocess.TimeoutExpired:
            logger.warning(f"yt-dlp video ({strategy_name}) timed out")
        except Exception as e:
            logger.warning(f"yt-dlp video ({strategy_name}) error: {e}")

    # Step 2: Extract audio from video (or download audio-only)
    if video_path and video_path.exists():
        audio_out = out_dir / f"{file_id}.mp3"
        extract_cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "libmp3lame", "-q:a", "5",
            "-y", str(audio_out),
        ]
        try:
            subprocess.run(extract_cmd, capture_output=True, text=True, timeout=120)
            if audio_out.exists():
                audio_path = audio_out
                logger.info(f"Audio extracted: {audio_out.name}")
        except Exception as e:
            logger.warning(f"Audio extraction from video failed: {e}")

    if not audio_path:
        # Fallback: download audio separately with same strategy chain
        audio_template = str(out_dir / f"{file_id}.%(ext)s")

        for strategy in strategies:
            audio_cmd = [
                "yt-dlp",
                *strategy,
                "-f", "bestaudio/best",
                "--extract-audio",
                "--audio-format", "mp3",
                "--audio-quality", "5",
                "--no-playlist",
                "--output", audio_template,
                "--no-check-certificates",
                "--quiet",
                "--no-warnings",
                url,
            ]

            strategy_name = " ".join(strategy) if strategy else "default"
            logger.info(f"yt-dlp audio attempt ({strategy_name})")

            try:
                result = subprocess.run(audio_cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    for f in out_dir.glob(f"{file_id}.*"):
                        if f.suffix in (".mp3", ".m4a", ".wav", ".ogg", ".webm"):
                            audio_path = f
                            break
                    if audio_path:
                        break
                else:
                    err = result.stderr[:200] if result.stderr else "unknown error"
                    logger.warning(f"yt-dlp audio ({strategy_name}) failed: {err}")
            except subprocess.TimeoutExpired:
                logger.warning(f"yt-dlp audio ({strategy_name}) timed out")
            except Exception as e:
                logger.warning(f"yt-dlp audio ({strategy_name}) error: {e}")

    if not audio_path:
        raise FileNotFoundError("All yt-dlp strategies failed — could not download audio")

    logger.info(f"Download complete: audio={audio_path.name}, video={'yes' if video_path else 'no'}")
    return DownloadResult(audio_path=audio_path, video_path=video_path)


async def _download_direct(url: str, out_dir: Path, file_id: str) -> DownloadResult:
    """Direct HTTP download for raw audio file URLs."""
    logger.info(f"Direct HTTP download: {url}")

    async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
        response = await client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        ext = _guess_extension(url, content_type)

        output_path = out_dir / f"{file_id}{ext}"
        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return DownloadResult(audio_path=output_path)


def _guess_extension(url: str, content_type: str) -> str:
    """Guess file extension from URL or content type."""
    ct_map = {
        "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/ogg": ".ogg", "audio/webm": ".webm",
        "video/mp4": ".mp4",
    }
    for ct, ext in ct_map.items():
        if ct in content_type:
            return ext

    url_lower = url.lower().split("?")[0]
    for ext in [".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".webm"]:
        if url_lower.endswith(ext):
            return ext

    return ".mp3"
