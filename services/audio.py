"""
Audio preprocessing utilities.
Normalizes and converts audio to Whisper's optimal format (16kHz mono WAV).
"""
import subprocess
import tempfile
from pathlib import Path
from core.logging import get_logger

logger = get_logger(__name__)


async def preprocess_audio(audio_path: Path) -> Path:
    """
    Convert any audio file to 16kHz mono WAV for optimal Whisper performance.
    Normalizes volume levels to handle quiet/loud speakers.
    Returns the path to the processed WAV file.
    """
    output_path = audio_path.with_suffix(".processed.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ac", "1",              # mono
        "-ar", "16000",          # 16kHz sample rate
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",  # EBU R128 loudness normalization
        "-acodec", "pcm_s16le",  # 16-bit PCM
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for long files
        )

        if result.returncode != 0:
            logger.warning(f"ffmpeg preprocessing failed: {result.stderr[:500]}")
            logger.info("Falling back to original audio file")
            return audio_path

        logger.info(f"Audio preprocessed: {audio_path.name} → {output_path.name}")
        return output_path

    except FileNotFoundError:
        logger.warning("ffmpeg not found — skipping audio preprocessing")
        return audio_path
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out — using original file")
        return audio_path
    except Exception as e:
        logger.warning(f"Audio preprocessing error: {e} — using original file")
        return audio_path
