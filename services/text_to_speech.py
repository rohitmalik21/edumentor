"""
EduMentor AI - Text-to-Speech Service
Converts explanations and answers into audio for accessible learning.
"""

import time
import os
import re

from config import Config
from utils.metrics_logger import metrics


def synthesize_speech(text: str, filename: str = "response_audio.mp3") -> dict:
    """
    Convert text to speech using gTTS (Google Text-to-Speech).

    Args:
        text: Text to convert to speech.
        filename: Output audio filename.

    Returns:
        Dictionary with 'audio_path' and metadata.
    """
    if not text or not text.strip():
        return {
            "audio_path": None,
            "error": "No text provided for speech synthesis.",
            "duration": 0,
        }

    start_time = time.time()

    try:
        from gtts import gTTS

        # Clean text for speech (remove markdown formatting)
        clean_text = _clean_text_for_speech(text)

        # Truncate if too long (gTTS has limits)
        if len(clean_text) > 5000:
            clean_text = clean_text[:5000] + ". Content truncated for audio."

        # Generate audio
        tts = gTTS(text=clean_text, lang="en", slow=False)

        # Save to audio directory
        audio_path = os.path.join(Config.AUDIO_DIR, filename)
        tts.save(audio_path)

        latency = time.time() - start_time

        # Log metrics
        metrics.log_request(
            service="text_to_speech",
            latency=latency,
            tokens_used=len(clean_text.split()),  # Word count as proxy
            success=True,
        )

        return {
            "audio_path": audio_path,
            "error": None,
            "duration": round(latency, 2),
            "word_count": len(clean_text.split()),
        }

    except ImportError:
        latency = time.time() - start_time
        metrics.log_request(
            service="text_to_speech",
            latency=latency,
            tokens_used=0,
            success=False,
        )
        return {
            "audio_path": None,
            "error": "gTTS is not installed. Run: pip install gTTS",
            "duration": round(latency, 2),
        }

    except Exception as e:
        latency = time.time() - start_time
        metrics.log_request(
            service="text_to_speech",
            latency=latency,
            tokens_used=0,
            success=False,
        )
        return {
            "audio_path": None,
            "error": f"Speech synthesis failed: {str(e)}",
            "duration": round(latency, 2),
        }


def _clean_text_for_speech(text: str) -> str:
    """
    Remove markdown formatting and special characters for natural speech.

    Args:
        text: Raw text with possible markdown.

    Returns:
        Clean text suitable for TTS.
    """
    # Remove markdown headers
    text = re.sub(r"#{1,6}\s*", "", text)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Remove bullet points
    text = re.sub(r"^\s*[-*+]\s*", "", text, flags=re.MULTILINE)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove multiple newlines
    text = re.sub(r"\n{2,}", ". ", text)
    # Replace single newlines with spaces
    text = text.replace("\n", " ")
    # Remove multiple spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()
