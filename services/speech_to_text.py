"""
EduMentor AI - Speech-to-Text Service
Converts student's spoken questions into text.
Uses SpeechRecognition library with Google Web Speech API (free, no API key).
"""

import time
import os
import tempfile
import wave
import subprocess

from utils.metrics_logger import metrics


def _get_ffmpeg_path() -> str:
    """Get ffmpeg binary path from imageio-ffmpeg package."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # Hope it's on PATH


def _convert_to_wav(audio_path: str) -> str:
    """Convert any audio file to WAV using ffmpeg."""
    wav_path = os.path.join(tempfile.gettempdir(), "edumentor_converted.wav")

    ffmpeg_path = _get_ffmpeg_path()

    cmd = [
        ffmpeg_path,
        "-y",              # Overwrite output
        "-i", audio_path,  # Input file
        "-ar", "16000",    # Sample rate 16kHz
        "-ac", "1",        # Mono
        "-sample_fmt", "s16",  # 16-bit
        wav_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        if os.path.exists(wav_path):
            return wav_path
    except Exception:
        pass

    return None


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio input to text using Google Web Speech API.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dictionary with 'text' (transcribed text) and metadata.
    """
    if not audio_path or not os.path.exists(audio_path):
        return {
            "text": "",
            "error": "No audio file provided or file not found.",
            "language": "",
            "duration": 0,
        }

    start_time = time.time()

    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300

        # Determine if we need to convert
        file_to_use = audio_path

        # If not WAV/FLAC/AIFF, convert to WAV using ffmpeg
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in (".wav", ".flac", ".aiff"):
            converted = _convert_to_wav(audio_path)
            if converted:
                file_to_use = converted
            else:
                raise Exception(
                    f"Cannot process '{ext}' audio. Conversion failed."
                )

        # Read and transcribe
        with sr.AudioFile(file_to_use) as source:
            audio_data = recognizer.record(source)

        # Use Google Web Speech API (free, no key required)
        transcribed_text = recognizer.recognize_google(audio_data)

        latency = time.time() - start_time
        metrics.log_request(
            service="speech_to_text",
            latency=latency,
            tokens_used=0,
            success=True,
        )

        return {
            "text": transcribed_text,
            "error": None,
            "language": "en",
            "duration": round(latency, 2),
        }

    except ImportError:
        latency = time.time() - start_time
        metrics.log_request(service="speech_to_text", latency=latency, tokens_used=0, success=False)
        return {
            "text": "",
            "error": "SpeechRecognition not installed. Run: pip install SpeechRecognition",
            "language": "",
            "duration": round(latency, 2),
        }

    except Exception as e:
        latency = time.time() - start_time
        metrics.log_request(service="speech_to_text", latency=latency, tokens_used=0, success=False)
        return {
            "text": "",
            "error": f"Transcription failed: {str(e)}",
            "language": "",
            "duration": round(latency, 2),
        }


def transcribe_from_gradio(audio_input) -> str:
    """
    Handle Gradio audio input and return transcribed text.

    Args:
        audio_input: Audio input from Gradio (file path or tuple).

    Returns:
        Transcribed text string.
    """
    if audio_input is None:
        return ""

    # Gradio passes a file path directly when type="filepath"
    if isinstance(audio_input, str):
        result = transcribe_audio(audio_input)

    elif isinstance(audio_input, tuple):
        # Gradio might pass (sample_rate, numpy_array)
        try:
            import numpy as np

            sample_rate, audio_data = audio_input

            # Save as WAV file
            temp_path = os.path.join(tempfile.gettempdir(), "edumentor_audio.wav")

            # Handle different numpy dtypes
            if hasattr(audio_data, 'dtype'):
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype != np.int16:
                    audio_data = audio_data.astype(np.int16)

            # Ensure 1D (mono)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            # Write WAV
            with wave.open(temp_path, "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            result = transcribe_audio(temp_path)

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            return f"[Error processing audio: {str(e)}]"
    else:
        return "Unsupported audio format."

    if result["error"]:
        return f"[Error: {result['error']}]"

    return result["text"]
