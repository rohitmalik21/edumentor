"""
EduMentor AI - Configuration Module
Loads environment variables and provides app-wide settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Redirect Hugging Face cache to D: drive (C: drive is full)
hf_home = os.getenv("HF_HOME", "")
if hf_home:
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_home
    os.makedirs(hf_home, exist_ok=True)


class Config:
    """Central configuration for EduMentor AI."""

    # LLM Provider: "gemini", "openai", or "local"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Model Names
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Local Hugging Face Model
    LOCAL_MODEL = os.getenv("LOCAL_MODEL", "google/flan-t5-large")

    # Embedding
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # App Settings
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "7860"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
    AUDIO_DIR = os.path.join(BASE_DIR, "audio_output")
    METRICS_DIR = os.path.join(BASE_DIR, "metrics_logs")
    FINETUNED_MODEL_DIR = os.path.join(BASE_DIR, "finetuned_model")

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.UPLOAD_DIR, cls.AUDIO_DIR, cls.METRICS_DIR]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def validate(cls):
        """Validate that required API keys are present."""
        if cls.LLM_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when using Gemini provider")
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        # Local provider needs no API key


# Create directories on import
Config.ensure_directories()
