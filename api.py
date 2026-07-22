"""
EduMentor AI - FastAPI REST API
Provides versioned REST endpoints for all 7 sub-tasks.

Usage:
    python api.py

Swagger docs: http://localhost:8000/docs
"""

import sys
import os
import time
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import Config
from utils.document_parser import extract_text_from_file
from utils.metrics_logger import metrics
from services.summarizer import summarize_material
from services.simplifier import simplify_concept, LEVELS
from services.question_answering import load_material_for_qa, answer_question
from services.quiz_generator import generate_quiz, format_quiz_for_display
from services.answer_evaluator import evaluate_answers
from services.speech_to_text import transcribe_audio
from services.text_to_speech import synthesize_speech


# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title="EduMentor AI API",
    description=(
        "API-based Personalized Learning & Assessment Assistant. "
        "Supports summarization, concept simplification, grounded Q&A (RAG), "
        "adaptive quiz generation, answer evaluation, speech-to-text, and text-to-speech."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# In-memory storage for session material
_material_store = {"text": "", "filename": ""}


# ============================================
# Request/Response Models (Pydantic)
# ============================================
class SummarizeRequest(BaseModel):
    text: str | None = Field(None, description="Text to summarize (optional if material already uploaded)")


class SummarizeResponse(BaseModel):
    summary: str
    word_count: int
    error: bool


class ExplainRequest(BaseModel):
    topic: str = Field(..., description="Topic or concept to explain")
    level: str = Field("Beginner", description="Explanation level: Beginner, School Student, Undergraduate, Advanced")


class ExplainResponse(BaseModel):
    explanation: str
    level: str
    error: bool


class AskRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the study material")


class AskResponse(BaseModel):
    answer: str
    context_used: str
    relevance_score: float
    error: bool


class QuizGenerateRequest(BaseModel):
    num_questions: int = Field(5, ge=1, le=15, description="Number of questions (1-15)")
    difficulty: str = Field("Medium", description="Difficulty: Easy, Medium, Hard")
    question_types: str = Field("MCQ, True/False, Short Answer", description="Comma-separated question types")


class QuizGenerateResponse(BaseModel):
    quiz: dict | None
    display_text: str
    error: str | None


class QuizEvaluateRequest(BaseModel):
    quiz_data: dict = Field(..., description="Quiz data from quiz/generate endpoint")
    student_answers: dict = Field(..., description="Student answers as {question_id: answer}")


class QuizEvaluateResponse(BaseModel):
    feedback: str
    score: int
    total: int
    percentage: float
    error: bool


class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    material_loaded: bool
    uptime_seconds: float


# ============================================
# Startup time tracking
# ============================================
_start_time = time.time()


# ============================================
# Endpoints
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check application health and configuration."""
    return HealthResponse(
        status="healthy",
        provider=Config.LLM_PROVIDER,
        model=Config.LOCAL_MODEL if Config.LLM_PROVIDER == "local" else (
            Config.GEMINI_MODEL if Config.LLM_PROVIDER == "gemini" else Config.OPENAI_MODEL
        ),
        material_loaded=bool(_material_store["text"]),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/api/v1/material/process", tags=["Material"])
async def process_material(file: UploadFile = File(...)):
    """
    Upload and process study material (PDF, DOCX, TXT).
    The material is stored in memory for use by other endpoints.
    """
    # Validate file type
    allowed_extensions = [".pdf", ".docx", ".txt", ".md"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed_extensions}")

    # Save to temp file for processing
    temp_path = os.path.join(tempfile.gettempdir(), f"edumentor_upload{ext}")
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        text = extract_text_from_file(temp_path)

        # Store material
        _material_store["text"] = text
        _material_store["filename"] = file.filename

        # Index for RAG
        index_status = load_material_for_qa(text)

        # Save to uploads folder
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        shutil.copy2(temp_path, save_path)

        return {
            "status": "success",
            "filename": file.filename,
            "word_count": len(text.split()),
            "char_count": len(text),
            "index_status": index_status,
        }
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/v1/summarize", response_model=SummarizeResponse, tags=["NLP"])
def summarize(request: SummarizeRequest = None):
    """
    Generate a structured summary of the study material.
    Uses uploaded material if no text is provided in the request.
    """
    text = ""
    if request and request.text:
        text = request.text
    else:
        text = _material_store.get("text", "")

    if not text:
        raise HTTPException(400, "No text provided and no material uploaded. Upload material first.")

    result = summarize_material(text)
    return SummarizeResponse(
        summary=result["summary"],
        word_count=len(text.split()),
        error=result.get("error", False),
    )


@app.post("/api/v1/explain", response_model=ExplainResponse, tags=["NLP"])
def explain(request: ExplainRequest):
    """
    Simplify and explain a concept at the specified comprehension level.
    Levels: Beginner, School Student, Undergraduate, Advanced.
    """
    if request.level not in LEVELS:
        raise HTTPException(400, f"Invalid level. Choose from: {LEVELS}")

    context = _material_store.get("text", "")[:3000]
    result = simplify_concept(request.topic, context, request.level)

    return ExplainResponse(
        explanation=result["explanation"],
        level=request.level,
        error=result.get("error", False),
    )


@app.post("/api/v1/ask", response_model=AskResponse, tags=["NLP"])
def ask(request: AskRequest):
    """
    Answer a question using only the uploaded study material (RAG-based).
    Returns the answer, supporting context, and relevance score.
    """
    if not _material_store.get("text"):
        raise HTTPException(400, "No study material loaded. Upload material first via /api/v1/material/process")

    result = answer_question(request.question)

    return AskResponse(
        answer=result["answer"],
        context_used=result["context_used"],
        relevance_score=result["relevance_score"],
        error=result.get("error", False),
    )


@app.post("/api/v1/quiz/generate", response_model=QuizGenerateResponse, tags=["NLP"])
def quiz_generate(request: QuizGenerateRequest):
    """
    Generate an adaptive quiz from the uploaded study material.
    Supports MCQ, True/False, and Short Answer question types.
    """
    material = _material_store.get("text", "")
    if not material:
        raise HTTPException(400, "No study material loaded. Upload material first.")

    result = generate_quiz(
        material=material,
        num_questions=request.num_questions,
        difficulty=request.difficulty,
        question_types=request.question_types,
    )

    display_text = ""
    if result.get("quiz"):
        display_text = format_quiz_for_display(result["quiz"])

    return QuizGenerateResponse(
        quiz=result.get("quiz"),
        display_text=display_text,
        error=result.get("error"),
    )


@app.post("/api/v1/quiz/evaluate", response_model=QuizEvaluateResponse, tags=["NLP"])
def quiz_evaluate(request: QuizEvaluateRequest):
    """
    Evaluate student answers against a quiz and provide detailed feedback.
    """
    result = evaluate_answers(request.quiz_data, request.student_answers)

    return QuizEvaluateResponse(
        feedback=result["feedback"],
        score=result["score"],
        total=result["total"],
        percentage=result.get("percentage", 0),
        error=result.get("error", False),
    )


@app.post("/api/v1/speech/transcribe", tags=["Speech"])
async def speech_transcribe(file: UploadFile = File(...)):
    """
    Convert speech audio to text (Speech-to-Text).
    Supports WAV, MP3, WEBM formats.
    """
    # Save uploaded audio to temp file
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"
    temp_path = os.path.join(tempfile.gettempdir(), f"edumentor_stt{ext}")

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        result = transcribe_audio(temp_path)
        return {
            "text": result["text"],
            "language": result.get("language", "en"),
            "duration": result.get("duration", 0),
            "error": result.get("error"),
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/v1/speech/synthesize", tags=["Speech"])
def speech_synthesize(request: SynthesizeRequest):
    """
    Convert text to speech audio (Text-to-Speech).
    Returns an MP3 audio file.
    """
    result = synthesize_speech(request.text, "api_tts_output.mp3")

    if result["error"]:
        raise HTTPException(500, f"TTS failed: {result['error']}")

    if result["audio_path"] and os.path.exists(result["audio_path"]):
        return FileResponse(
            result["audio_path"],
            media_type="audio/mpeg",
            filename="edumentor_speech.mp3",
        )
    else:
        raise HTTPException(500, "Audio file was not generated.")


@app.get("/api/v1/metrics", tags=["LLMOps"])
def get_metrics():
    """
    Get LLMOps metrics: latency, token usage, throughput, success rate, etc.
    """
    summary = metrics.get_summary()
    return JSONResponse(content=summary)


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  EduMentor AI - FastAPI REST API")
    print("=" * 60)
    print(f"  Swagger Docs: http://localhost:8000/docs")
    print(f"  ReDoc:        http://localhost:8000/redoc")
    print(f"  Health:       http://localhost:8000/health")
    print(f"  LLM Provider: {Config.LLM_PROVIDER}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
