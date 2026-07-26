# EduMentor AI - Personalized Learning & Assessment Assistant

**Course:** API Driven Cloud Native Solutions  
**Group:** 32  
**Assignment:** 2  

| S.No. | Name | Student ID |
|-------|------|------------|
| 1 | Rohit Malik | 2024AC05988 |
| 2 | Suraj Prakash Uniyal | 2024AD05123 |
| 3 | Sudhakar Katam | 2024AC05889 |
| 4 | C S Krishna Chaitanya P | 2024AD05457 |
| 5 | Nikhil Gupta | 2024AC05640 |

---

## Project Overview

An API-based AI application that transforms student-provided study material into an interactive learning experience using **Natural Language Processing** and **Speech Recognition**.

| Item | Detail |
|------|--------|
| **Domain** | Education |
| **AI Categories** | NLP + Speech Recognition |
| **Sub-tasks** | 7 (Summarization, Simplification, Grounded QA, Quiz Generation, Answer Evaluation, STT, TTS) |
| **LLM Provider** | Local (FLAN-T5) / Google Gemini / OpenAI (configurable) |
| **Fine-tuning** | FLAN-T5-small on SciQ dataset |
| **UI Framework** | Gradio |
| **REST API** | FastAPI with Swagger documentation |
| **LLMOps Metrics** | 8 tracked (Latency, Success Rate, Tokens, Throughput, Relevance, Confidence, Prompt Versioning, System Resources) |

---

## Architecture

```
edumentor/
├── app.py                         # Gradio UI entry point
├── api.py                         # FastAPI REST API entry point
├── run.py                         # Launcher (starts both UI + API)
├── config.py                      # Configuration & environment
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template (no secrets)
├── .gitignore                     # Git ignore rules
├── services/                      # AI service modules
│   ├── summarizer.py              # Sub-task 1: Study material summarization
│   ├── simplifier.py              # Sub-task 2: Concept simplification by level
│   ├── question_answering.py      # Sub-task 3: RAG-based grounded QA (FAISS)
│   ├── quiz_generator.py          # Sub-task 4: Adaptive quiz generation
│   ├── answer_evaluator.py        # Sub-task 5: Answer scoring & feedback
│   ├── speech_to_text.py          # Sub-task 6: Speech-to-Text
│   └── text_to_speech.py          # Sub-task 7: Text-to-Speech (gTTS)
├── utils/                         # Shared utilities
│   ├── document_parser.py         # PDF/DOCX/TXT extraction (with table support)
│   ├── llm_client.py              # Unified LLM API client (local/Gemini/OpenAI)
│   └── metrics_logger.py          # LLMOps metrics tracking
├── ui/                            # Gradio interface (6 tabs)
│   ├── theme.py                   # Custom styling & branding
│   ├── tab_learn.py               # Learn tab (Upload, Summarize, Simplify)
│   ├── tab_ask.py                 # Ask Tutor tab (Text & Voice QA)
│   ├── tab_practice.py            # Practice tab (Quiz Generation)
│   ├── tab_results.py             # Results tab (Evaluation & Feedback)
│   ├── tab_monitoring.py          # Monitoring tab (LLMOps Dashboard)
│   └── tab_finetune.py            # Fine-tuning tab (Train & Compare)
└── fine_tuning/                   # Model fine-tuning
    ├── finetune_flan_t5.py        # Training script
    └── inference_finetuned.py     # Inference with fine-tuned model
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/rohitmalik21/edumentor.git
cd edumentor
```

### 2. Create a virtual environment (Python 3.11, 3.12, or 3.13 recommended)

```bash
py -3.13 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
```

> **Note:** Python 3.14 is NOT supported. Use 3.11, 3.12, or 3.13 for full compatibility.

### 3. Install PyTorch with GPU support (optional)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install all dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux
```

Edit `.env` and choose your LLM provider:
- `LLM_PROVIDER=local` → Free, no API key needed (uses FLAN-T5)
- `LLM_PROVIDER=gemini` → Free tier at https://aistudio.google.com/apikey
- `LLM_PROVIDER=openai` → Pay-per-use at https://platform.openai.com/api-keys

### 6. Run the application

```bash
# Run both Gradio UI + FastAPI together
python run.py

# Or run individually
python app.py           # Gradio UI only (port 7860)
python api.py           # FastAPI REST API only (port 8000)
python run.py --ui-only
python run.py --api-only
```

### 7. Access the application

| Interface | URL |
|-----------|-----|
| Gradio UI | http://localhost:7860 |
| Swagger API Docs | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Health Check | http://localhost:8000/health |
| LLMOps Metrics | http://localhost:8000/api/v1/metrics |

---

## REST API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/material/process` | Upload & process study material (PDF/DOCX/TXT) |
| POST | `/api/v1/summarize` | Generate structured summary |
| POST | `/api/v1/explain` | Simplify a concept at chosen level |
| POST | `/api/v1/ask` | Grounded Q&A from study material (RAG) |
| POST | `/api/v1/quiz/generate` | Generate adaptive quiz |
| POST | `/api/v1/quiz/evaluate` | Evaluate student answers |
| POST | `/api/v1/speech/transcribe` | Convert speech to text |
| POST | `/api/v1/speech/synthesize` | Convert text to speech |
| GET | `/api/v1/metrics` | LLMOps metrics dashboard |
| GET | `/health` | Application health check |

---

## Fine-tuning (Assignment Requirement)

Fine-tune FLAN-T5-small for educational question generation:

**Option 1: Via the app UI**
- Go to the "Fine-tuning" tab in Gradio
- Configure epochs, batch size, training samples
- Click "Start Fine-tuning"

**Option 2: Via command line**
```bash
python fine_tuning/finetune_flan_t5.py
```

**What it does:**
1. Downloads the SciQ dataset (cached after first download)
2. Fine-tunes FLAN-T5-small for question generation
3. Evaluates with ROUGE/BLEU metrics
4. Generates before/after comparison
5. Saves the model to `finetuned_model/`
6. The fine-tuned model is automatically used for quiz generation

---

## Student Journey

```
Upload Material → Summarize & Simplify → Ask Questions (Text/Voice)
       ↓                                         ↓
   Learn Tab                              Ask Tutor Tab
                                                 ↓
                          Generate Quiz ← Practice Tab
                                                 ↓
                          Get Feedback → Results Tab
                                                 ↓
                          View Metrics → Monitoring Tab
                                                 ↓
                         Train Model  → Fine-tuning Tab
```

---

## Sub-tasks Mapping

| # | Sub-task | Category | Service | Model Used |
|---|----------|----------|---------|------------|
| 1 | Study Material Summarization | NLP | `summarizer.py` | FLAN-T5 / Gemini / OpenAI |
| 2 | Concept Simplification | NLP | `simplifier.py` | FLAN-T5 / Gemini / OpenAI |
| 3 | Grounded Question Answering (RAG) | NLP | `question_answering.py` | Sentence-Transformers + FAISS + LLM |
| 4 | Adaptive Quiz Generation | NLP | `quiz_generator.py` | Fine-tuned FLAN-T5 / Gemini / OpenAI |
| 5 | Answer Evaluation & Feedback | NLP | `answer_evaluator.py` | Pattern matching + LLM |
| 6 | Speech-to-Text | Speech | `speech_to_text.py` | Google Web Speech API |
| 7 | Text-to-Speech | Speech | `text_to_speech.py` | gTTS |

---

## LLMOps Metrics

| # | Metric | Description | Implementation |
|---|--------|-------------|----------------|
| 1 | API Latency | Response time per request | Per-request timing |
| 2 | Success/Failure Rate | Percentage of successful API calls | Request outcome tracking |
| 3 | Token Usage | Total tokens consumed | Cumulative token count |
| 4 | Throughput | Requests per minute | Sliding window calculation |
| 5 | Relevance Score | How well answers match source material | RAG retrieval quality |
| 6 | Confidence Score | Model's self-assessed certainty | Similarity scoring |
| 7 | Prompt Versioning | Track which prompt templates are active | Template tracking per service |
| 8 | System Resources | CPU and Memory usage | psutil monitoring |

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| UI Framework | Gradio |
| REST API | FastAPI + Uvicorn |
| LLM (Local) | Google FLAN-T5-small / FLAN-T5-large |
| LLM (Cloud) | Google Gemini API / OpenAI API |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| Speech-to-Text | SpeechRecognition + Google Web Speech API |
| Text-to-Speech | gTTS |
| Fine-tuning | Hugging Face Transformers + Datasets |
| GPU Support | PyTorch CUDA (RTX 3050 Ti) |
| Metrics | psutil + custom logger |
| Document Parsing | PyPDF2 + python-docx |
| Deployment Ready | Docker / AWS EC2 / AWS Bedrock compatible |

---

## GPU Support

The application supports NVIDIA GPU acceleration for model inference and fine-tuning:

```
GPU: NVIDIA GeForce RTX 3050 Ti (4GB VRAM)
PyTorch: 2.6.0+cu124
CUDA: Available
```

Fine-tuning with GPU takes ~3-5 minutes vs ~20-30 minutes on CPU.

---

## License

This project is developed for academic purposes as part of the M.Tech AI program.
