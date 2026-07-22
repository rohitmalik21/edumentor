# EduMentor AI - Personalized Learning & Assessment Assistant

An API-based AI application that transforms student-provided study material into an interactive learning experience using **Natural Language Processing** and **Speech Recognition**.

## Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Education |
| **AI Categories** | NLP + Speech Recognition |
| **Sub-tasks** | 7 (Summarization, Simplification, Grounded QA, Quiz Generation, Answer Evaluation, STT, TTS) |
| **LLM Provider** | Google Gemini (default) / OpenAI (optional) |
| **Fine-tuning** | FLAN-T5-small on SciQ dataset |
| **UI Framework** | Gradio |
| **LLMOps Metrics** | 8 tracked (Latency, Success Rate, Tokens, Throughput, Relevance, Confidence, Prompt Versioning, System Resources) |

## Architecture

```
edumentor/
├── app.py                    # Main entry point
├── config.py                 # Configuration & environment
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── services/                 # AI service modules
│   ├── summarizer.py         # Study material summarization
│   ├── simplifier.py         # Concept simplification by level
│   ├── question_answering.py # RAG-based grounded QA (FAISS)
│   ├── quiz_generator.py     # Adaptive quiz generation
│   ├── answer_evaluator.py   # Answer scoring & feedback
│   ├── speech_to_text.py     # Whisper-based STT
│   └── text_to_speech.py     # gTTS-based TTS
├── utils/                    # Shared utilities
│   ├── document_parser.py    # PDF/DOCX/TXT extraction
│   ├── llm_client.py         # Unified LLM API client
│   └── metrics_logger.py     # LLMOps metrics tracking
├── ui/                       # Gradio interface
│   ├── theme.py              # Custom styling
│   ├── tab_learn.py          # Learn tab (Upload, Summarize, Simplify)
│   ├── tab_ask.py            # Ask Tutor tab (Text & Voice QA)
│   ├── tab_practice.py       # Practice tab (Quiz)
│   ├── tab_results.py        # Results tab (Evaluation)
│   └── tab_monitoring.py     # Monitoring tab (LLMOps Dashboard)
└── fine_tuning/              # Model fine-tuning
    ├── finetune_flan_t5.py   # Training script
    └── inference_finetuned.py # Inference with fine-tuned model
```

## Setup & Installation

### 1. Clone and navigate to the project

```bash
cd edumentor
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Edit `.env` and add your API key:
- **Google Gemini** (free): Get key at https://aistudio.google.com/apikey
- **OpenAI** (paid): Get key at https://platform.openai.com/api-keys

### 5. Run the application

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Fine-tuning (Required for Assignment)

```bash
python fine_tuning/finetune_flan_t5.py
```

This will:
- Download the SciQ dataset
- Fine-tune FLAN-T5-small for question generation
- Save the model to `finetuned_model/`
- Generate a comparison report (base vs fine-tuned)

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
```

## Sub-tasks Mapping

| # | Sub-task | Category | Service |
|---|----------|----------|---------|
| 1 | Study Material Summarization | NLP | `summarizer.py` |
| 2 | Concept Simplification | NLP | `simplifier.py` |
| 3 | Grounded Question Answering (RAG) | NLP | `question_answering.py` |
| 4 | Adaptive Quiz Generation | NLP | `quiz_generator.py` |
| 5 | Answer Evaluation & Feedback | NLP | `answer_evaluator.py` |
| 6 | Speech-to-Text | Speech | `speech_to_text.py` |
| 7 | Text-to-Speech | Speech | `text_to_speech.py` |

## LLMOps Metrics

| # | Metric | Implementation |
|---|--------|----------------|
| 1 | API Latency | Per-request timing |
| 2 | Success/Failure Rate | Request outcome tracking |
| 3 | Token Usage | Cumulative token count |
| 4 | Throughput | Requests per minute |
| 5 | Relevance Score | RAG retrieval quality |
| 6 | Confidence Score | Model certainty |
| 7 | Prompt Versioning | Template tracking |
| 8 | System Resources | CPU & Memory |

## Technologies Used

- **Google Gemini API** - Primary LLM for generation tasks
- **Sentence Transformers** - Text embeddings for RAG
- **FAISS** - Vector similarity search
- **OpenAI Whisper** - Speech-to-text
- **gTTS** - Text-to-speech
- **Hugging Face Transformers** - Fine-tuning FLAN-T5
- **Gradio** - Interactive web UI
- **psutil** - System metrics
