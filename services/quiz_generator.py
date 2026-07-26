"""
EduMentor AI - Adaptive Quiz Generation Service
Generates MCQs, True/False, and short-answer questions from study material.
Supports cloud LLMs, local models, and fine-tuned model (when available).
"""

import json
import os
import re
import random

from config import Config
from utils.llm_client import get_llm_response
from utils.metrics_logger import metrics

PROMPT_VERSION = "v1.3"

# Cache for fine-tuned model
_finetuned_model = None
_finetuned_tokenizer = None


def _is_finetuned_model_available() -> bool:
    """Check if the fine-tuned model exists."""
    return os.path.exists(os.path.join(Config.FINETUNED_MODEL_DIR, "config.json"))


def _load_finetuned_model():
    """Load fine-tuned model (cached after first load)."""
    global _finetuned_model, _finetuned_tokenizer

    if _finetuned_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        print("  Loading fine-tuned model for quiz generation...")
        _finetuned_tokenizer = AutoTokenizer.from_pretrained(Config.FINETUNED_MODEL_DIR)
        _finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(Config.FINETUNED_MODEL_DIR)
        _finetuned_model.eval()
        print("  Fine-tuned model loaded!")

    return _finetuned_model, _finetuned_tokenizer


def generate_quiz(
    material: str,
    num_questions: int = 5,
    difficulty: str = "Medium",
    question_types: str = "MCQ, True/False, Short Answer",
) -> dict:
    """
    Generate an adaptive quiz from study material.
    Uses fine-tuned model if available, otherwise falls back to base model.

    Args:
        material: Study material text.
        num_questions: Number of questions to generate (1-15).
        difficulty: Easy, Medium, or Hard.
        question_types: Comma-separated question types.

    Returns:
        Dictionary with quiz data or error message.
    """
    if not material or not material.strip():
        return {"quiz": None, "error": "Please load study material first."}

    # Validate inputs
    num_questions = max(1, min(15, num_questions))
    if difficulty not in ["Easy", "Medium", "Hard"]:
        difficulty = "Medium"

    metrics.log_prompt_version("quiz_generation", PROMPT_VERSION, "adaptive")

    # Priority: Fine-tuned model > Cloud API > Local base model
    if _is_finetuned_model_available() and Config.LLM_PROVIDER == "local":
        return _generate_quiz_finetuned(material, num_questions, difficulty)
    elif Config.LLM_PROVIDER == "local":
        return _generate_quiz_local(material, num_questions, difficulty)
    else:
        return _generate_quiz_cloud(material, num_questions, difficulty, question_types)


def _generate_quiz_finetuned(material: str, num_questions: int, difficulty: str) -> dict:
    """Generate quiz using the fine-tuned model (best quality for local)."""
    import torch

    try:
        model, tokenizer = _load_finetuned_model()
    except Exception as e:
        # Fallback to base local model if fine-tuned model fails
        result = _generate_quiz_local(material, num_questions, difficulty)
        if result.get("quiz"):
            result["model_used"] = "FLAN-T5-small (base)"
            result["fine_tuned"] = False
            result["fallback_used"] = True
        return result

    questions = []

    # Split material into key sentences
    sentences = [s.strip() for s in re.split(r'[.\n]', material) if len(s.strip()) > 30]

    if not sentences:
        return {"quiz": None, "error": "Study material is too short to generate questions."}

    # Pick diverse sentences
    if len(sentences) > num_questions:
        selected = random.sample(sentences, num_questions)
    else:
        selected = sentences[:]
        while len(selected) < num_questions:
            selected.append(random.choice(sentences))

    import time as _time
    _quiz_start = _time.time()

    for i, sentence in enumerate(selected, 1):
        input_text = f"Generate a question from the following educational passage: {sentence[:300]}"
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=128, num_beams=4)

        question_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if not question_text.endswith("?"):
            question_text += "?"

        # Generate MCQ options
        options, correct_answer = _generate_options_from_sentence(sentence, i)

        questions.append({
            "id": i,
            "type": "mcq",
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": f"Based on: {sentence[:100]}...",
        })

    _quiz_latency = _time.time() - _quiz_start

    # Log metrics with real measured values
    metrics.log_request(
        service="quiz_generation",
        latency=_quiz_latency,
        tokens_used=sum(len(s.split()) for s in selected),  # Estimated input tokens (word count)
        success=True,
    )

    return {
        "quiz": {
            "quiz_title": "Quiz on Your Study Material (Fine-tuned Model)",
            "difficulty": difficulty,
            "questions": questions,
        },
        "raw_response": "Generated using fine-tuned FLAN-T5 model",
        "model_used": "Fine-tuned FLAN-T5-small (SciQ)",
        "fine_tuned": True,
        "fallback_used": False,
        "error": None,
    }


def _generate_quiz_local(material: str, num_questions: int, difficulty: str) -> dict:
    """Generate quiz using local model (one question at a time for better quality)."""
    questions = []

    # Split material into key sentences for focused prompting
    sentences = [s.strip() for s in re.split(r'[.\n]', material) if len(s.strip()) > 30]

    if not sentences:
        return {"quiz": None, "error": "Study material is too short to generate questions."}

    # Pick diverse sentences to base questions on
    if len(sentences) > num_questions:
        selected = random.sample(sentences, num_questions)
    else:
        selected = sentences[:num_questions]
        # Repeat if we need more
        while len(selected) < num_questions:
            selected.append(random.choice(sentences))

    for i, sentence in enumerate(selected, 1):
        # Very short, focused prompt that fits in 512 tokens
        prompt = f"Generate a question about: {sentence[:200]}"

        try:
            question_text = get_llm_response(
                prompt=prompt,
                service_name="quiz_generation",
                temperature=0.7,
                max_tokens=100,
            )

            # Clean up the response
            question_text = question_text.strip()
            if not question_text.endswith("?"):
                question_text = question_text + "?"

            # Generate MCQ options based on the sentence
            options, correct_answer = _generate_options_from_sentence(sentence, i)

            questions.append({
                "id": i,
                "type": "mcq",
                "question": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": f"Based on: {sentence[:100]}...",
            })

        except Exception:
            # Fallback: create question directly from the sentence
            question_text = _create_question_from_sentence(sentence)
            options, correct_answer = _generate_options_from_sentence(sentence, i)

            questions.append({
                "id": i,
                "type": "mcq",
                "question": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": f"Based on: {sentence[:100]}...",
            })

    return {
        "quiz": {
            "quiz_title": "Quiz on Your Study Material",
            "difficulty": difficulty,
            "questions": questions,
        },
        "raw_response": "Generated using focused local model approach",
        "model_used": "FLAN-T5-small (base)",
        "fine_tuned": False,
        "fallback_used": False,
        "error": None,
    }


def _create_question_from_sentence(sentence: str) -> str:
    """Create a fill-in-the-blank or factual question from a sentence."""
    # Extract key terms (longer words are usually important)
    words = sentence.split()
    important_words = [w for w in words if len(w) > 5 and w[0].isupper()]

    if not important_words:
        important_words = [w for w in words if len(w) > 5]

    if important_words:
        # Create a "what is" question
        key_word = random.choice(important_words)
        return f"According to the study material, what is related to {key_word}?"
    else:
        return f"Which of the following is correct based on the study material?"


def _generate_options_from_sentence(sentence: str, seed: int) -> tuple[list, str]:
    """Generate MCQ options where one is correct (from the sentence)."""
    random.seed(seed)

    # Extract key noun phrases / meaningful fragments from the sentence
    words = sentence.split()

    # Find a meaningful key phrase as the correct answer (not just random words)
    # Look for phrases after "is", "are", "provides", "enables", etc.
    correct_text = ""

    # Try to find a definition or key fact
    definition_patterns = [
        r'(?:is|are|refers to|provides|enables|means)\s+(.{10,80}?)(?:\.|,|$)',
        r'(?:used to|responsible for|designed to)\s+(.{10,80}?)(?:\.|,|$)',
    ]

    for pattern in definition_patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            correct_text = match.group(1).strip()
            break

    # Fallback: take a meaningful chunk
    if not correct_text:
        if len(words) > 6:
            # Take the main content part (skip first 2-3 words which are often subject)
            start = min(3, len(words) // 3)
            end = min(start + 8, len(words))
            correct_text = " ".join(words[start:end])
        else:
            correct_text = " ".join(words[:5])

    # Clean up
    correct_text = correct_text.strip(".,;: ")
    if len(correct_text) < 5:
        correct_text = " ".join(words[:6])

    # Generate plausible but incorrect distractors
    distractors = _generate_distractors(correct_text, sentence)

    # Combine and shuffle
    all_options = [correct_text] + distractors[:3]
    random.shuffle(all_options)

    # Find correct answer letter
    correct_idx = all_options.index(correct_text)
    letters = ["A", "B", "C", "D"]
    correct_letter = letters[correct_idx]

    # Format options
    options = [f"{letters[i]}) {opt}" for i, opt in enumerate(all_options)]

    return options, correct_letter


def _generate_distractors(correct_answer: str, source_sentence: str) -> list[str]:
    """Generate plausible but incorrect distractors for MCQ."""
    distractors = []

    # Strategy 1: Negate or modify the correct answer
    if "network" in correct_answer.lower():
        distractors.append("a physical hardware device in the data center")
    elif "virtual" in correct_answer.lower():
        distractors.append("a physical on-premises infrastructure")
    elif "security" in correct_answer.lower():
        distractors.append("a performance optimization technique")
    elif "subnet" in correct_answer.lower():
        distractors.append("a complete standalone network")
    elif "route" in correct_answer.lower() or "routing" in correct_answer.lower():
        distractors.append("a security encryption protocol")
    elif "connect" in correct_answer.lower():
        distractors.append("an isolation mechanism to block all traffic")
    else:
        distractors.append(f"Not related to {correct_answer.split()[0] if correct_answer.split() else 'this'}")

    # Strategy 2: Domain-specific common wrong answers
    domain_distractors = [
        "a billing and cost management feature",
        "an application deployment service",
        "a database management system",
        "a storage optimization layer",
        "a compute scaling mechanism",
        "a monitoring and alerting service",
        "a user authentication protocol",
        "a container orchestration platform",
    ]

    # Pick distractors that don't overlap with the correct answer
    for d in domain_distractors:
        if len(distractors) >= 3:
            break
        # Make sure distractor doesn't share key words with correct answer
        correct_words = set(correct_answer.lower().split())
        distractor_words = set(d.lower().split())
        overlap = correct_words & distractor_words
        if len(overlap) <= 1:  # Minimal overlap
            distractors.append(d)

    # Ensure we have exactly 3
    while len(distractors) < 3:
        distractors.append("None of the above")

    return distractors[:3]


def _generate_quiz_cloud(material: str, num_questions: int, difficulty: str, question_types: str) -> dict:
    """Generate quiz using cloud API (can handle complex JSON prompts)."""
    if len(material) > 6000:
        material = material[:6000]

    prompt = f"""You are EduMentor AI. Generate exactly {num_questions} quiz questions based ONLY on this study material.
Difficulty: {difficulty}. Types: {question_types}.

Study Material:
{material}

Return JSON format:
{{"quiz_title": "Quiz", "difficulty": "{difficulty}", "questions": [{{"id": 1, "type": "mcq", "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_answer": "A", "explanation": "..."}}]}}

Generate quiz now:"""

    try:
        response = get_llm_response(
            prompt=prompt,
            service_name="quiz_generation",
            temperature=0.6,
            max_tokens=3000,
        )

        # Try JSON parsing
        quiz_data = _parse_json_response(response)
        if quiz_data:
            return {"quiz": quiz_data, "raw_response": response, "error": None}

        # Fallback to text parsing
        quiz_data = _parse_text_response(response, num_questions, difficulty)
        if quiz_data:
            return {"quiz": quiz_data, "raw_response": response, "error": None}

        return {"quiz": None, "raw_response": response, "error": "Failed to parse quiz. See raw response."}

    except Exception as e:
        return {"quiz": None, "error": f"Error generating quiz: {str(e)}"}


def _parse_json_response(response: str) -> dict | None:
    """Try to extract JSON quiz data from response."""
    try:
        json_match = re.search(r'\{[\s\S]*"questions"[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(response)
    except (json.JSONDecodeError, AttributeError):
        return None


def _parse_text_response(response: str, num_questions: int, difficulty: str) -> dict | None:
    """Parse text-based quiz output."""
    lines = response.strip().split("\n")
    q_lines = [l.strip() for l in lines if l.strip() and "?" in l]

    if q_lines:
        questions = []
        for i, q in enumerate(q_lines[:num_questions], 1):
            q = re.sub(r'^[\d]+[:\.\)]\s*', '', q)
            questions.append({
                "id": i,
                "type": "short_answer",
                "question": q,
                "options": [],
                "correct_answer": "Refer to study material",
                "explanation": "Check your notes for the answer.",
            })
        return {
            "quiz_title": "Quiz",
            "difficulty": difficulty,
            "questions": questions,
        }
    return None


def format_quiz_for_display(quiz_data: dict) -> str:
    """Format quiz data into readable text for the UI."""
    if not quiz_data or "questions" not in quiz_data:
        return "No quiz data available."

    output = []
    title = quiz_data.get("quiz_title", "Quiz")
    difficulty = quiz_data.get("difficulty", "")
    output.append(f"# {title}")
    output.append(f"**Difficulty:** {difficulty}\n")

    for q in quiz_data["questions"]:
        qid = q.get("id", "?")
        qtype = q.get("type", "").upper()
        question = q.get("question", "")
        options = q.get("options", [])

        output.append(f"---\n**Q{qid}** [{qtype}]: {question}")

        if options:
            for opt in options:
                output.append(f"  - {opt}")
        output.append("")

    return "\n".join(output)


def get_quiz_answers(quiz_data: dict) -> str:
    """Get the answer key for a quiz."""
    if not quiz_data or "questions" not in quiz_data:
        return "No quiz data available."

    output = ["# Answer Key\n"]
    for q in quiz_data["questions"]:
        qid = q.get("id", "?")
        correct = q.get("correct_answer", "N/A")
        explanation = q.get("explanation", "")
        output.append(f"**Q{qid}:** {correct}")
        if explanation:
            output.append(f"  *Explanation:* {explanation}\n")
        else:
            output.append("")

    return "\n".join(output)
