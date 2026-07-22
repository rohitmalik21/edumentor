"""
EduMentor AI - Answer Evaluation and Feedback Service
Scores student answers, explains errors, and recommends next steps.
"""

from config import Config
from utils.llm_client import get_llm_response
from utils.metrics_logger import metrics

PROMPT_VERSION = "v1.1"


def evaluate_answers(quiz_data: dict, student_answers: dict) -> dict:
    """
    Evaluate student answers against a quiz and provide feedback.

    Args:
        quiz_data: The quiz dictionary with questions and correct answers.
        student_answers: Dictionary mapping question IDs to student's answers.

    Returns:
        Dictionary with evaluation results.
    """
    if not quiz_data or "questions" not in quiz_data:
        return {"feedback": "No quiz data provided.", "score": 0, "total": 0, "error": True}

    if not student_answers:
        return {"feedback": "No student answers provided.", "score": 0, "total": 0, "error": True}

    total = len(quiz_data["questions"])

    # Calculate score first (works without LLM)
    score = _calculate_score(quiz_data, student_answers)
    percentage = round((score / total) * 100, 1) if total > 0 else 0

    # For local models, generate feedback without LLM (faster + reliable)
    if Config.LLM_PROVIDER == "local":
        feedback = _generate_local_feedback(quiz_data, student_answers, score, total)
    else:
        feedback = _generate_cloud_feedback(quiz_data, student_answers, score, total)

    return {
        "feedback": feedback,
        "score": score,
        "total": total,
        "percentage": percentage,
        "error": False,
    }


def _generate_local_feedback(quiz_data: dict, student_answers: dict, score: int, total: int) -> str:
    """Generate detailed feedback without relying on LLM (for local/small models)."""
    percentage = round((score / total) * 100, 1) if total > 0 else 0
    lines = []

    lines.append("### Detailed Feedback\n")

    weak_topics = []

    for q in quiz_data["questions"]:
        qid = str(q.get("id", ""))
        question = q.get("question", "")
        correct = q.get("correct_answer", "")
        explanation = q.get("explanation", "")

        # Get student's answer
        student_ans = str(student_answers.get(qid, student_answers.get(int(qid) if qid.isdigit() else qid, ""))).strip()

        if not student_ans:
            student_ans = "(No answer provided)"

        # Check if correct
        is_correct = _check_answer(correct, student_ans, q.get("type", ""))

        if is_correct:
            lines.append(f"**Q{qid}:** CORRECT ✓")
            lines.append(f"- Your answer: {student_ans}")
            lines.append(f"- Correct answer: {correct}\n")
        else:
            lines.append(f"**Q{qid}:** INCORRECT ✗")
            lines.append(f"- Your answer: {student_ans}")
            lines.append(f"- Correct answer: {correct}")
            if explanation:
                lines.append(f"- Explanation: {explanation}")
            lines.append("")
            weak_topics.append(question[:60])

    lines.append("---")
    lines.append(f"\n### Weak Topics:")
    if weak_topics:
        for topic in weak_topics:
            lines.append(f"- {topic}")
    else:
        lines.append("- None! Great job!")

    lines.append(f"\n### Recommendations:")
    if percentage >= 80:
        lines.append("- Excellent performance! Try a harder difficulty level.")
        lines.append("- Challenge yourself with more complex topics.")
    elif percentage >= 50:
        lines.append("- Good effort! Review the incorrect answers above.")
        lines.append("- Focus on the weak topics and retry at the same level.")
    else:
        lines.append("- Revisit the study material carefully.")
        lines.append("- Use the Summarize and Simplify features in the Learn tab.")
        lines.append("- Start with Easy difficulty and work your way up.")

    return "\n".join(lines)


def _generate_cloud_feedback(quiz_data: dict, student_answers: dict, score: int, total: int) -> str:
    """Generate feedback using cloud LLM."""
    quiz_text = _format_quiz_for_evaluation(quiz_data)
    answers_text = _format_student_answers(student_answers)

    prompt = (
        f"You are EduMentor AI. Evaluate these student answers.\n\n"
        f"Quiz Questions and Correct Answers:\n{quiz_text}\n\n"
        f"Student's Answers:\n{answers_text}\n\n"
        f"Score: {score}/{total}\n\n"
        f"For each question, say if it's CORRECT or INCORRECT, explain why, "
        f"then list weak topics and give revision recommendations."
    )

    metrics.log_prompt_version("answer_evaluation", PROMPT_VERSION, prompt[:200])

    try:
        response = get_llm_response(
            prompt=prompt,
            service_name="answer_evaluation",
            temperature=0.2,
            max_tokens=2000,
        )
        return response
    except Exception as e:
        # Fallback to local feedback on error
        return _generate_local_feedback(quiz_data, student_answers, score, total)


def _check_answer(correct: str, student: str, qtype: str) -> bool:
    """Check if student answer matches correct answer."""
    correct = correct.strip().lower()
    student = student.strip().lower()

    if not student or student == "(no answer provided)":
        return False

    # Direct match
    if correct == student:
        return True

    # Letter match for MCQ (e.g., "A" matches "a" or "a)")
    if len(correct) == 1 and correct.isalpha():
        if student.startswith(correct) or student == correct:
            return True

    # Partial match
    if correct in student or student in correct:
        return True

    # True/False
    if qtype == "true_false":
        if correct[:4] == student[:4]:
            return True

    return False


def _calculate_score(quiz_data: dict, student_answers: dict) -> int:
    """Calculate score by comparing student answers to correct answers."""
    score = 0
    for q in quiz_data["questions"]:
        qid = str(q.get("id", ""))
        correct = q.get("correct_answer", "")
        qtype = q.get("type", "")

        student = str(
            student_answers.get(qid, student_answers.get(int(qid) if qid.isdigit() else qid, ""))
        ).strip()

        if _check_answer(correct, student, qtype):
            score += 1

    return score


def _format_quiz_for_evaluation(quiz_data: dict) -> str:
    """Format quiz questions and answers for the evaluation prompt."""
    lines = []
    for q in quiz_data["questions"]:
        qid = q.get("id", "?")
        question = q.get("question", "")
        correct = q.get("correct_answer", "")
        lines.append(f"Q{qid}: {question} | Correct: {correct}")
    return "\n".join(lines)


def _format_student_answers(student_answers: dict) -> str:
    """Format student answers for the evaluation prompt."""
    lines = []
    for qid, answer in sorted(student_answers.items(), key=lambda x: str(x[0])):
        lines.append(f"Q{qid}: {answer}")
    return "\n".join(lines)
