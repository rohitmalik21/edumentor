"""
EduMentor AI - Concept Simplification Service
Explains difficult content at the student's chosen comprehension level.
"""

from utils.llm_client import get_llm_response
from utils.metrics_logger import metrics

PROMPT_VERSION = "v1.0"
SIMPLIFICATION_PROMPT = """You are EduMentor AI, an expert at explaining complex topics simply.

## Topic/Concept to Explain:
{topic}

## Context from Study Material:
{context}

## Target Level: {level}

## Level Guidelines:
- Beginner: Use everyday language, simple analogies, no jargon. Assume no prior knowledge.
- School Student: Use basic academic language, relatable examples from daily life.
- Undergraduate: Use proper terminology with explanations, include relevant details.
- Advanced: Full technical depth, connections to related concepts, edge cases.

## Instructions:
1. Explain the concept clearly at the specified level.
2. Use at least one analogy or real-world example.
3. Highlight what makes this concept important.
4. If applicable, mention common misconceptions.

Provide a well-structured explanation that helps the student truly understand."""


LEVELS = ["Beginner", "School Student", "Undergraduate", "Advanced"]


def simplify_concept(topic: str, context: str = "", level: str = "Beginner") -> dict:
    """
    Explain a concept at the student's chosen comprehension level.

    Args:
        topic: The concept or topic to explain.
        context: Optional context from study material.
        level: Comprehension level (Beginner/School Student/Undergraduate/Advanced).

    Returns:
        Dictionary with 'explanation' key containing the simplified explanation.
    """
    if not topic or not topic.strip():
        return {"explanation": "Please enter a topic or concept to explain.", "error": True}

    if level not in LEVELS:
        level = "Beginner"

    prompt = SIMPLIFICATION_PROMPT.format(
        topic=topic,
        context=context if context else "No additional context provided.",
        level=level,
    )

    metrics.log_prompt_version("simplification", PROMPT_VERSION, SIMPLIFICATION_PROMPT)

    try:
        response = get_llm_response(
            prompt=prompt,
            service_name="simplification",
            temperature=0.5,
            max_tokens=1500,
        )
        return {"explanation": response, "error": False}
    except Exception as e:
        return {"explanation": f"Error generating explanation: {str(e)}", "error": True}
