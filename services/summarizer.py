"""
EduMentor AI - Study Material Summarization Service
Converts lengthy notes into concise summaries with key concepts and terms.
For local models, uses extractive summarization + LLM enhancement.
"""

import re

from config import Config
from utils.llm_client import get_llm_response
from utils.metrics_logger import metrics

PROMPT_VERSION = "v1.2"

# Prompt for cloud APIs (Gemini/OpenAI)
CLOUD_SUMMARIZATION_PROMPT = """You are EduMentor AI, an expert educational summarizer.

Given the following study material, create a comprehensive yet concise learning summary.

## Study Material:
{text}

## Instructions:
Provide your response in the following structured format:

### Summary
(A clear, concise summary of the main content in 3-5 paragraphs)

### Key Concepts
(List the most important concepts as bullet points)

### Key Terms
(List important terminology with brief definitions)

### Revision Points
(List the most important points a student should remember for exams)

Keep the language clear and student-friendly. Focus on accuracy and completeness."""


def summarize_material(text: str) -> dict:
    """
    Summarize study material into structured learning notes.

    Args:
        text: The study material text to summarize.

    Returns:
        Dictionary with 'summary' key containing the formatted summary.
    """
    if not text or not text.strip():
        return {"summary": "Please provide study material to summarize.", "error": True}

    metrics.log_prompt_version("summarization", PROMPT_VERSION, "adaptive")

    if Config.LLM_PROVIDER == "local":
        return _summarize_local(text)
    else:
        return _summarize_cloud(text)


def _summarize_cloud(text: str) -> dict:
    """Summarize using cloud API (Gemini/OpenAI)."""
    if len(text) > 12000:
        text = text[:12000] + "\n\n[... content truncated for processing ...]"

    prompt = CLOUD_SUMMARIZATION_PROMPT.format(text=text)

    try:
        response = get_llm_response(
            prompt=prompt,
            service_name="summarization",
            temperature=0.3,
            max_tokens=2048,
        )
        return {"summary": response, "error": False}
    except Exception as e:
        return {"summary": f"Error generating summary: {str(e)}", "error": True}


def _summarize_local(text: str) -> dict:
    """
    Summarize using local FLAN-T5 model combined with extractive approach.
    FLAN-T5-small is too weak for long-form summarization, so we:
    1. Extract the most important sentences (extractive)
    2. Use FLAN-T5 to generate a one-line summary per section
    3. Combine into a structured output
    """
    # Step 1: Extract document structure (headings and key paragraphs)
    sections = _extract_document_structure(text)

    # Step 2: For each section, generate a brief summary using LLM
    section_summaries = []
    for section_title, section_content in sections[:10]:  # Max 10 sections
        if len(section_content) < 30:
            continue

        # Use LLM for a brief summary of each section
        prompt = f"Summarize in one sentence: {section_content[:400]}"
        try:
            response = get_llm_response(
                prompt=prompt,
                service_name="summarization",
                temperature=0.3,
                max_tokens=100,
            )
            if response and response.strip():
                section_summaries.append((section_title, response.strip()))
        except Exception:
            # Fallback: use first sentence of the section
            first_sentence = section_content.split(". ")[0]
            if first_sentence:
                section_summaries.append((section_title, first_sentence + "."))

    # Step 3: Extract key concepts and terms
    key_concepts = _extract_key_concepts(text)
    key_terms = _extract_key_terms(text)
    revision_points = _extract_revision_points(text)

    # Step 4: Format everything into structured output
    output = _format_structured_summary(section_summaries, key_concepts, key_terms, revision_points, text)

    return {"summary": output, "error": False}


def _extract_document_structure(text: str) -> list[tuple[str, str]]:
    """
    Extract sections from the document based on headings/numbering patterns.
    Returns list of (title, content) tuples.
    """
    sections = []

    # Split by numbered headings (1. Title, 2.1 Title, etc.)
    parts = re.split(r'\n(?=\d+\.[\d.]*\s+[A-Z])', text)

    if len(parts) > 3:
        for part in parts:
            lines = part.strip().split("\n")
            if lines:
                title = lines[0].strip()[:100]
                content = " ".join(lines[1:]).strip()
                if content and len(content) > 50:
                    sections.append((title, content))
    else:
        # Try splitting by paragraphs with capitalized first words
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
        for para in paragraphs[:15]:
            first_line = para.split("\n")[0][:100]
            sections.append((first_line, para))

    return sections if sections else [("Document", text[:3000])]


def _extract_key_concepts(text: str) -> list[str]:
    """Extract key concepts from the text using pattern matching."""
    concepts = []

    # Look for definition patterns: "X is..." or "X refers to..."
    patterns = [
        r'([A-Z][^.]{5,60})\s+(?:is|are|refers to|provides|enables|allows)\s+([^.]{20,150})\.',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for subject, definition in matches[:8]:
            concept = f"**{subject.strip()}**: {definition.strip()}"
            if concept not in concepts:
                concepts.append(concept)

    # Also look for bullet points or listed items
    bullet_matches = re.findall(r'(?:^|\n)\s*[-•*]\s*(.{20,150})', text)
    for item in bullet_matches[:5]:
        cleaned = item.strip()
        if cleaned and cleaned not in concepts:
            concepts.append(cleaned)

    return concepts[:10]


def _extract_key_terms(text: str) -> list[tuple[str, int]]:
    """Extract important terms based on frequency and capitalization."""
    # Find multi-word capitalized terms (likely technical terms)
    multi_word = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
    # Find acronyms
    acronyms = re.findall(r'\b([A-Z]{2,6})\b', text)

    # Count frequencies
    term_freq = {}
    for term in multi_word:
        if len(term) > 5 and term not in ["The", "This", "That", "These", "Those"]:
            term_freq[term] = term_freq.get(term, 0) + 1

    for term in acronyms:
        if len(term) >= 2 and term not in ["AWS", "VPC", "II", "IP"]:  # Keep AWS/VPC but skip generic
            term_freq[term] = term_freq.get(term, 0) + 1

    # Also add AWS/VPC explicitly if present
    for important in ["AWS", "VPC", "EC2", "S3", "IAM", "CIDR", "NAT", "API"]:
        if important in text:
            count = text.count(important)
            if count >= 2:
                term_freq[important] = count

    # Sort by frequency
    sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_terms[:15]


def _extract_revision_points(text: str) -> list[str]:
    """Extract key points suitable for exam revision."""
    points = []

    # Look for "important", "must", "key", "note" sentences
    sentences = re.split(r'[.!]\s+', text)
    important_keywords = ['must', 'important', 'key', 'required', 'essential',
                          'always', 'never', 'recommended', 'best practice',
                          'default', 'note that', 'remember']

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 30 and len(sentence) < 200:
            for keyword in important_keywords:
                if keyword in sentence.lower():
                    points.append(sentence)
                    break

    # Also include sentences with colons (often definitions)
    for sentence in sentences:
        if ": " in sentence and len(sentence) > 30 and len(sentence) < 200:
            if sentence not in points:
                points.append(sentence)

    return points[:10]


def _format_structured_summary(
    section_summaries: list[tuple[str, str]],
    key_concepts: list[str],
    key_terms: list[tuple[str, int]],
    revision_points: list[str],
    original_text: str,
) -> str:
    """Format all extracted information into a structured summary."""
    output = []

    # Main Summary
    output.append("### Summary\n")
    if section_summaries:
        for title, summary in section_summaries:
            # Clean title
            clean_title = title[:80].strip()
            if clean_title:
                output.append(f"**{clean_title}**")
            output.append(f"{summary}\n")
    else:
        output.append("Unable to generate detailed summary. See key concepts below.\n")

    # Key Concepts
    if key_concepts:
        output.append("\n### Key Concepts\n")
        for concept in key_concepts:
            output.append(f"- {concept}")

    # Key Terms
    if key_terms:
        output.append("\n\n### Key Terms\n")
        output.append("| Term | Frequency |")
        output.append("|------|-----------|")
        for term, freq in key_terms:
            output.append(f"| {term} | {freq} mentions |")

    # Revision Points
    if revision_points:
        output.append("\n\n### Revision Points\n")
        for i, point in enumerate(revision_points, 1):
            output.append(f"{i}. {point}")

    # Document stats
    word_count = len(original_text.split())
    output.append(f"\n\n---\n*Document: {word_count:,} words analyzed*")

    return "\n".join(output)
