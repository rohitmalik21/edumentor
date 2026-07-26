"""
EduMentor AI - Grounded Question Answering Service (RAG)
Answers questions using ONLY the uploaded study material with relevance scoring.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from utils.llm_client import get_llm_response
from utils.metrics_logger import metrics
from utils.document_parser import chunk_text
from config import Config

PROMPT_VERSION = "v1.0"
QA_PROMPT = """You are EduMentor AI, a grounded question-answering assistant.

## Important Rules:
1. Answer ONLY using the provided context passages.
2. If the answer cannot be found in the context, say: "The answer was not found in the supplied material."
3. Quote or reference the relevant passage when possible.
4. Be concise but complete.

## Context Passages:
{context}

## Student's Question:
{question}

## Instructions:
Provide your answer in this format:

**Answer:** (Your answer based on the context)

**Supporting Evidence:** (Quote the relevant part of the context that supports your answer)

**Confidence:** (Rate your confidence: High / Medium / Low based on how well the context supports the answer)"""


class RAGEngine:
    """Retrieval-Augmented Generation engine using FAISS and sentence embeddings."""

    def __init__(self):
        self._model = None
        self._index = None
        self._chunks = []
        self._is_loaded = False

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(Config.EMBEDDING_MODEL)
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def build_index(self, text: str):
        """
        Build a FAISS vector index from the study material.

        Args:
            text: Full study material text.
        """
        self._chunks = chunk_text(text, chunk_size=500, overlap=50)

        if not self._chunks:
            self._is_loaded = False
            return

        # Generate embeddings
        embeddings = self.model.encode(self._chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        # Build FAISS index
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings)
        self._is_loaded = True

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The student's question.
            top_k: Number of passages to retrieve.

        Returns:
            List of dicts with 'text' and 'score' keys.
        """
        if not self._is_loaded:
            return []

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype="float32")

        distances, indices = self._index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._chunks):
                results.append({
                    "text": self._chunks[idx],
                    "score": float(1 / (1 + distances[0][i])),  # Convert distance to similarity
                })
        return results


# Global RAG engine instance
rag_engine = RAGEngine()


def load_material_for_qa(text: str) -> str:
    """
    Process and index study material for question answering.

    Args:
        text: Study material text.

    Returns:
        Status message.
    """
    if not text or not text.strip():
        return "No text provided to index."

    rag_engine.build_index(text)
    chunk_count = len(rag_engine._chunks)
    return f"Study material indexed successfully. Created {chunk_count} searchable passages."


def answer_question(question: str) -> dict:
    """
    Answer a question using only the indexed study material (RAG).

    Args:
        question: The student's question.

    Returns:
        Dictionary with answer, supporting context, and relevance score.
    """
    if not question or not question.strip():
        return {
            "answer": "Please ask a question.",
            "context_used": "",
            "relevance_score": 0,
            "error": True,
        }

    if not rag_engine.is_loaded:
        return {
            "answer": "No study material has been loaded. Please upload material first.",
            "context_used": "",
            "relevance_score": 0,
            "error": True,
        }

    # Retrieve relevant passages
    retrieved = rag_engine.retrieve(question, top_k=3)

    if not retrieved:
        return {
            "answer": "The answer was not found in the supplied material.",
            "context_used": "",
            "relevance_score": 0,
            "error": False,
        }

    # Build context from retrieved passages
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        context_parts.append(f"[Passage {i}] (Relevance: {r['score']:.2f})\n{r['text']}")
    context = "\n\n".join(context_parts)

    # Average relevance score
    avg_relevance = sum(r["score"] for r in retrieved) / len(retrieved)

    metrics.log_prompt_version("question_answering", PROMPT_VERSION, QA_PROMPT)

    # Measure actual latency for the QA process
    import time as _time
    _qa_start = _time.time()

    # For local models: combine LLM short answer with retrieved passages for complete response
    if Config.LLM_PROVIDER == "local":
        answer = _build_local_answer(question, retrieved, avg_relevance)
    else:
        # Cloud models can generate detailed answers from context
        prompt = QA_PROMPT.format(context=context, question=question)
        try:
            answer = get_llm_response(
                prompt=prompt,
                service_name="question_answering",
                temperature=0.2,
                max_tokens=1000,
            )
        except Exception as e:
            answer = f"Error: {str(e)}"

    _qa_latency = _time.time() - _qa_start

    # Log metrics with real measured values
    metrics.log_request(
        service="question_answering",
        latency=_qa_latency,
        tokens_used=len(context.split()),  # Approximate tokens from context length
        success=True,
        relevance=avg_relevance,  # Cosine similarity score (0-1), not a calibrated probability
    )

    return {
        "answer": answer,
        "context_used": context,
        "relevance_score": round(avg_relevance, 3),
        "error": False,
    }


def _build_local_answer(question: str, retrieved: list[dict], relevance: float) -> str:
    """
    Build a comprehensive answer for local models.
    Since FLAN-T5-small can't generate good answers, we present
    the retrieved passages directly as the answer in a clear format.
    """
    output = []

    # Header
    output.append(f"**Based on your study material, here is what was found for:** *{question}*\n")

    # Show the relevant evidence from the material directly as the answer
    for i, r in enumerate(retrieved, 1):
        score_pct = int(r['score'] * 100)
        passage_text = r['text'].strip()

        if i == 1:
            output.append(f"**Most Relevant Answer** (Match: {score_pct}%):\n")
            output.append(f"> {passage_text}\n")
        else:
            output.append(f"**Additional Context {i}** (Match: {score_pct}%):\n")
            output.append(f"> {passage_text}\n")

    # Confidence indicator
    if relevance >= 0.7:
        output.append("\n**Confidence:** High — answer is well-supported by the material.")
    elif relevance >= 0.5:
        output.append("\n**Confidence:** Medium — relevant passages found.")
    else:
        output.append("\n**Confidence:** Low — limited supporting evidence found.")

    return "\n".join(output)
