"""
EduMentor AI - LLMOps Metrics Logger
Tracks latency, token usage, throughput, success/failure rates, and model confidence.
"""

import time
import json
import os
import threading
from datetime import datetime
from collections import deque

import psutil

from config import Config


class MetricsLogger:
    """Centralized metrics collection for LLMOps monitoring."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure one metrics logger across the app."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Request tracking
        self.requests = deque(maxlen=1000)  # Last 1000 requests
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Latency tracking
        self.latencies = deque(maxlen=1000)

        # Token usage
        self.total_tokens_used = 0
        self.token_history = deque(maxlen=1000)

        # Per-service metrics
        self.service_metrics = {
            "summarization": {"calls": 0, "avg_latency": 0, "total_latency": 0},
            "simplification": {"calls": 0, "avg_latency": 0, "total_latency": 0},
            "question_answering": {"calls": 0, "avg_latency": 0, "total_latency": 0},
            "quiz_generation": {"calls": 0, "avg_latency": 0, "total_latency": 0},
            "answer_evaluation": {"calls": 0, "avg_latency": 0, "total_latency": 0},
            "speech_to_text": {"calls": 0, "avg_latency": 0, "total_latency": 0},
            "text_to_speech": {"calls": 0, "avg_latency": 0, "total_latency": 0},
        }

        # Quality metrics
        self.relevance_scores = deque(maxlen=500)
        self.confidence_scores = deque(maxlen=500)

        # Prompt versioning
        self.prompt_versions = {}

    def log_request(
        self,
        service: str,
        latency: float,
        tokens_used: int = 0,
        success: bool = True,
        confidence: float = None,
        relevance: float = None,
    ):
        """
        Log a single API request with all relevant metrics.

        Args:
            service: Name of the service (e.g., 'summarization').
            latency: Response time in seconds.
            tokens_used: Estimated token count.
            success: Whether the request succeeded.
            confidence: Model confidence score (0-1).
            relevance: Answer relevance score (0-1).
        """
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.latencies.append(latency)
        self.total_tokens_used += tokens_used
        self.token_history.append({"tokens": tokens_used, "time": time.time()})

        if confidence is not None:
            self.confidence_scores.append(confidence)
        if relevance is not None:
            self.relevance_scores.append(relevance)

        # Update service-specific metrics
        if service in self.service_metrics:
            svc = self.service_metrics[service]
            svc["calls"] += 1
            svc["total_latency"] += latency
            svc["avg_latency"] = svc["total_latency"] / svc["calls"]

        # Store request record
        self.requests.append({
            "service": service,
            "latency": round(latency, 3),
            "tokens": tokens_used,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })

    def log_prompt_version(self, service: str, version: str, template: str):
        """Track prompt template versions for reproducibility."""
        self.prompt_versions[service] = {
            "version": version,
            "template_preview": template[:200],
            "updated_at": datetime.now().isoformat(),
        }

    def get_summary(self) -> dict:
        """Get a comprehensive metrics summary for the dashboard."""
        avg_latency = (
            sum(self.latencies) / len(self.latencies) if self.latencies else 0
        )
        avg_confidence = (
            sum(self.confidence_scores) / len(self.confidence_scores)
            if self.confidence_scores
            else 0
        )
        avg_relevance = (
            sum(self.relevance_scores) / len(self.relevance_scores)
            if self.relevance_scores
            else 0
        )

        # Calculate throughput (requests per minute over last 60 seconds)
        now = time.time()
        recent_requests = [
            r for r in self.requests
            if (now - datetime.fromisoformat(r["timestamp"]).timestamp()) < 60
        ]
        throughput = len(recent_requests)

        # System resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                f"{(self.successful_requests / self.total_requests * 100):.1f}%"
                if self.total_requests > 0
                else "N/A"
            ),
            "avg_latency_seconds": round(avg_latency, 3),
            "total_tokens_used": self.total_tokens_used,
            "throughput_per_minute": throughput,
            "avg_confidence_score": round(avg_confidence, 3),
            "avg_relevance_score": round(avg_relevance, 3),
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "service_breakdown": self.service_metrics,
            "prompt_versions": self.prompt_versions,
        }

    def get_latency_history(self) -> list[float]:
        """Get latency history for charting."""
        return list(self.latencies)

    def get_token_history(self) -> list[dict]:
        """Get token usage history for charting."""
        return list(self.token_history)

    def export_metrics(self) -> str:
        """Export metrics to a JSON file."""
        filepath = os.path.join(
            Config.METRICS_DIR,
            f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(filepath, "w") as f:
            json.dump(self.get_summary(), f, indent=2, default=str)
        return filepath


# Global metrics instance
metrics = MetricsLogger()
