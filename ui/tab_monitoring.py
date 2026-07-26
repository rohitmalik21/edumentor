"""
EduMentor AI - Monitoring Tab
LLMOps metrics dashboard with real-time statistics.
"""

import json

import gradio as gr

from utils.metrics_logger import metrics


def refresh_metrics():
    """Get current metrics summary for display."""
    summary = metrics.get_summary()

    # Format overview metrics
    overview = f"""## LLMOps Metrics Dashboard

| Metric | Value |
|--------|-------|
| Total Requests | {summary['total_requests']} |
| Successful | {summary['successful_requests']} |
| Failed | {summary['failed_requests']} |
| Success Rate | {summary['success_rate']} |
| Avg Latency | {summary['avg_latency_seconds']}s |
| P50 Latency | {summary['p50_latency_seconds']}s |
| P95 Latency | {summary['p95_latency_seconds']}s |
| Estimated Tokens Used | {summary['estimated_total_tokens']} |
| Throughput (req/min) | {summary['throughput_per_minute']} |
| Avg Similarity Score | {summary['avg_similarity_score']} |
| CPU Usage | {summary['cpu_usage_percent']}% |
| Memory Usage | {summary['memory_usage_percent']}% |
"""

    # Format service breakdown
    service_lines = ["## Service Breakdown\n"]
    service_lines.append("| Service | Calls | Avg Latency |")
    service_lines.append("|---------|-------|-------------|")
    for svc_name, svc_data in summary["service_breakdown"].items():
        calls = svc_data["calls"]
        avg_lat = f"{svc_data['avg_latency']:.3f}s" if calls > 0 else "N/A"
        service_lines.append(f"| {svc_name.replace('_', ' ').title()} | {calls} | {avg_lat} |")
    service_text = "\n".join(service_lines)

    # Prompt versions
    prompt_lines = ["## Prompt Versions\n"]
    if summary["prompt_versions"]:
        prompt_lines.append("| Service | Version | Last Updated |")
        prompt_lines.append("|---------|---------|--------------|")
        for svc, info in summary["prompt_versions"].items():
            prompt_lines.append(
                f"| {svc.replace('_', ' ').title()} | {info['version']} | {info['updated_at'][:19]} |"
            )
    else:
        prompt_lines.append("No prompts executed yet.")
    prompt_text = "\n".join(prompt_lines)

    # Full JSON for export
    full_json = json.dumps(summary, indent=2, default=str)

    return overview, service_text, prompt_text, full_json


def export_metrics_file():
    """Export metrics to a JSON file."""
    filepath = metrics.export_metrics()
    return f"Metrics exported to: {filepath}"


def create_monitoring_tab():
    """Build the Monitoring tab interface."""
    with gr.Tab("Monitoring", id="monitoring"):
        gr.Markdown("## LLMOps Monitoring Dashboard")
        gr.Markdown(
            "Real-time metrics tracking for API performance, "
            "token usage, and model quality assessment."
        )

        with gr.Row():
            refresh_btn = gr.Button("Refresh Metrics", variant="primary")
            export_btn = gr.Button("Export to JSON", variant="secondary")

        export_status = gr.Textbox(label="Export Status", interactive=False, visible=True)

        # Overview metrics
        overview_output = gr.Markdown(label="Overview")

        # Service breakdown
        service_output = gr.Markdown(label="Service Breakdown")

        # Prompt versioning
        prompt_output = gr.Markdown(label="Prompt Versions")

        gr.Markdown("---")

        # Raw JSON
        with gr.Accordion("Raw Metrics JSON", open=False):
            json_output = gr.Code(label="Full Metrics", language="json")

        refresh_btn.click(
            fn=refresh_metrics,
            inputs=[],
            outputs=[overview_output, service_output, prompt_output, json_output],
        )

        export_btn.click(
            fn=export_metrics_file,
            inputs=[],
            outputs=[export_status],
        )

        gr.Markdown("---")

        # Metrics explanation
        gr.Markdown("""### Metrics Tracked (LLMOps Principles)

| # | Metric | Description | Why It Matters |
|---|--------|-------------|----------------|
| 1 | **API Latency** | Response time per request (avg, p50, p95) | User experience & SLA compliance |
| 2 | **Success/Failure Rate** | Percentage of successful API calls | Reliability monitoring |
| 3 | **Estimated Token Usage** | Approximate tokens consumed (word-based estimate) | Cost management |
| 4 | **Throughput** | Requests per minute | Capacity planning |
| 5 | **Similarity Score** | L2-derived retrieval similarity score (0-1) | RAG retrieval quality |
| 6 | **Prompt Versioning** | Track which prompt templates are active | Reproducibility & A/B testing |
| 7 | **System Resources** | CPU and memory usage | Infrastructure monitoring |
| 8 | **Per-Service Breakdown** | Calls and avg latency per service | Identify bottlenecks |
        """)
