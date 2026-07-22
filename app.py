"""
EduMentor AI - Main Application Entry Point
Launches the Gradio interface with all tabs.

Usage:
    python app.py

The application will be available at http://localhost:7860
"""

import sys
import os

# Ensure the app directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from config import Config
from ui.theme import CUSTOM_CSS, APP_HEADER_HTML
from ui.tab_learn import create_learn_tab
from ui.tab_ask import create_ask_tab
from ui.tab_practice import create_practice_tab
from ui.tab_results import create_results_tab
from ui.tab_monitoring import create_monitoring_tab
from ui.tab_finetune import create_finetune_tab


def create_app() -> gr.Blocks:
    """Build and return the complete Gradio application."""

    with gr.Blocks(
        title="EduMentor AI - Personalized Learning Assistant",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
    ) as app:

        # Header
        gr.HTML(APP_HEADER_HTML)

        # Navigation Tabs
        with gr.Tabs():
            # Tab 1: Learn (Upload, Summarize, Simplify)
            create_learn_tab()

            # Tab 2: Ask Tutor (Text & Voice QA)
            create_ask_tab()

            # Tab 3: Practice (Quiz Generation)
            create_practice_tab()

            # Tab 4: Results (Evaluation & Feedback)
            create_results_tab()

            # Tab 5: Monitoring (LLMOps Dashboard)
            create_monitoring_tab()

            # Tab 6: Fine-tuning (Model Training & Comparison)
            create_finetune_tab()

        # Footer
        gr.HTML("""
        <div class="app-footer">
            <p><strong>EduMentor AI</strong> | Personalized Learning & Assessment Assistant</p>
            <p>Built with Gradio | NLP + Speech Recognition | LLMOps Enabled</p>
            <p style="font-size: 0.8em;">Categories: Natural Language Processing & Speech Recognition</p>
        </div>
        """)

    return app


def main():
    """Launch the application."""
    print("=" * 60)
    print("  EduMentor AI - Personalized Learning Assistant")
    print("=" * 60)
    print(f"  LLM Provider: {Config.LLM_PROVIDER}")
    print(f"  Model: {Config.GEMINI_MODEL if Config.LLM_PROVIDER == 'gemini' else Config.OPENAI_MODEL}")
    print(f"  Embedding: {Config.EMBEDDING_MODEL}")
    print(f"  Host: {Config.APP_HOST}:{Config.APP_PORT}")
    print("=" * 60)

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"\n  ERROR: {e}")
        print("  Please copy .env.example to .env and fill in your API keys.")
        print("  Get a free Gemini key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    # Build and launch
    app = create_app()
    app.launch(
        server_name=Config.APP_HOST,
        server_port=Config.APP_PORT,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
