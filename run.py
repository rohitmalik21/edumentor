"""
EduMentor AI - Application Launcher
Starts both Gradio UI and FastAPI REST API together.

Usage:
    python run.py              → Start both (UI + API)
    python run.py --ui-only    → Start Gradio UI only (port 7860)
    python run.py --api-only   → Start FastAPI API only (port 8000)

Access:
    Gradio UI:     http://localhost:7860
    Swagger API:   http://localhost:8000/docs
    Health Check:  http://localhost:8000/health
    Metrics:       http://localhost:8000/api/v1/metrics
"""

import sys
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config


def start_gradio():
    """Start the Gradio UI server."""
    from app import create_app
    app = create_app()
    app.launch(
        server_name=Config.APP_HOST,
        server_port=Config.APP_PORT,
        share=False,
        show_error=True,
        quiet=True,
    )


def start_fastapi():
    """Start the FastAPI REST API server."""
    import uvicorn
    from api import app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def main():
    args = sys.argv[1:]

    print("=" * 60)
    print("  EduMentor AI - Application Launcher")
    print("=" * 60)
    print(f"  LLM Provider: {Config.LLM_PROVIDER}")
    print(f"  Model: {Config.LOCAL_MODEL if Config.LLM_PROVIDER == 'local' else Config.GEMINI_MODEL}")
    print("=" * 60)

    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    if "--ui-only" in args:
        print("\n  Starting Gradio UI only...")
        print(f"  URL: http://localhost:{Config.APP_PORT}")
        print("=" * 60)
        start_gradio()

    elif "--api-only" in args:
        print("\n  Starting FastAPI REST API only...")
        print(f"  Swagger: http://localhost:8000/docs")
        print("=" * 60)
        start_fastapi()

    else:
        print("\n  Starting both Gradio UI + FastAPI API...")
        print(f"  Gradio UI:   http://localhost:{Config.APP_PORT}")
        print(f"  Swagger API: http://localhost:8000/docs")
        print(f"  Health:      http://localhost:8000/health")
        print(f"  Metrics:     http://localhost:8000/api/v1/metrics")
        print("=" * 60)

        # Start FastAPI in a separate thread
        api_thread = threading.Thread(target=start_fastapi, daemon=True)
        api_thread.start()

        # Start Gradio in main thread (blocking)
        start_gradio()


if __name__ == "__main__":
    main()
