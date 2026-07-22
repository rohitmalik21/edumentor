"""
EduMentor AI - Learn Tab
Upload material, summarize, and simplify concepts.
Saves uploaded files permanently and allows loading previous uploads.
"""

import os
import shutil
import time

import gradio as gr

from config import Config
from utils.document_parser import extract_text_from_file
from services.summarizer import summarize_material
from services.simplifier import simplify_concept, LEVELS
from services.question_answering import load_material_for_qa
from services.text_to_speech import synthesize_speech


# Shared state for study material
study_material_state = {"text": ""}


def _get_saved_files() -> list[str]:
    """Get list of previously uploaded files."""
    upload_dir = Config.UPLOAD_DIR
    if not os.path.exists(upload_dir):
        return []
    files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
    # Sort by modification time (newest first)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(upload_dir, f)), reverse=True)
    return files


def _save_uploaded_file(file_path: str) -> str:
    """Save uploaded file to permanent uploads directory."""
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

    filename = os.path.basename(file_path)

    # Add timestamp to avoid overwriting files with same name
    name, ext = os.path.splitext(filename)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    new_filename = f"{name}_{timestamp}{ext}"

    dest_path = os.path.join(Config.UPLOAD_DIR, new_filename)
    shutil.copy2(file_path, dest_path)

    return new_filename


def process_upload(file, pasted_text):
    """Handle file upload or pasted text."""
    text = ""
    saved_name = ""

    if file is not None:
        try:
            # Handle Gradio file (could be path string or file object)
            file_path = file if isinstance(file, str) else file.name
            text = extract_text_from_file(file_path)

            # Save to permanent storage
            saved_name = _save_uploaded_file(file_path)
        except Exception as e:
            return f"Error processing file: {str(e)}", "", gr.update(choices=_get_saved_files())

    if pasted_text and pasted_text.strip():
        text = pasted_text if not text else text + "\n\n" + pasted_text

    if not text:
        return "Please upload a file or paste text.", "", gr.update(choices=_get_saved_files())

    # Store material for other tabs
    study_material_state["text"] = text

    # Index material for QA
    index_status = load_material_for_qa(text)

    word_count = len(text.split())
    status = f"Material loaded successfully."
    if saved_name:
        status += f"\nSaved as: {saved_name}"
    status += f"\nWord count: {word_count:,}"
    status += f"\n{index_status}"

    preview = text[:2000] + ("..." if len(text) > 2000 else "")

    return status, preview, gr.update(choices=_get_saved_files())


def load_previous_file(selected_file):
    """Load a previously uploaded file."""
    if not selected_file:
        return "No file selected.", ""

    file_path = os.path.join(Config.UPLOAD_DIR, selected_file)

    if not os.path.exists(file_path):
        return f"File not found: {selected_file}", ""

    try:
        text = extract_text_from_file(file_path)

        # Store material for other tabs
        study_material_state["text"] = text

        # Index material for QA
        index_status = load_material_for_qa(text)

        word_count = len(text.split())
        status = f"Loaded: {selected_file}\nWord count: {word_count:,}\n{index_status}"
        preview = text[:2000] + ("..." if len(text) > 2000 else "")

        return status, preview
    except Exception as e:
        return f"Error loading file: {str(e)}", ""


def generate_summary():
    """Generate summary of loaded material."""
    if not study_material_state["text"]:
        return "No study material loaded. Please upload or paste material first."

    result = summarize_material(study_material_state["text"])
    return result["summary"]


def explain_concept(topic, level):
    """Simplify a concept at the chosen level."""
    context = study_material_state["text"][:3000] if study_material_state["text"] else ""
    result = simplify_concept(topic, context, level)
    return result["explanation"]


def speak_text(text):
    """Convert text to speech."""
    if not text or not text.strip():
        return None
    result = synthesize_speech(text, "learn_audio.mp3")
    if result["error"]:
        return None
    return result["audio_path"]


def create_learn_tab():
    """Build the Learn tab interface."""
    with gr.Tab("Learn", id="learn"):
        gr.Markdown("## Upload & Process Study Material")
        gr.Markdown("Upload your notes (PDF, DOCX, TXT) or paste text directly. Files are saved for future use.")

        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(
                    label="Upload Study Material",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                    type="filepath",
                )
                pasted_text = gr.Textbox(
                    label="Or Paste Text Here",
                    placeholder="Paste your study notes, textbook content, or any learning material...",
                    lines=6,
                )
                upload_btn = gr.Button("Process Material", variant="primary", size="lg")

            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Status", interactive=False, lines=4)
                material_preview = gr.Textbox(
                    label="Material Preview",
                    interactive=False,
                    lines=8,
                    max_lines=15,
                )

        gr.Markdown("---")

        # Previously Uploaded Files Section
        gr.Markdown("## Previously Uploaded Documents")
        gr.Markdown("Load a document you've uploaded before — no need to upload again.")

        with gr.Row():
            previous_files_dropdown = gr.Dropdown(
                choices=_get_saved_files(),
                label="Select a Previous Document",
                scale=3,
                interactive=True,
            )
            load_prev_btn = gr.Button("Load Selected", variant="secondary", scale=1)
            refresh_btn = gr.Button("Refresh List", variant="secondary", scale=1)

        # Wire up buttons
        upload_btn.click(
            fn=process_upload,
            inputs=[file_upload, pasted_text],
            outputs=[status_output, material_preview, previous_files_dropdown],
        )

        load_prev_btn.click(
            fn=load_previous_file,
            inputs=[previous_files_dropdown],
            outputs=[status_output, material_preview],
        )

        refresh_btn.click(
            fn=lambda: gr.update(choices=_get_saved_files()),
            inputs=[],
            outputs=[previous_files_dropdown],
        )

        gr.Markdown("---")

        # Summarization Section
        gr.Markdown("## Summarize Material")
        summarize_btn = gr.Button("Generate Summary", variant="secondary")
        summary_output = gr.Markdown(label="Summary")
        summary_audio_btn = gr.Button("Read Summary Aloud", size="sm")
        summary_audio = gr.Audio(label="Audio", type="filepath")

        summarize_btn.click(
            fn=generate_summary,
            inputs=[],
            outputs=[summary_output],
        )
        summary_audio_btn.click(
            fn=speak_text,
            inputs=[summary_output],
            outputs=[summary_audio],
        )

        gr.Markdown("---")

        # Concept Simplification Section
        gr.Markdown("## Simplify a Concept")
        with gr.Row():
            concept_input = gr.Textbox(
                label="Topic or Concept",
                placeholder="e.g., Explain neural networks, What is photosynthesis?",
                scale=3,
            )
            level_dropdown = gr.Dropdown(
                choices=LEVELS,
                value="Beginner",
                label="Explanation Level",
                scale=1,
            )
        simplify_btn = gr.Button("Explain", variant="secondary")
        explanation_output = gr.Markdown(label="Explanation")
        explain_audio_btn = gr.Button("Read Explanation Aloud", size="sm")
        explain_audio = gr.Audio(label="Audio", type="filepath")

        simplify_btn.click(
            fn=explain_concept,
            inputs=[concept_input, level_dropdown],
            outputs=[explanation_output],
        )
        explain_audio_btn.click(
            fn=speak_text,
            inputs=[explanation_output],
            outputs=[explain_audio],
        )

    return study_material_state
