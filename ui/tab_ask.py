"""
EduMentor AI - Ask Tutor Tab
Ask questions by text or voice, get grounded answers from study material.
"""

import gradio as gr

from services.question_answering import answer_question
from services.speech_to_text import transcribe_from_gradio
from services.text_to_speech import synthesize_speech


def ask_by_text(question):
    """Answer a text question using RAG."""
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    result = answer_question(question)

    if result["error"]:
        return result["answer"], "", "N/A"

    relevance = f"Relevance Score: {result['relevance_score']}"
    return result["answer"], result["context_used"], relevance


def ask_by_voice(audio):
    """Transcribe voice and answer the question."""
    if audio is None:
        return "No audio detected.", "", "", ""

    # Transcribe
    transcribed = transcribe_from_gradio(audio)

    if not transcribed or transcribed.startswith("[Error"):
        return transcribed, "", "", ""

    # Answer the transcribed question
    answer, context, relevance = ask_by_text(transcribed)
    return transcribed, answer, context, relevance


def speak_answer(text):
    """Convert answer to speech."""
    if not text or not text.strip():
        return None
    result = synthesize_speech(text, "answer_audio.mp3")
    if result["error"]:
        return None
    return result["audio_path"]


def create_ask_tab():
    """Build the Ask Tutor tab interface."""
    with gr.Tab("Ask Tutor", id="ask"):
        gr.Markdown("## Ask Questions About Your Study Material")
        gr.Markdown(
            "Ask any question — the AI will answer using **only** your uploaded material. "
            "You can type or use your microphone."
        )

        # Text-based Q&A
        gr.Markdown("### Type Your Question")
        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is the main function of mitochondria?",
                lines=2,
                scale=4,
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1)

        answer_output = gr.Markdown(label="Answer")

        with gr.Accordion("Supporting Context (from your material)", open=False):
            context_output = gr.Textbox(
                label="Retrieved Passages",
                interactive=False,
                lines=6,
            )

        relevance_output = gr.Textbox(label="Relevance", interactive=False, lines=1)

        with gr.Row():
            answer_audio_btn = gr.Button("Read Answer Aloud", size="sm")
            answer_audio = gr.Audio(label="Audio Response", type="filepath")

        ask_btn.click(
            fn=ask_by_text,
            inputs=[question_input],
            outputs=[answer_output, context_output, relevance_output],
        )
        # Also trigger on Enter
        question_input.submit(
            fn=ask_by_text,
            inputs=[question_input],
            outputs=[answer_output, context_output, relevance_output],
        )
        answer_audio_btn.click(
            fn=speak_answer,
            inputs=[answer_output],
            outputs=[answer_audio],
        )

        gr.Markdown("---")

        # Voice-based Q&A
        gr.Markdown("### Ask by Voice")
        gr.Markdown("Click the microphone to record your question.")

        audio_input = gr.Audio(
            label="Record Your Question",
            sources=["microphone"],
            type="filepath",
        )
        voice_ask_btn = gr.Button("Process Voice Question", variant="secondary")

        voice_transcription = gr.Textbox(label="Transcribed Question", interactive=False)
        voice_answer = gr.Markdown(label="Answer")
        voice_context = gr.Textbox(label="Supporting Context", interactive=False, lines=4)
        voice_relevance = gr.Textbox(label="Relevance", interactive=False, lines=1)

        voice_ask_btn.click(
            fn=ask_by_voice,
            inputs=[audio_input],
            outputs=[voice_transcription, voice_answer, voice_context, voice_relevance],
        )
