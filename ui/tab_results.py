"""
EduMentor AI - Results Tab
Scores, explanations, weak topics, and revision recommendations.
"""

import gradio as gr

from services.answer_evaluator import evaluate_answers
from ui.tab_practice import quiz_state


def run_evaluation():
    """Evaluate student answers against the quiz."""
    quiz_data = quiz_state.get("quiz_data")
    student_answers = quiz_state.get("student_answers")

    if not quiz_data:
        return "No quiz available. Generate a quiz in the Practice tab first.", "", ""

    if not student_answers:
        return "No answers submitted. Submit your answers in the Practice tab first.", "", ""

    result = evaluate_answers(quiz_data, student_answers)

    if result["error"]:
        return result["feedback"], "", ""

    # Score summary
    score_text = f"## Score: {result['score']}/{result['total']} ({result['percentage']}%)"

    # Performance badge
    pct = result["percentage"]
    if pct >= 80:
        badge = "Excellent! Ready for harder challenges."
    elif pct >= 60:
        badge = "Good effort! Review weak areas and try again."
    elif pct >= 40:
        badge = "Keep practicing. Focus on the topics below."
    else:
        badge = "Revisit the study material and start with easier questions."

    performance = f"{score_text}\n\n**Assessment:** {badge}"

    return performance, result["feedback"], ""


def create_results_tab():
    """Build the Results tab interface."""
    with gr.Tab("Results", id="results"):
        gr.Markdown("## Evaluation & Feedback")
        gr.Markdown(
            "Get detailed feedback on your quiz answers, including explanations "
            "for incorrect responses and personalized revision recommendations."
        )

        evaluate_btn = gr.Button("Evaluate My Answers", variant="primary", size="lg")

        # Score Display
        score_display = gr.Markdown(label="Score")

        gr.Markdown("---")

        # Detailed Feedback
        gr.Markdown("### Detailed Feedback")
        feedback_output = gr.Markdown(label="Feedback")

        # Additional Notes
        notes_output = gr.Textbox(label="Notes", interactive=False, visible=False)

        evaluate_btn.click(
            fn=run_evaluation,
            inputs=[],
            outputs=[score_display, feedback_output, notes_output],
        )

        gr.Markdown("---")

        # Revision Tips
        gr.Markdown("### How to Use Your Results")
        gr.Markdown("""
        1. **Review incorrect answers** — Read the explanations carefully
        2. **Revisit weak topics** — Go back to the Learn tab and focus on those areas
        3. **Simplify concepts** — Use the concept simplifier for topics you don't understand
        4. **Retry the quiz** — Generate a new quiz at the same or different difficulty
        5. **Progress gradually** — Move from Easy → Medium → Hard as you improve
        """)
