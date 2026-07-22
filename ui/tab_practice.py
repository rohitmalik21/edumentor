"""
EduMentor AI - Practice Tab
Generate adaptive quizzes and submit answers.
"""

import gradio as gr

from services.quiz_generator import generate_quiz, format_quiz_for_display, get_quiz_answers
from ui.tab_learn import study_material_state


# Store current quiz state
quiz_state = {"quiz_data": None}


def create_quiz(num_questions, difficulty, question_types):
    """Generate a quiz from the study material."""
    material = study_material_state.get("text", "")

    if not material:
        return "Please upload study material in the Learn tab first.", ""

    # Convert checkboxes to comma-separated string
    if isinstance(question_types, list):
        question_types = ", ".join(question_types)

    result = generate_quiz(
        material=material,
        num_questions=int(num_questions),
        difficulty=difficulty,
        question_types=question_types,
    )

    if result["error"]:
        return f"Error: {result['error']}", result.get("raw_response", "")

    quiz_state["quiz_data"] = result["quiz"]
    display = format_quiz_for_display(result["quiz"])
    return display, ""


def show_answers():
    """Show the answer key for the current quiz."""
    if not quiz_state["quiz_data"]:
        return "No quiz generated yet. Generate a quiz first."
    return get_quiz_answers(quiz_state["quiz_data"])


def get_quiz_data():
    """Return current quiz data for the Results tab."""
    return quiz_state.get("quiz_data")


def create_practice_tab():
    """Build the Practice tab interface."""
    with gr.Tab("Practice", id="practice"):
        gr.Markdown("## Generate Adaptive Quiz")
        gr.Markdown(
            "Create a quiz based on your study material. "
            "Choose the number of questions, difficulty, and question types."
        )

        with gr.Row():
            num_questions = gr.Slider(
                minimum=3,
                maximum=15,
                value=5,
                step=1,
                label="Number of Questions",
            )
            difficulty = gr.Radio(
                choices=["Easy", "Medium", "Hard"],
                value="Medium",
                label="Difficulty Level",
            )

        question_types = gr.CheckboxGroup(
            choices=["MCQ", "True/False", "Short Answer"],
            value=["MCQ", "True/False"],
            label="Question Types",
        )

        generate_btn = gr.Button("Generate Quiz", variant="primary", size="lg")

        quiz_display = gr.Markdown(label="Your Quiz")

        with gr.Accordion("Raw Response (if parsing failed)", open=False):
            raw_response = gr.Textbox(label="Raw LLM Response", interactive=False, lines=10)

        generate_btn.click(
            fn=create_quiz,
            inputs=[num_questions, difficulty, question_types],
            outputs=[quiz_display, raw_response],
        )

        gr.Markdown("---")

        # Answer Submission
        gr.Markdown("## Submit Your Answers")
        gr.Markdown(
            "Enter your answers below (one per line in format: `1: A` or `1: Your answer`)"
        )

        answers_input = gr.Textbox(
            label="Your Answers",
            placeholder="1: A\n2: True\n3: Mitochondria is the powerhouse of the cell\n4: B\n5: False",
            lines=8,
        )

        submit_btn = gr.Button("Submit for Evaluation", variant="primary")
        submit_status = gr.Textbox(label="Status", interactive=False)

        def submit_answers(answers_text):
            """Parse answers and store for evaluation."""
            if not answers_text or not answers_text.strip():
                return "Please enter your answers."

            if not quiz_state["quiz_data"]:
                return "No quiz to evaluate. Generate a quiz first."

            # Parse answers (format: "1: A" or "1: answer text")
            parsed = {}
            for line in answers_text.strip().split("\n"):
                line = line.strip()
                if ":" in line:
                    parts = line.split(":", 1)
                    qid = parts[0].strip().replace("Q", "").replace("q", "")
                    answer = parts[1].strip()
                    parsed[qid] = answer

            if not parsed:
                return "Could not parse answers. Use format: 1: A"

            quiz_state["student_answers"] = parsed
            return f"Answers submitted ({len(parsed)} responses). Go to the Results tab for evaluation."

        submit_btn.click(
            fn=submit_answers,
            inputs=[answers_input],
            outputs=[submit_status],
        )

        gr.Markdown("---")

        # Answer Key
        gr.Markdown("## Answer Key")
        show_answers_btn = gr.Button("Show Answer Key", variant="secondary")
        answer_key_output = gr.Markdown(label="Answer Key")

        show_answers_btn.click(
            fn=show_answers,
            inputs=[],
            outputs=[answer_key_output],
        )

    return quiz_state
