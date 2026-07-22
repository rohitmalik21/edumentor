"""
EduMentor AI - Inference with Fine-tuned FLAN-T5 Model
Use the fine-tuned model for educational question generation.

This module can be used standalone or integrated with the quiz generation service.
"""

import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class FineTunedQuestionGenerator:
    """Generate educational questions using the fine-tuned FLAN-T5 model."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    def load_model(self):
        """Load the fine-tuned model from disk."""
        model_path = Config.FINETUNED_MODEL_DIR

        if not os.path.exists(model_path) or not os.listdir(model_path):
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}. "
                "Run fine_tuning/finetune_flan_t5.py first."
            )

        print(f"Loading fine-tuned model from {model_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self._model.eval()
        self._is_loaded = True
        print("Fine-tuned model loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def generate_question(self, passage: str, num_beams: int = 4) -> str:
        """
        Generate a question from an educational passage.

        Args:
            passage: Educational text passage.
            num_beams: Beam search width for generation.

        Returns:
            Generated question string.
        """
        if not self._is_loaded:
            self.load_model()

        input_text = f"Generate a question from the following educational passage: {passage}"
        input_ids = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).input_ids

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                max_length=128,
                num_beams=num_beams,
                early_stopping=True,
            )

        question = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question

    def generate_multiple_questions(self, passage: str, num_questions: int = 3) -> list[str]:
        """
        Generate multiple diverse questions from a passage.

        Args:
            passage: Educational text passage.
            num_questions: Number of questions to generate.

        Returns:
            List of generated questions.
        """
        if not self._is_loaded:
            self.load_model()

        input_text = f"Generate a question from the following educational passage: {passage}"
        input_ids = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).input_ids

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids,
                max_length=128,
                num_beams=num_questions * 2,
                num_return_sequences=num_questions,
                early_stopping=True,
                diversity_penalty=0.5,
                num_beam_groups=num_questions,
            )

        questions = [
            self._tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        return questions


# Global instance
finetuned_generator = FineTunedQuestionGenerator()


# ============================================
# Standalone Demo
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("EduMentor AI - Fine-tuned Question Generator Demo")
    print("=" * 60)

    test_passages = [
        "The human heart has four chambers: two atria and two ventricles. The right side pumps blood to the lungs, while the left side pumps blood to the rest of the body.",
        "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, gravity gives weight to physical objects and causes the ocean tides.",
        "DNA stands for deoxyribonucleic acid. It is a molecule that carries the genetic instructions used in the growth, development, functioning, and reproduction of all known living organisms.",
    ]

    try:
        generator = FineTunedQuestionGenerator()
        generator.load_model()

        for i, passage in enumerate(test_passages, 1):
            print(f"\n--- Passage {i} ---")
            print(f"Text: {passage[:100]}...")

            question = generator.generate_question(passage)
            print(f"Generated Question: {question}")

            questions = generator.generate_multiple_questions(passage, num_questions=3)
            print("Multiple Questions:")
            for j, q in enumerate(questions, 1):
                print(f"  {j}. {q}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run the fine-tuning script first:")
        print("  python fine_tuning/finetune_flan_t5.py")
