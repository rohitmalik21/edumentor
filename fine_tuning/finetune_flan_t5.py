"""
EduMentor AI - Fine-tuning FLAN-T5-small for Educational Question Generation
Fine-tunes on the SciQ dataset to generate questions from educational passages.

Usage:
    python fine_tuning/finetune_flan_t5.py

This script will:
1. Load the SciQ dataset from Hugging Face
2. Preprocess it for question generation (passage → question)
3. Fine-tune FLAN-T5-small
4. Evaluate with ROUGE/BLEU metrics
5. Save the fine-tuned model
6. Show before/after comparison examples
"""

import os
import sys
import json
from datetime import datetime

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ============================================
# Configuration
# ============================================
MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = Config.FINETUNED_MODEL_DIR
DATASET_NAME = "allenai/sciq"

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
WARMUP_STEPS = 100

# Dataset split
TRAIN_SIZE = 5000  # Use subset for faster training
EVAL_SIZE = 500


# ============================================
# Data Preparation
# ============================================
def load_and_prepare_dataset():
    """Load SciQ dataset and format for question generation."""
    print("Loading SciQ dataset...")
    dataset = load_dataset(DATASET_NAME)

    print(f"  Train size: {len(dataset['train'])}")
    print(f"  Validation size: {len(dataset['validation'])}")
    print(f"  Test size: {len(dataset['test'])}")

    # Use subsets for manageable training time
    train_dataset = dataset["train"].select(range(min(TRAIN_SIZE, len(dataset["train"]))))
    eval_dataset = dataset["validation"].select(range(min(EVAL_SIZE, len(dataset["validation"]))))

    print(f"\n  Using {len(train_dataset)} train / {len(eval_dataset)} eval samples")
    return train_dataset, eval_dataset


def preprocess_function(examples, tokenizer):
    """
    Convert SciQ examples into question generation format.

    Input format:  "Generate a question from the following passage: {support}"
    Target format: "{question}"
    """
    inputs = []
    targets = []

    for support, question in zip(examples["support"], examples["question"]):
        # Skip empty entries
        if not support or not support.strip() or not question or not question.strip():
            inputs.append("Generate a question from the following passage: No passage available.")
            targets.append("What is the main concept?")
            continue

        input_text = f"Generate a question from the following educational passage: {support.strip()}"
        target_text = question.strip()

        inputs.append(input_text)
        targets.append(target_text)

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Replace padding token IDs with -100 so they're ignored in loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ============================================
# Metrics Computation
# ============================================
def compute_metrics(eval_preds, tokenizer, rouge_metric, bleu_metric):
    """Compute ROUGE and BLEU scores for evaluation."""
    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels with pad token for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute ROUGE
    rouge_results = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    # Compute BLEU (requires tokenized inputs)
    bleu_preds = [pred.split() for pred in decoded_preds]
    bleu_refs = [[label.split()] for label in decoded_labels]

    try:
        bleu_results = bleu_metric.compute(predictions=bleu_preds, references=bleu_refs)
        bleu_score = bleu_results["bleu"]
    except Exception:
        bleu_score = 0.0

    return {
        "rouge1": round(rouge_results["rouge1"], 4),
        "rouge2": round(rouge_results["rouge2"], 4),
        "rougeL": round(rouge_results["rougeL"], 4),
        "bleu": round(bleu_score, 4),
    }


# ============================================
# Before/After Comparison
# ============================================
def compare_models(base_model, finetuned_model, tokenizer, test_passages):
    """Generate questions from both models and compare outputs."""
    print("\n" + "=" * 60)
    print("BEFORE vs AFTER FINE-TUNING COMPARISON")
    print("=" * 60)

    comparisons = []

    for i, passage in enumerate(test_passages, 1):
        input_text = f"Generate a question from the following educational passage: {passage}"
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).input_ids

        # Base model output
        with torch.no_grad():
            base_output = base_model.generate(input_ids, max_length=MAX_TARGET_LENGTH, num_beams=4)
        base_question = tokenizer.decode(base_output[0], skip_special_tokens=True)

        # Fine-tuned model output
        with torch.no_grad():
            ft_output = finetuned_model.generate(input_ids, max_length=MAX_TARGET_LENGTH, num_beams=4)
        ft_question = tokenizer.decode(ft_output[0], skip_special_tokens=True)

        comparison = {
            "passage": passage[:200],
            "base_model_question": base_question,
            "finetuned_model_question": ft_question,
        }
        comparisons.append(comparison)

        print(f"\n--- Example {i} ---")
        print(f"Passage: {passage[:150]}...")
        print(f"Base Model:      {base_question}")
        print(f"Fine-tuned:      {ft_question}")

    return comparisons


# ============================================
# Main Training Pipeline
# ============================================
def main():
    """Run the full fine-tuning pipeline."""
    print("=" * 60)
    print("EduMentor AI - FLAN-T5 Fine-tuning for Question Generation")
    print("=" * 60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Keep a copy of the base model for comparison
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print(f"  Model parameters: {model.num_parameters():,}")

    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset()

    # Preprocess
    print("\nPreprocessing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval",
    )

    print(f"  Tokenized train: {len(tokenized_train)} samples")
    print(f"  Tokenized eval: {len(tokenized_eval)} samples")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Load metrics
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Disable wandb/tensorboard for simplicity
        save_total_limit=2,
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda preds: compute_metrics(preds, tokenizer, rouge_metric, bleu_metric),
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    train_result = trainer.train()

    # Save model
    print("\nSaving fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Evaluate
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    eval_results = trainer.evaluate()
    print(f"\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")

    # Before/After comparison
    test_passages = [
        "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
        "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They are responsible for producing most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
        "Newton's third law states that for every action there is an equal and opposite reaction. This means that forces always occur in pairs. When one object exerts a force on another, the second object exerts an equal force back on the first.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
        "The water cycle describes the continuous movement of water on, above, and below the surface of the Earth. Water evaporates from oceans, condenses into clouds, falls as precipitation, and flows back to the ocean through rivers.",
    ]

    # Load the fine-tuned model fresh for comparison
    finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
    comparisons = compare_models(base_model, finetuned_model, tokenizer, test_passages)

    # Save training report
    report = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "train_samples": len(tokenized_train),
        "eval_samples": len(tokenized_eval),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "training_loss": train_result.training_loss,
        "eval_results": eval_results,
        "comparisons": comparisons,
        "timestamp": datetime.now().isoformat(),
    }

    report_path = os.path.join(OUTPUT_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n\nTraining report saved to: {report_path}")
    print(f"Fine-tuned model saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
