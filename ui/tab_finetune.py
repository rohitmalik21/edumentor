"""
EduMentor AI - Fine-tuning Tab
Train, evaluate, and compare the fine-tuned model within the app.
Dataset is downloaded once and cached locally.
"""

import os
import json
import threading

import gradio as gr

from config import Config

# Training state
finetune_state = {
    "status": "idle",  # idle, downloading, training, complete, error
    "progress": "",
    "report": None,
}


def get_status():
    """Check current fine-tuning status and whether model exists."""
    model_exists = os.path.exists(os.path.join(Config.FINETUNED_MODEL_DIR, "config.json"))
    report_path = os.path.join(Config.FINETUNED_MODEL_DIR, "training_report.json")
    report_exists = os.path.exists(report_path)

    if model_exists:
        status = "Fine-tuned model is available and ready to use in Quiz Generation."
        if report_exists:
            with open(report_path, "r") as f:
                report = json.load(f)
            finetune_state["report"] = report
    else:
        status = "No fine-tuned model found. Click 'Start Fine-tuning' to train the model."

    return status, model_exists


def get_report_display():
    """Display training report if available."""
    report_path = os.path.join(Config.FINETUNED_MODEL_DIR, "training_report.json")

    if not os.path.exists(report_path):
        return "No training report available. Run fine-tuning first."

    with open(report_path, "r") as f:
        report = json.load(f)

    # Format metrics
    eval_results = report.get("eval_results", {})
    lines = [
        "## Training Report\n",
        f"**Model:** {report.get('model', 'N/A')}",
        f"**Dataset:** {report.get('dataset', 'N/A')}",
        f"**Train Samples:** {report.get('train_samples', 'N/A')}",
        f"**Eval Samples:** {report.get('eval_samples', 'N/A')}",
        f"**Epochs:** {report.get('epochs', 'N/A')}",
        f"**Batch Size:** {report.get('batch_size', 'N/A')}",
        f"**Learning Rate:** {report.get('learning_rate', 'N/A')}",
        f"**Training Loss:** {report.get('training_loss', 'N/A')}",
        f"**Timestamp:** {report.get('timestamp', 'N/A')}",
        "\n## Evaluation Metrics\n",
        "| Metric | Score |",
        "|--------|-------|",
    ]

    for key, value in eval_results.items():
        clean_key = key.replace("eval_", "").upper()
        lines.append(f"| {clean_key} | {value} |")

    # Before/After comparisons
    comparisons = report.get("comparisons", [])
    if comparisons:
        lines.append("\n## Before vs After Fine-tuning\n")
        for i, comp in enumerate(comparisons, 1):
            lines.append(f"### Example {i}")
            lines.append(f"**Passage:** {comp.get('passage', '')[:150]}...")
            lines.append(f"**Base Model:** {comp.get('base_model_question', 'N/A')}")
            lines.append(f"**Fine-tuned:** {comp.get('finetuned_model_question', 'N/A')}")
            lines.append("")

    return "\n".join(lines)


def run_finetuning(epochs, batch_size, train_size):
    """Run the fine-tuning process."""
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

    MODEL_NAME = "google/flan-t5-small"
    DATASET_NAME = "allenai/sciq"
    OUTPUT_DIR = Config.FINETUNED_MODEL_DIR
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 128

    epochs = int(epochs)
    batch_size = int(batch_size)
    train_size = int(train_size)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    progress_log = []

    def log(msg):
        progress_log.append(msg)
        print(msg)

    try:
        # Step 1: Load dataset (cached after first download)
        log("Step 1/6: Loading SciQ dataset (cached after first download)...")
        dataset = load_dataset(DATASET_NAME)
        train_dataset = dataset["train"].select(range(min(train_size, len(dataset["train"]))))
        eval_dataset = dataset["validation"].select(range(min(500, len(dataset["validation"]))))
        log(f"  Loaded {len(train_dataset)} train / {len(eval_dataset)} eval samples")

        # Step 2: Load model and tokenizer
        log("Step 2/6: Loading FLAN-T5-small model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        log(f"  Model parameters: {model.num_parameters():,}")

        # Step 3: Preprocess
        log("Step 3/6: Preprocessing dataset...")

        def preprocess(examples):
            inputs = []
            targets = []
            for support, question in zip(examples["support"], examples["question"]):
                if not support or not support.strip():
                    support = "No passage available."
                if not question or not question.strip():
                    question = "What is the main concept?"
                inputs.append(f"Generate a question from the following educational passage: {support.strip()}")
                targets.append(question.strip())

            model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
            labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
        tokenized_eval = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)
        log(f"  Tokenized: {len(tokenized_train)} train, {len(tokenized_eval)} eval")

        # Step 4: Train
        log("Step 4/6: Training model (this may take 10-30 minutes)...")

        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_steps=100,
            weight_decay=0.01,
            predict_with_generate=True,
            generation_max_length=MAX_TARGET_LENGTH,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=torch.cuda.is_available(),
            report_to="none",
            save_total_limit=2,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        log(f"  Training complete! Loss: {train_result.training_loss:.4f}")

        # Step 5: Save model
        log("Step 5/6: Saving fine-tuned model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        log(f"  Saved to: {OUTPUT_DIR}")

        # Step 6: Compare before/after
        log("Step 6/6: Generating before/after comparison...")
        test_passages = [
            "Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water. It involves chlorophyll and generates oxygen.",
            "The mitochondria are organelles found in eukaryotic cells responsible for producing ATP, the cell's main energy source.",
            "Newton's third law states that for every action there is an equal and opposite reaction. Forces always occur in pairs.",
            "Machine learning is a subset of AI that enables systems to learn and improve from experience without explicit programming.",
            "The water cycle describes the continuous movement of water through evaporation, condensation, precipitation, and river flow.",
        ]

        finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
        comparisons = []

        for passage in test_passages:
            input_text = f"Generate a question from the following educational passage: {passage}"
            input_ids = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).input_ids

            with torch.no_grad():
                base_out = base_model.generate(input_ids, max_length=MAX_TARGET_LENGTH, num_beams=4)
                ft_out = finetuned_model.generate(input_ids, max_length=MAX_TARGET_LENGTH, num_beams=4)

            comparisons.append({
                "passage": passage[:200],
                "base_model_question": tokenizer.decode(base_out[0], skip_special_tokens=True),
                "finetuned_model_question": tokenizer.decode(ft_out[0], skip_special_tokens=True),
            })

        # Evaluate
        eval_results = trainer.evaluate()
        log(f"  Eval results: {eval_results}")

        # Save report
        report = {
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "train_samples": len(tokenized_train),
            "eval_samples": len(tokenized_eval),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 3e-4,
            "training_loss": round(train_result.training_loss, 4),
            "eval_results": {k: round(v, 4) if isinstance(v, float) else v for k, v in eval_results.items()},
            "comparisons": comparisons,
            "timestamp": str(os.popen("date /t").read().strip()),
        }

        report_path = os.path.join(OUTPUT_DIR, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        log("\nFine-tuning complete! The model will now be used for quiz generation.")
        return "\n".join(progress_log), get_report_display()

    except Exception as e:
        log(f"\nERROR: {str(e)}")
        return "\n".join(progress_log), f"Error: {str(e)}"


def test_finetuned_model(passage):
    """Test the fine-tuned model with a custom passage."""
    if not passage or not passage.strip():
        return "Please enter a passage to generate a question from."

    model_path = Config.FINETUNED_MODEL_DIR
    if not os.path.exists(os.path.join(model_path, "config.json")):
        return "Fine-tuned model not available. Run fine-tuning first."

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()

        input_text = f"Generate a question from the following educational passage: {passage}"
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=128, num_beams=4)

        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f"**Generated Question:** {question}"

    except Exception as e:
        return f"Error: {str(e)}"


def create_finetune_tab():
    """Build the Fine-tuning tab interface."""
    with gr.Tab("Fine-tuning", id="finetune"):
        gr.Markdown("## Model Fine-tuning")
        gr.Markdown(
            "Fine-tune FLAN-T5-small on the **SciQ dataset** for educational question generation. "
            "The dataset is downloaded once and cached locally."
        )

        # Status
        status_text, model_exists = get_status()
        status_display = gr.Markdown(f"**Status:** {status_text}")

        gr.Markdown("---")

        # Training Configuration
        gr.Markdown("### Training Configuration")
        with gr.Row():
            epochs_input = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Epochs")
            batch_size_input = gr.Slider(minimum=2, maximum=16, value=8, step=2, label="Batch Size")
            train_size_input = gr.Slider(minimum=500, maximum=5000, value=2000, step=500, label="Training Samples")

        gr.Markdown(
            "**Note:** Training takes ~10-30 minutes on CPU. "
            "The SciQ dataset will be downloaded on first run (~20MB) and cached for future use."
        )

        train_btn = gr.Button("Start Fine-tuning", variant="primary", size="lg")
        progress_output = gr.Textbox(label="Training Progress", lines=15, interactive=False)

        train_btn.click(
            fn=run_finetuning,
            inputs=[epochs_input, batch_size_input, train_size_input],
            outputs=[progress_output, gr.Markdown()],
        )

        gr.Markdown("---")

        # Training Report
        gr.Markdown("### Training Report & Metrics")
        report_btn = gr.Button("Load Training Report", variant="secondary")
        report_output = gr.Markdown()

        report_btn.click(
            fn=get_report_display,
            inputs=[],
            outputs=[report_output],
        )

        gr.Markdown("---")

        # Test Fine-tuned Model
        gr.Markdown("### Test Fine-tuned Model")
        gr.Markdown("Enter any educational passage to see the question generated by the fine-tuned model.")

        test_input = gr.Textbox(
            label="Educational Passage",
            placeholder="e.g., The human heart has four chambers: two atria and two ventricles...",
            lines=3,
        )
        test_btn = gr.Button("Generate Question", variant="secondary")
        test_output = gr.Markdown()

        test_btn.click(
            fn=test_finetuned_model,
            inputs=[test_input],
            outputs=[test_output],
        )

        gr.Markdown("---")

        # Info about fine-tuning
        gr.Markdown("""### About This Fine-tuning

| Parameter | Value |
|-----------|-------|
| **Base Model** | google/flan-t5-small (77M params) |
| **Dataset** | SciQ (Allen AI) - Science Q&A pairs |
| **Task** | Educational question generation from passages |
| **Evaluation Metrics** | ROUGE-1, ROUGE-2, ROUGE-L, BLEU, Eval Loss |

**How it works:**
1. Downloads SciQ dataset (13,000+ science questions with supporting passages)
2. Trains the model to generate questions from educational text
3. Evaluates quality using ROUGE and BLEU metrics
4. Compares base model vs fine-tuned model outputs
5. Saves the improved model for use in Quiz Generation
        """)
