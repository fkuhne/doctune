"""
train_dpo.py — Direct Preference Optimization (DPO).

Runs Phase 4 of the training pipeline: aligns the SFT-trained model using
preference tuples (chosen vs. rejected) to penalize hallucinations.

Supports hyperparameter sweeping over beta and learning rate values.

Usage:
    python train_dpo.py --model-id <huggingface-model-id>

Requirements:
    pip install -e ".[training]"
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import TrainingArguments
from peft import PeftModel
from trl import DPOTrainer

from doctune.utils.model_utils import (
    derive_run_name,
    load_tokenizer,
    load_base_model,
    cleanup_memory,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DPO training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Doctune DPO Training")
    parser.add_argument(
        "--model-id", type=str, required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument("--sft-adapter", type=str, default="./doctune-sft")
    parser.add_argument("--dataset", type=str, default="alignment_dataset.jsonl")
    parser.add_argument("--eval-dataset", type=str, default="golden_eval.jsonl")
    parser.add_argument("--output", type=str, default="./doctune-dpo")
    parser.add_argument(
        "--betas", type=float, nargs="+", default=[0.1, 0.25], 
        help="DPO beta values to sweep"
    )
    parser.add_argument(
        "--lrs", type=float, nargs="+", default=[5e-6, 1e-6], 
        help="Learning rates to sweep"
    )
    return parser.parse_args()


def main() -> None: # pylint: disable=too-many-locals
    """Entry point for DPO training with hyperparameter sweep."""
    args = parse_args()

    # 1. Load Tokenizer & Base Model via centralized helpers
    tokenizer = load_tokenizer(args.model_id, padding_side="right")
    base_model = load_base_model(args.model_id, tokenizer)

    # 2. Load SFT Adapters
    print(f"Loading SFT Adapters from {args.sft_adapter}...")
    model = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)

    # 3. Load and Format Datasets
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset, split="train")

    def format_dpo_dataset(example: dict) -> dict:
        """Format a single row for DPOTrainer (prompt, chosen, rejected strings)."""
        prompt_messages = [{"role": "user", "content": example["prompt"]}]
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": formatted_prompt,
            "chosen": f"{example['chosen']}{tokenizer.eos_token}",
            "rejected": f"{example['rejected']}{tokenizer.eos_token}",
        }

    dpo_dataset = dataset.map(format_dpo_dataset)
    dpo_eval_dataset = eval_dataset.map(format_dpo_dataset)

    # 4. Hyperparameter Sweep
    base_run_name = derive_run_name(args.model_id, "dpo")

    for beta_val in args.betas:
        for lr_val in args.lrs:
            run_name = f"{base_run_name}-beta{beta_val}-lr{lr_val}"
            output_dir = f"./{run_name}"
            print(f"\n--- Starting DPO Sweep: Beta={beta_val}, LR={lr_val} ---")

            training_args = TrainingArguments(
                output_dir=output_dir,
                run_name=run_name,
                report_to="mlflow",
                num_train_epochs=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=16,
                gradient_checkpointing=True,
                optim="adamw_torch",
                learning_rate=lr_val,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                logging_steps=10,
                save_strategy="epoch",
                eval_strategy="epoch", # Changed from evaluation_strategy
                bf16=True,
                max_grad_norm=0.3,
                remove_unused_columns=False,
            )

            # pylint: disable=unexpected-keyword-arg
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                train_dataset=dpo_dataset,
                eval_dataset=dpo_eval_dataset,
                tokenizer=tokenizer,
                beta=beta_val,
                max_length=2048,
                max_prompt_length=1024,
            )

            print(f"Initiating DPO Training for {run_name}...")
            trainer.train()
            trainer.save_model(output_dir)
            print(f"Alignment complete for {run_name}. Saved to {output_dir}")

            # Free trainer memory between sweeps
            cleanup_memory(trainer)

    # Final cleanup
    cleanup_memory(model, base_model)


if __name__ == "__main__":
    main()
