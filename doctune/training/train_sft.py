"""
train_sft.py — Supervised Fine-Tuning (SFT).

Runs Phase 3 of the training pipeline: injects domain-specific knowledge
into any HuggingFace foundation model using LoRA adapters and the SFTTrainer.

Usage:
    python train_sft.py --model-id <huggingface-model-id>

Requirements:
    pip install -e ".[training]"
"""

from __future__ import annotations

import argparse
import logging

from peft import LoraConfig, TaskType
from trl import SFTTrainer

from doctune.training.training_utils import (
    add_common_train_args,
    build_training_args,
    load_datasets,
)
from doctune.utils.model_utils import (
    clear_gpu_cache,
    derive_run_name,
    detect_lora_target_modules,
    load_base_model,
    load_tokenizer,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SFT training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Doctune SFT Training")
    add_common_train_args(parser)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Entry point for SFT training."""
    args = parse_args()

    # 1. Load Tokenizer & Model
    tokenizer = load_tokenizer(args.model_id, padding_side="right")
    model = load_base_model(args.model_id, tokenizer)

    # 2. LoRA Configuration — auto-detect target modules from model architecture
    target_modules = detect_lora_target_modules(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        modules_to_save=["lm_head", "embed_tokens"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 3. Load and Format Dataset
    dataset, eval_dataset = load_datasets(args.dataset, args.eval_dataset)

    def formatting_prompts_func(example: dict) -> list[str]:
        """Convert JSON rows into conversational strings via the chat template."""
        output_texts: list[str] = []
        for prompt, chosen in zip(example["prompt"], example["chosen"]):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        return output_texts

    # 4. Training Arguments
    run_name = derive_run_name(args.model_id, "sft")
    output_dir = args.output or "./doctune-sft"
    training_args = build_training_args(
        output_dir=output_dir,
        run_name=run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )

    # 5. Initialize & Run SFTTrainer
    # pylint: disable=unexpected-keyword-arg
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Initiating Supervised Fine-Tuning...")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Training complete. LoRA adapters saved to %s", output_dir)

    # 6. Cleanup
    del trainer, model
    clear_gpu_cache()


if __name__ == "__main__":
    main()
