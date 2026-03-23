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
import logging

from peft import PeftModel
from trl import DPOTrainer

from doctune.training.training_utils import (
    add_common_train_args,
    build_training_args,
    load_datasets,
)
from doctune.utils.model_utils import (
    clear_gpu_cache,
    derive_run_name,
    load_base_model,
    load_tokenizer,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DPO training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Doctune DPO Training")
    add_common_train_args(parser)
    parser.add_argument("--sft-adapter", type=str, default="./doctune-sft")
    parser.add_argument(
        "--betas", type=float, nargs="+", default=[0.1, 0.25],
        help="DPO beta values to sweep",
    )
    parser.add_argument(
        "--lrs", type=float, nargs="+", default=[5e-6, 1e-6],
        help="Learning rates to sweep",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for DPO training with hyperparameter sweep."""
    args = parse_args()

    # 1. Load Tokenizer & Base Model
    tokenizer = load_tokenizer(args.model_id, padding_side="right")
    base_model = load_base_model(args.model_id, tokenizer)

    # 2. Load SFT Adapters
    logger.info("Loading SFT Adapters from %s...", args.sft_adapter)
    model = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)

    # 3. Load and Format Datasets
    dataset, eval_dataset = load_datasets(args.dataset, args.eval_dataset)

    def format_dpo_dataset(example: dict) -> dict:
        """Format a single row for DPOTrainer (prompt, chosen, rejected strings)."""
        prompt_messages = [{"role": "user", "content": example["prompt"]}]
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
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
            logger.info("Starting DPO Sweep: Beta=%s, LR=%s", beta_val, lr_val)

            training_args = build_training_args(
                output_dir=output_dir,
                run_name=run_name,
                batch_size=2,
                grad_accum=16,
                lr=lr_val,
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
                max_length=args.max_seq_length,
                max_prompt_length=1024,
            )

            logger.info("Initiating DPO Training for %s...", run_name)
            trainer.train()
            trainer.save_model(output_dir)
            logger.info("Alignment complete for %s. Saved to %s", run_name, output_dir)

            # Free trainer memory between sweeps
            del trainer
            clear_gpu_cache()

    # Final cleanup
    del model, base_model
    clear_gpu_cache()


if __name__ == "__main__":
    main()
