"""
training_utils.py — Shared helpers for SFT and DPO training scripts.

Centralizes common CLI argument definitions, TrainingArguments construction,
and dataset loading to eliminate duplication between train_sft.py and
train_dpo.py.
"""

from __future__ import annotations

import argparse
import logging

from datasets import Dataset, load_dataset
from transformers import TrainingArguments

logger = logging.getLogger(__name__)


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared by all training scripts.

    Args:
        parser: The argument parser to populate.
    """
    parser.add_argument(
        "--model-id", type=str, required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument("--dataset", type=str, default="alignment_dataset.jsonl")
    parser.add_argument("--eval-dataset", type=str, default="golden_eval.jsonl")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (derived from run name if omitted)")
    parser.add_argument("--max-seq-length", type=int, default=2048)


def build_training_args(
    *,
    output_dir: str,
    run_name: str,
    epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 2e-4,
    remove_unused_columns: bool = True,
) -> TrainingArguments:
    """Build a ``TrainingArguments`` with project-wide defaults.

    Args:
        output_dir: Directory for checkpoints and final model.
        run_name: MLflow run name.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        grad_accum: Gradient accumulation steps.
        lr: Learning rate.
        remove_unused_columns: Whether to drop unused dataset columns.

    Returns:
        A fully-configured ``TrainingArguments`` instance.
    """
    return TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        report_to="mlflow",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        remove_unused_columns=remove_unused_columns,
    )


def load_datasets(data_path: str, eval_path: str) -> tuple[Dataset, Dataset]:
    """Load train and eval datasets from JSONL files.

    Args:
        data_path: Path to the training JSONL file.
        eval_path: Path to the evaluation JSONL file.

    Returns:
        A ``(train_dataset, eval_dataset)`` tuple.
    """
    train_ds = load_dataset("json", data_files=data_path, split="train")
    eval_ds = load_dataset("json", data_files=eval_path, split="train")
    return train_ds, eval_ds
