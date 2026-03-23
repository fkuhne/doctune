"""
merge_model.py — Weight Merging for Deployment.

Runs Phase 6 (Step 1) of the training pipeline: fuses the DPO-aligned LoRA
adapters back into the base model to create a standalone binary for deployment.

Usage:
    python merge_model.py --model-id <huggingface-model-id>

Requirements:
    pip install -e ".[training]"
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from doctune.utils.model_utils import load_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for weight merging.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Doctune Weight Merging")
    parser.add_argument(
        "--model-id", type=str, required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument("--adapter", type=str, default="./doctune-dpo")
    parser.add_argument("--output", type=str, default="./doctune-merged")
    return parser.parse_args()


def main() -> None:
    """Entry point for weight merging."""
    args = parse_args()

    # 1. Load Base Model to CPU (avoids VRAM spikes during merging)
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # 2. Load Tokenizer and resize embeddings
    tokenizer = load_tokenizer(args.model_id)
    base_model.resize_token_embeddings(len(tokenizer))

    # 3. Load LoRA Adapters
    print("Loading LoRA Adapters...")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    # 4. Fuse Weights
    print("Fusing Weights...")
    merged_model = model.merge_and_unload()

    # 5. Save Standalone Model + Tokenizer
    print(f"Saving Standalone Model to {args.output}...")
    merged_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("Merge Complete. Model is ready for production inference.")


if __name__ == "__main__":
    main()
