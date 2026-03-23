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
import logging

from peft import PeftModel

from doctune.utils.model_utils import clear_gpu_cache, load_base_model, load_tokenizer

logger = logging.getLogger(__name__)


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

    # 1. Load base model to CPU (avoids VRAM spikes during merging)
    logger.info("Loading base model '%s'...", args.model_id)
    tokenizer = load_tokenizer(args.model_id)
    base_model = load_base_model(args.model_id, tokenizer, device_map="cpu")

    # 2. Load LoRA adapters
    logger.info("Loading LoRA adapters from '%s'...", args.adapter)
    model = PeftModel.from_pretrained(base_model, args.adapter)

    # 3. Fuse weights
    logger.info("Fusing weights...")
    merged_model = model.merge_and_unload()

    # 4. Save standalone model + tokenizer
    logger.info("Saving merged model to '%s'...", args.output)
    merged_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    logger.info("Merge complete. Model is ready for production inference.")

    # 5. Cleanup
    del merged_model, model, base_model
    clear_gpu_cache()


if __name__ == "__main__":
    main()
