"""
model_utils.py — Shared utilities for model-agnostic fine-tuning.

Provides auto-detection of LoRA target modules, attention implementation,
chat template formatting, run name derivation, and centralized tokenizer /
model loading for any HuggingFace causal LM.
"""

from __future__ import annotations

import re
import logging

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ==============================================================================
# LoRA Target Detection
# ==============================================================================
def detect_lora_target_modules(model: nn.Module) -> list[str]:
    """Auto-detect all Linear layer names suitable for LoRA injection.

    Scans the model's ``named_modules()`` and returns the unique base names
    of all ``nn.Linear`` layers (e.g. ``["q_proj", "v_proj", "dense"]``).

    Args:
        model: A loaded HuggingFace model instance.

    Returns:
        Sorted list of unique linear-layer base names. Falls back to common
        transformer projection names if none are detected.
    """
    target_modules: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            base_name = name.split(".")[-1]
            target_modules.add(base_name)

    # lm_head / embed_tokens are handled separately via modules_to_save
    target_modules.discard("lm_head")
    target_modules.discard("embed_tokens")

    if not target_modules:
        logger.warning("Could not auto-detect Linear layers. Using common defaults.")
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    detected = sorted(target_modules)
    print(f"Auto-detected LoRA target modules: {detected}")
    return detected


# ==============================================================================
# Attention Implementation Detection
# ==============================================================================
def detect_attn_implementation() -> str:
    """Return the best available attention implementation.

    Returns:
        ``"flash_attention_2"`` if the ``flash_attn`` package is importable,
        otherwise ``"eager"``.
    """
    try:
        # pylint: disable=import-outside-toplevel,unused-import
        import flash_attn  # noqa: F401
        print("Flash Attention 2 detected — using flash_attention_2.")
        return "flash_attention_2"
    except ImportError:
        print("Flash Attention 2 not available — using eager attention.")
        return "eager"


# ==============================================================================
# Centralized Tokenizer Loading
# ==============================================================================
def load_tokenizer(
    model_id: str,
    padding_side: str = "right",
) -> AutoTokenizer:
    """Load and configure a tokenizer for fine-tuning.

    Adds a dedicated ``<|pad|>`` token if the tokenizer lacks one, and
    sets the requested padding side.

    Args:
        model_id: HuggingFace model identifier (e.g. ``"meta-llama/Llama-3.1-8B"``).
        padding_side: ``"right"`` for training, ``"left"`` for generation.

    Returns:
        A configured ``AutoTokenizer`` instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = padding_side
    return tokenizer


# ==============================================================================
# Centralized Model Loading
# ==============================================================================
def load_base_model(
    model_id: str,
    tokenizer: AutoTokenizer,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """Load a HuggingFace causal LM with auto-detected attention and resized embeddings.

    Args:
        model_id: HuggingFace model identifier.
        tokenizer: The tokenizer (used to resize embeddings if pad token was added).
        device_map: Device placement strategy (default ``"auto"``).
        torch_dtype: Weight precision (default ``torch.bfloat16``).

    Returns:
        A loaded ``AutoModelForCausalLM`` with embeddings resized to match
        the tokenizer vocabulary.
    """
    attn_impl = detect_attn_implementation()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model


# ==============================================================================
# Chat Template Formatting
# ==============================================================================
def format_prompt_for_eval(tokenizer: AutoTokenizer, prompt_text: str) -> str:
    """Format a user prompt for generation using the tokenizer's chat template.

    Uses ``apply_chat_template()`` if available, otherwise falls back to a
    simple generic format.

    Args:
        tokenizer: The model's tokenizer.
        prompt_text: Raw user question text.

    Returns:
        A formatted prompt string ready for tokenization.
    """
    messages = [{"role": "user", "content": prompt_text}]

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception: # pylint: disable=broad-exception-caught
            pass  # Fall through to generic format

    return f"User: {prompt_text}\nAssistant: "


# ==============================================================================
# Run Name Derivation
# ==============================================================================
def derive_run_name(model_id: str, stage: str) -> str:
    """Derive a clean MLflow run name from a HuggingFace model ID.

    Args:
        model_id: Full HuggingFace model path (e.g. ``"meta-llama/Llama-3.1-8B"``).
        stage: Training stage label (e.g. ``"sft"``, ``"dpo"``).

    Returns:
        A slug like ``"llama-3.1-8b-sft-v1"``.
    """
    name = model_id.split("/")[-1]
    slug = re.sub(r"[^a-z0-9._-]", "-", name.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return f"{slug}-{stage}-v1"


# ==============================================================================
# Memory Cleanup
# ==============================================================================
def cleanup_memory(*objects: object) -> None:
    """Delete objects and free GPU memory if available.

    Args:
        *objects: Any Python objects to delete (models, trainers, etc.).
    """
    for obj in objects:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared.")
