"""
model_utils.py — Shared utilities for model-agnostic fine-tuning

Provides auto-detection of LoRA target modules, attention implementation,
chat template formatting, and run name derivation for any HuggingFace model.
"""

import re

import torch.nn as nn


def detect_lora_target_modules(model) -> list[str]:
    """Auto-detect all Linear layer names suitable for LoRA injection.

    Scans the model's named modules and returns the unique base names of all
    ``nn.Linear`` layers (e.g. ``["q_proj", "v_proj", "dense", ...]``).
    Works for any HuggingFace architecture (LLaMA, Mistral, Phi, Gemma, Qwen, OLMo, etc.).
    """
    target_modules: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extract the leaf name (e.g. "model.layers.0.self_attn.q_proj" → "q_proj")
            base_name = name.split(".")[-1]
            target_modules.add(base_name)

    # Remove lm_head / embed_tokens — these are handled separately via modules_to_save
    target_modules.discard("lm_head")
    target_modules.discard("embed_tokens")

    if not target_modules:
        # Fallback: common projection names found in most transformer architectures
        print("WARNING: Could not auto-detect Linear layers. Using common defaults.")
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    detected = sorted(target_modules)
    print(f"Auto-detected LoRA target modules: {detected}")
    return detected


def detect_attn_implementation() -> str:
    """Return the best available attention implementation.

    Returns ``"flash_attention_2"`` if the ``flash_attn`` package is installed,
    otherwise falls back to ``"eager"``.
    """
    try:
        import flash_attn  # noqa: F401
        print("Flash Attention 2 detected — using flash_attention_2.")
        return "flash_attention_2"
    except ImportError:
        print("Flash Attention 2 not available — using eager attention.")
        return "eager"


def format_prompt_for_eval(tokenizer, prompt_text: str) -> str:
    """Format a user prompt for generation using the tokenizer's chat template.

    Uses ``apply_chat_template()`` if the tokenizer has a chat template configured.
    Falls back to a simple generic format otherwise.
    """
    messages = [{"role": "user", "content": prompt_text}]

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass  # Fall through to generic format

    # Generic fallback for models without a chat template
    return f"User: {prompt_text}\nAssistant: "


def derive_run_name(model_id: str, stage: str) -> str:
    """Derive a clean MLflow run name from a HuggingFace model ID.

    Examples:
        derive_run_name("meta-llama/Llama-3.1-8B", "sft") → "llama-3.1-8b-sft-v1"
        derive_run_name("allenai/OLMo-2-0425-1B", "dpo")  → "olmo-2-0425-1b-dpo-v1"
        derive_run_name("mistralai/Mistral-7B-v0.3", "sft") → "mistral-7b-v0.3-sft-v1"
    """
    # Take just the model name (after the org prefix)
    name = model_id.split("/")[-1]
    # Lowercase and clean up
    slug = re.sub(r"[^a-z0-9._-]", "-", name.lower())
    # Collapse multiple dashes
    slug = re.sub(r"-+", "-", slug).strip("-")
    return f"{slug}-{stage}-v1"
