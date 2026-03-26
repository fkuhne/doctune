"""pricing.py - Local token pricing table and cost helpers.

This module provides a small, editable pricing table for model usage
so pipeline runs can estimate cost from token counts without external calls.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ModelPricing(BaseModel):
    """Per-model pricing in USD per 1M tokens."""

    model_config = ConfigDict(frozen=True)

    input: float
    cached_input: float | None
    output: float

# Prices are USD per 1M tokens (standard tier) and can be expanded as needed.
# Source baseline: https://developers.openai.com/api/docs/pricing
OPENAI_PRICING_PER_1M: dict[str, ModelPricing] = {
    # GPT-5 family
    "gpt-5.2": ModelPricing(input=1.75, cached_input=0.175, output=14.00),
    "gpt-5.2-pro": ModelPricing(input=21.00, cached_input=None, output=168.00),
    "gpt-5.1": ModelPricing(input=1.25, cached_input=0.125, output=10.00),
    "gpt-5": ModelPricing(input=1.25, cached_input=0.125, output=10.00),
    "gpt-5-mini": ModelPricing(input=0.25, cached_input=0.025, output=2.00),
    "gpt-5-nano": ModelPricing(input=0.05, cached_input=0.005, output=0.40),
    "gpt-5-pro": ModelPricing(input=15.00, cached_input=None, output=120.00),
    "gpt-5.4": ModelPricing(input=2.50, cached_input=None, output=15.00),
    "gpt-5.4-mini": ModelPricing(input=0.75, cached_input=None, output=4.50),
    "gpt-5.4-nano": ModelPricing(input=0.20, cached_input=None, output=1.25),
    "gpt-5.4-pro": ModelPricing(input=30.00, cached_input=None, output=180.00),
    "gpt-5.3-chat-latest": ModelPricing(input=1.75, cached_input=None, output=14.00),
    "gpt-5.3-codex": ModelPricing(input=1.75, cached_input=None, output=14.00),
    # Common text models used in existing pipelines (verify against your account if needed)
    "gpt-4o": ModelPricing(input=2.50, cached_input=1.25, output=10.00),
    "gpt-4o-mini": ModelPricing(input=0.15, cached_input=0.075, output=0.60),
    "gpt-4.1": ModelPricing(input=2.00, cached_input=0.50, output=8.00),
    "gpt-4.1-mini": ModelPricing(input=0.40, cached_input=0.10, output=1.60),
    "gpt-4.1-nano": ModelPricing(input=0.10, cached_input=0.025, output=0.40),
    "o1": ModelPricing(input=15.00, cached_input=None, output=60.00),
    "o3": ModelPricing(input=2.00, cached_input=None, output=8.00),
    "o3-mini": ModelPricing(input=1.10, cached_input=None, output=4.40),
    "o4-mini": ModelPricing(input=4.00, cached_input=None, output=16.00),
}


def _normalize_model(model: str) -> str:
    """Return the best pricing-table key for a possibly versioned model id."""
    if model in OPENAI_PRICING_PER_1M:
        return model

    # Prefer longest prefix match so specific entries win over generic ones.
    for candidate in sorted(OPENAI_PRICING_PER_1M, key=len, reverse=True):
        if model.startswith(candidate):
            return candidate
    return model


def compute_model_usage_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Compute estimated USD cost from token usage.

    Args:
        model: Model identifier (supports prefix matching for dated variants).
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        cached_input_tokens: Number of input tokens billed at cached-input rate.

    Returns:
        Estimated USD cost. Unknown models return ``0.0`` and log a warning.
    """
    resolved = _normalize_model(model)
    pricing = OPENAI_PRICING_PER_1M.get(resolved)
    if pricing is None:
        logger.warning("No pricing configured for model '%s'; defaulting cost to 0.0", model)
        return 0.0

    safe_input_tokens = max(0, int(input_tokens))
    safe_output_tokens = max(0, int(output_tokens))
    safe_cached_tokens = max(0, min(int(cached_input_tokens), safe_input_tokens))

    non_cached_input_tokens = safe_input_tokens - safe_cached_tokens
    cached_input_rate = pricing.cached_input if pricing.cached_input is not None else pricing.input

    return (
        (non_cached_input_tokens * pricing.input)
        + (safe_cached_tokens * cached_input_rate)
        + (safe_output_tokens * pricing.output)
    ) / 1_000_000
