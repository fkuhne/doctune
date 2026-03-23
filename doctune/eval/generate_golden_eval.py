"""
generate_golden_eval.py — Golden evaluation set generator.

Generates complex, multi-step reasoning scenarios for evaluating
domain-specific QA models. Supports OpenAI, Anthropic, and Ollama backends.

Usage:
    python -m doctune.eval.generate_golden_eval [--model MODEL] [--count N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from doctune.utils.provider_utils import build_client, detect_provider, retry_on_rate_limit

logger = logging.getLogger(__name__)

# ==============================================================================
# JSON Instruction Fragments (for non-OpenAI providers)
# ==============================================================================

_JSON_INSTRUCTION = (
    '\n\nYou MUST respond with valid JSON only, using a single key "scenarios" '
    "containing an array of objects.\n"
    'Each object MUST have keys: "prompt", "chosen", "rejected".\n'
    "Do not include any text outside the JSON object."
)


# ==============================================================================
# Unified Scenario Generation
# ==============================================================================


@retry_on_rate_limit()
def _generate_openai(
    client: object, model: str, system_prompt: str, user_prompt: str,
) -> list[dict]:
    """Generate scenarios via OpenAI JSON mode."""
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        text={"format": {"type": "json_object"}},
    )
    data = json.loads(response.output_text)
    return data.get("scenarios", [])


@retry_on_rate_limit()
def _generate_anthropic(
    client: object, model: str, system_prompt: str, user_prompt: str,
) -> list[dict]:
    """Generate scenarios via Anthropic."""
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt + _JSON_INSTRUCTION}],
        temperature=0.7,
    )
    data = json.loads(response.content[0].text)
    return data.get("scenarios", [])


@retry_on_rate_limit()
def _generate_ollama(
    client: object, model: str, system_prompt: str, user_prompt: str,
) -> list[dict]:
    """Generate scenarios via Ollama's OpenAI-compatible endpoint."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + _JSON_INSTRUCTION},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("scenarios", [])


_GENERATE_FNS = {
    "openai": _generate_openai,
    "anthropic": _generate_anthropic,
    "ollama": _generate_ollama,
}


def generate_scenarios(
    client: object,
    provider: str,
    model: str,
    system_prompt: str,
    domain: str,
    batch_size: int = 10,
) -> list[dict]:
    """Generate a batch of evaluation scenarios using the configured provider.

    Args:
        client: The initialized API client.
        provider: One of ``"openai"``, ``"anthropic"``, or ``"ollama"``.
        model: Model identifier.
        system_prompt: System instructions for the teacher model.
        domain: Subject-matter domain string.
        batch_size: Number of scenarios to request per batch.

    Returns:
        List of scenario dicts with ``"prompt"``, ``"chosen"``, ``"rejected"`` keys.
    """
    user_prompt = (
        f"Generate exactly {batch_size} complex, edge-case {domain} "
        "scenarios focusing on multi-step reasoning. Output valid JSON."
    )
    generate_fn = _GENERATE_FNS[provider]
    return generate_fn(client, model, system_prompt, user_prompt)


# ==============================================================================
# Main
# ==============================================================================


def _build_system_prompt(domain: str, batch_size: int) -> str:
    """Build the system prompt for golden eval generation.

    Args:
        domain: Subject-matter domain string.
        batch_size: Number of scenarios to request (embedded in prompt).

    Returns:
        The formatted system prompt string.
    """
    return (
        f"You are an expert in {domain}. Your objective is to create highly complex, "
        "edge-case scenarios that test deep domain knowledge and multi-step reasoning.\n"
        "You must focus purely on multi-step reasoning questions that "
        "require deep diagnostic logic.\n\n"
        f"Output exactly {batch_size} scenarios.\n"
        'Return the output strictly in JSON format using a single key "scenarios", '
        "which contains an array of objects.\n"
        "Each object MUST have the following keys:\n"
        '- "prompt": A detailed user query describing a complex, multi-layered issue.\n'
        '- "chosen": The correct, step-by-step diagnostic and resolution process.\n'
        '- "rejected": A plausible but incorrect or factually flawed resolution.\n'
    )


def main() -> None:
    """Entry point for golden evaluation set generation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate golden evaluation scenarios for domain QA"
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="Teacher model ID (e.g., gpt-4o, claude-3-5-sonnet-20241022, llama3.1:8b)",
    )
    parser.add_argument(
        "--provider", default=None,
        help="API provider: 'openai', 'anthropic', or 'ollama' (auto-detected)",
    )
    parser.add_argument("--api-key", default=None, help="API key (falls back to env vars)")
    parser.add_argument("--count", type=int, default=100, help="Number of scenarios to generate")
    parser.add_argument("--output", default="golden_eval.jsonl", help="Output file path")
    args = parser.parse_args()

    provider = args.provider or detect_provider(args.model)
    client = build_client(provider, api_key=args.api_key)

    domain = os.environ.get("DOMAIN", "technical documentation")
    batch_size = 10
    system_prompt = _build_system_prompt(domain, batch_size)

    logger.info("Generating %d complex %s scenarios using %s:%s...", args.count, domain, provider, args.model)

    scenarios: list[dict] = []
    consecutive_errors = 0
    max_consecutive_errors = 5

    while len(scenarios) < args.count:
        try:
            logger.info("Generating batch... (%d/%d)", len(scenarios), args.count)
            batch = generate_scenarios(client, provider, args.model, system_prompt, domain, batch_size)
            if not batch:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive empty batches. Stopping.")
                    break
                continue
            scenarios.extend(batch)
            consecutive_errors = 0
        except json.JSONDecodeError as exc:
            logger.warning("Batch returned invalid JSON: %s", exc)
            consecutive_errors += 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error generating batch: %s", exc)
            consecutive_errors += 1

        if consecutive_errors >= max_consecutive_errors:
            logger.error("Too many consecutive errors (%d). Stopping.", consecutive_errors)
            break

    # Trim to exactly the target count
    scenarios = scenarios[: args.count]

    if not scenarios:
        logger.error("No scenarios generated. Exiting.")
        sys.exit(1)

    logger.info("Saving to %s...", args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + "\n")

    logger.info("Successfully generated and saved %d scenarios to %s.", len(scenarios), args.output)


if __name__ == "__main__":
    main()
