"""
generate_golden_eval.py — Golden evaluation set generator.

Generates complex, multi-step reasoning scenarios for evaluating
domain-specific QA models. Supports OpenAI, Anthropic, and Ollama backends.

Usage:
    python generate_golden_eval.py [--model MODEL] [--count N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from doctune.utils.provider_utils import build_client, detect_provider, retry_on_rate_limit

logger = logging.getLogger(__name__)

# ==============================================================================
# Generation Functions (per provider)
# ==============================================================================


@retry_on_rate_limit()
def generate_scenarios_openai(
    client: object,
    model: str,
    system_prompt: str,
    domain: str,
    batch_size: int = 10,
) -> list[dict]:
    """Generate evaluation scenarios via OpenAI JSON mode.

    Args:
        client: OpenAI client instance.
        model: Model identifier (e.g. ``"gpt-4o"``).
        system_prompt: System instructions.
        domain: Subject-matter domain string.
        batch_size: Number of scenarios per batch.

    Returns:
        List of scenario dicts with ``"prompt"``, ``"chosen"``, ``"rejected"`` keys.
    """
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=(
            f"Generate exactly {batch_size} complex, edge-case {domain} "
            "scenarios focusing on multi-step reasoning. Output valid JSON."
        ),
        text={"format": {"type": "json_object"}},
    )
    data = json.loads(response.output_text)
    return data.get("scenarios", [])


@retry_on_rate_limit()
def generate_scenarios_anthropic(
    client: object,
    model: str,
    system_prompt: str,
    domain: str,
    batch_size: int = 10,
) -> list[dict]:
    """Generate evaluation scenarios via Anthropic JSON mode.

    Args:
        client: Anthropic client instance.
        model: Model identifier.
        system_prompt: System instructions.
        domain: Subject-matter domain string.
        batch_size: Number of scenarios per batch.

    Returns:
        List of scenario dicts.
    """
    user_prompt = (
        f"Generate exactly {batch_size} complex, edge-case {domain} "
        "scenarios focusing on multi-step reasoning.\n\n"
        'You MUST respond with valid JSON only, using a single key "scenarios" '
        "containing an array of objects.\n"
        "Do not include any text outside the JSON object."
    )
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7,
    )
    data = json.loads(response.content[0].text)
    return data.get("scenarios", [])


@retry_on_rate_limit()
def generate_scenarios_ollama(
    client: object,
    model: str,
    system_prompt: str,
    domain: str,
    batch_size: int = 10,
) -> list[dict]:
    """Generate evaluation scenarios via Ollama's OpenAI-compatible endpoint.

    Args:
        client: OpenAI-compatible Ollama client.
        model: Model identifier (e.g. ``"llama3.1:8b"``).
        system_prompt: System instructions.
        domain: Subject-matter domain string.
        batch_size: Number of scenarios per batch.

    Returns:
        List of scenario dicts.
    """
    user_prompt = (
        f"Generate exactly {batch_size} complex, edge-case {domain} "
        "scenarios focusing on multi-step reasoning.\n\n"
        'You MUST respond with valid JSON only, using a single key "scenarios" '
        "containing an array of objects.\n"
        'Each object MUST have keys: "prompt", "chosen", "rejected".\n'
        "Do not include any text outside the JSON object."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("scenarios", [])


# ==============================================================================
# Main
# ==============================================================================

_GENERATE_FNS = {
    "openai": generate_scenarios_openai,
    "anthropic": generate_scenarios_anthropic,
    "ollama": generate_scenarios_ollama,
}


def main() -> None:
    """Entry point for golden evaluation set generation."""
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

    system_prompt = (
        f"You are an expert in {domain}. Your objective is to create highly complex, "
        "edge-case scenarios that test deep domain knowledge and multi-step reasoning.\n"
        "You must focus purely on multi-step reasoning questions that "
        "require deep diagnostic logic.\n\n"
        "Output exactly 10 scenarios.\n"
        'Return the output strictly in JSON format using a single key "scenarios", '
        "which contains an array of objects.\n"
        "Each object MUST have the following keys:\n"
        '- "prompt": A detailed user query describing a complex, multi-layered issue.\n'
        '- "chosen": The correct, step-by-step diagnostic and resolution process.\n'
        '- "rejected": A plausible but incorrect or factually flawed resolution.\n'
    )

    generate_fn = _GENERATE_FNS[provider]
    batch_size = 10

    print(f"Generating {args.count} complex {domain} scenarios using {provider}:{args.model}...")

    scenarios: list[dict] = []
    consecutive_errors = 0
    max_consecutive_errors = 5

    while len(scenarios) < args.count:
        try:
            print(f"Generating batch... ({len(scenarios)}/{args.count})")
            batch = generate_fn(client, args.model, system_prompt, domain, batch_size)
            if not batch:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive empty batches. Stopping.")
                    break
                continue
            scenarios.extend(batch)
            consecutive_errors = 0
        except json.JSONDecodeError as e:
            logger.warning("Batch returned invalid JSON: %s", e)
            consecutive_errors += 1
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Error generating batch: %s", e)
            print(f"Error generating batch: {e}")
            consecutive_errors += 1

        if consecutive_errors >= max_consecutive_errors:
            logger.error("Too many consecutive errors (%d). Stopping.", consecutive_errors)
            break

    # Trim to exactly the target count
    scenarios = scenarios[: args.count]

    print(f"Saving to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + "\n")

    print(f"Successfully generated and saved {len(scenarios)} scenarios to {args.output}.")


if __name__ == "__main__":
    main()
