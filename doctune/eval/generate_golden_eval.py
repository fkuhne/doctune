"""
generate_golden_eval.py — Golden evaluation set generator.

Generates type-balanced evaluation scenarios (factual, procedural,
edge-case/diagnostic) for evaluating domain-specific QA models.
Supports OpenAI, Anthropic, and Ollama backends.

Features:
    - Type-balanced distribution (30% factual, 40% procedural, 30% edge-case)
    - Checkpoint/resume support for interrupted runs
    - Pre-flight cost estimation with interactive confirmation

Usage:
    python -m doctune.eval.generate_golden_eval [--model MODEL] [--count N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from doctune.utils.pricing import estimate_batch_cost
from doctune.utils.provider_utils import (
    build_client,
    check_provider_separation,
    detect_provider,
    get_alternative_models,
    retry_on_rate_limit,
)

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

# ---------------------------------------------------------------------------
# Eval contamination guard
# ---------------------------------------------------------------------------

def _check_family_separation(eval_model: str, train_model: str) -> None:
    """Enforce that eval and training models come from different providers.

    Args:
        eval_model: Model identifier to be used for eval generation.
        train_model: Model identifier used to generate training data.

    Raises:
        SystemExit: If both models resolve to the same provider family,
            unless the user explicitly overrides with ``--allow-same-family``.
    """
    is_separated, eval_provider, train_provider = check_provider_separation(
        eval_model, train_model,
    )

    if not is_separated:
        print(
            f"\n  [CONTAMINATION WARNING]\n"
            f"  Eval model  : {eval_model!r}  (provider: {eval_provider})\n"
            f"  Train model : {train_model!r}  (provider: {train_provider})\n"
            f"\n"
            f"  Both models are from the same provider family.\n"
            f"  The eval set will share distributional biases with the training\n"
            f"  data, making accuracy metrics unreliable.\n"
            f"\n"
            f"  Recommended eval models when training provider is '{train_provider}':\n"
        )
        for alt in get_alternative_models(train_provider):
            print(f"    {alt}")
        print(
            f"\n  To override this check (not recommended), pass:\n"
            f"    --allow-same-family\n"
        )
        sys.exit(1)

    print(
        f"  [OK] Family separation confirmed: "
        f"eval={eval_provider!r}, train={train_provider!r}"
    )


# ---------------------------------------------------------------------------
# Eval scenario type definitions
# ---------------------------------------------------------------------------

_SCENARIO_TYPES: dict[str, dict] = {
    "factual": {
        "label": "factual",
        "description": (
            "Direct knowledge retrieval — the user asks for a specific attribute, "
            "specification, or fact (e.g. error code meaning, supported paper size, "
            "button location).  No multi-step reasoning required."
        ),
        "share": 0.30,
    },
    "procedural": {
        "label": "procedural",
        "description": (
            "Step-by-step how-to execution — the user asks how to perform a procedure "
            "in the correct sequence (e.g. configure duplex printing, install a toner "
            "cartridge, connect to Wi-Fi).  The rejected answer must contain a "
            "wrong-sequence or wrong-component flaw."
        ),
        "share": 0.40,
    },
    "edge_case": {
        "label": "edge-case / diagnostic",
        "description": (
            "Multi-step diagnostic reasoning under unusual conditions — the user "
            "describes a symptom combination or failure mode that requires ruling "
            "out multiple causes before reaching a resolution.  High complexity only."
        ),
        "share": 0.30,
    },
}


def _allocate_type_counts(total: int) -> dict[str, int]:
    """Allocate scenario counts across types, rounding to sum exactly to total.

    Args:
        total: Target total number of scenarios.

    Returns:
        Dict mapping type label to integer count.
    """
    raw = {k: v["share"] * total for k, v in _SCENARIO_TYPES.items()}
    counts = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    # Distribute remainder to the types with the largest fractional parts
    fractional = sorted(raw.keys(), key=lambda k: -(raw[k] - counts[k]))
    for k in fractional[:remainder]:
        counts[k] += 1
    return counts


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_SUFFIX = ".checkpoint.jsonl"


def _checkpoint_path(output_path: str) -> str:
    """Return the checkpoint file path for the given output path."""
    return output_path + _CHECKPOINT_SUFFIX


def _load_checkpoint(output_path: str) -> list[dict]:
    """Load previously generated scenarios from a checkpoint file, if present."""
    cp = _checkpoint_path(output_path)
    if not os.path.exists(cp):
        return []
    scenarios = []
    with open(cp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    scenarios.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if scenarios:
        logger.info("[CHECKPOINT] Resuming from %d existing scenarios.", len(scenarios))
    return scenarios


def _append_checkpoint(output_path: str, batch: list[dict]) -> None:
    """Append a completed batch to the checkpoint file immediately."""
    cp = _checkpoint_path(output_path)
    with open(cp, "a", encoding="utf-8") as f:
        for scenario in batch:
            f.write(json.dumps(scenario) + "\n")


def _clear_checkpoint(output_path: str) -> None:
    """Remove the checkpoint file after a successful run."""
    cp = _checkpoint_path(output_path)
    if os.path.exists(cp):
        os.remove(cp)


# ---------------------------------------------------------------------------
# Pre-flight cost estimation
# ---------------------------------------------------------------------------


def _preflight_check(model: str, total: int, yes: bool) -> None:
    """Print cost estimate and prompt for confirmation unless *yes* is True."""
    estimate = estimate_batch_cost(model, total)
    print(f"\n  Model   : {model}")
    print(f"  Scenarios: {total}")
    print(f"  Est. cost: ${estimate:.3f} USD  (rough estimate only)\n")
    if yes:
        return
    answer = input("  Proceed? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        print("  Aborted.")
        sys.exit(0)


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


def _build_system_prompt(domain: str, batch_size: int, scenario_type: str) -> str:
    """Build the system prompt for a specific scenario type.

    Args:
        domain: Subject-matter domain string.
        batch_size: Number of scenarios to request (embedded in prompt).
        scenario_type: One of ``"factual"``, ``"procedural"``, ``"edge_case"``.

    Returns:
        The formatted system prompt string.
    """
    type_cfg = _SCENARIO_TYPES[scenario_type]
    return (
        f"You are an expert in {domain}. "
        f"Your objective is to create evaluation scenarios of type: "
        f"**{type_cfg['label']}**.\n\n"
        f"Type definition: {type_cfg['description']}\n\n"
        f"Output exactly {batch_size} scenarios of this type only. "
        "Do not mix types.\n"
        'Return the output strictly in JSON format using a single key "scenarios", '
        "which contains an array of objects.\n"
        "Each object MUST have the following keys:\n"
        '- "prompt": A realistic user query of this type.\n'
        '- "chosen": The correct, complete answer.\n'
        '- "rejected": A plausible but factually flawed answer — '
        "wrong sequence, wrong component, or subtle hallucination.\n"
        '- "type": The scenario type label (always '
        f'"{type_cfg["label"]}").\n'
    )


def main() -> None:
    """Entry point for golden evaluation set generation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate golden evaluation scenarios for domain QA"
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="Teacher model ID",
    )
    parser.add_argument(
        "--provider", default=None,
        help="API provider: 'openai', 'anthropic', or 'ollama' (auto-detected)",
    )
    parser.add_argument("--api-key", default=None, help="API key (falls back to env vars)")
    parser.add_argument(
        "--count", type=int, default=300,       # raised from 100
        help="Total number of scenarios to generate (recommended: 250-500)",
    )
    parser.add_argument(
        "--train-model",
        default=None,
        help=(
            "Model ID used to generate training data (e.g. 'gpt-4o'). "
            "When provided, the eval model must be from a different provider "
            "family to prevent contamination."
        ),
    )
    parser.add_argument(
        "--allow-same-family",
        action="store_true",
        help="Bypass the family-separation check (not recommended).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,   # raised from hardcoded 10
        help="Scenarios per API call (default 20; lower if hitting token limits)",
    )
    parser.add_argument("--output", default="golden_eval.jsonl", help="Output file path")
    parser.add_argument(
        "--domain", default=None,
        help="Subject-matter domain string (falls back to $DOMAIN env var, then 'technical documentation')",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip cost confirmation prompt",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing checkpoint and start fresh",
    )
    args = parser.parse_args()

    provider = args.provider or detect_provider(args.model)
    client = build_client(provider, api_key=args.api_key)

    # Contamination guard — enforce family separation
    if args.train_model and not args.allow_same_family:
        _check_family_separation(args.model, args.train_model)
    elif args.train_model and args.allow_same_family:
        print(
            "  [WARN] --allow-same-family passed. "
            "Eval results may not accurately reflect out-of-distribution performance."
        )
    elif args.allow_same_family:
        print(
            "  [INFO] --allow-same-family has no effect without --train-model."
        )
    else:
        print(
            "  [INFO] --train-model not specified. "
            "Family separation cannot be verified. "
            "Pass --train-model <model-id> to enforce it."
        )

    domain = args.domain or os.environ.get("DOMAIN", "technical documentation")

    # Pre-flight cost estimate
    _preflight_check(args.model, args.count, args.yes)

    # Allocate counts across types
    type_counts = _allocate_type_counts(args.count)
    logger.info(
        "Type allocation: %s",
        ", ".join(f"{k}={v}" for k, v in type_counts.items()),
    )

    # Load checkpoint
    scenarios: list[dict] = [] if args.no_resume else _load_checkpoint(args.output)
    already_by_type: dict[str, int] = {}
    for s in scenarios:
        t = s.get("type", "edge-case / diagnostic")
        already_by_type[t] = already_by_type.get(t, 0) + 1

    # Map type label back to key for checkpoint resume
    _label_to_key = {v["label"]: k for k, v in _SCENARIO_TYPES.items()}

    for type_key, type_cfg in _SCENARIO_TYPES.items():
        target = type_counts[type_key]
        already = already_by_type.get(type_cfg["label"], 0)
        remaining = target - already

        if remaining <= 0:
            logger.info(
                "[%s] Already complete (%d/%d) — skipping.",
                type_cfg["label"], already, target,
            )
            continue

        logger.info(
            "Generating %d %s scenarios (%d already cached)...",
            remaining, type_cfg["label"], already,
        )

        system_prompt = _build_system_prompt(domain, args.batch_size, type_key)
        consecutive_errors = 0
        type_scenarios: list[dict] = []

        while len(type_scenarios) < remaining:
            needed = remaining - len(type_scenarios)
            batch_sz = min(args.batch_size, needed)

            try:
                logger.info(
                    "  [%s] Batch... (%d/%d)",
                    type_cfg["label"], len(type_scenarios), remaining,
                )
                batch = generate_scenarios(
                    client, provider, args.model, system_prompt, domain, batch_sz,
                )
                if not batch:
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        logger.error("Too many empty batches for type %s. Stopping.", type_key)
                        break
                    continue

                # Tag each scenario with its type
                for s in batch:
                    s.setdefault("type", type_cfg["label"])

                type_scenarios.extend(batch)
                _append_checkpoint(args.output, batch)   # persist immediately
                consecutive_errors = 0
                time.sleep(0.5)   # brief pause to avoid burst rate limits

            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON in batch: %s", exc)
                consecutive_errors += 1
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Batch error: %s", exc)
                consecutive_errors += 1

            if consecutive_errors >= 5:
                break

        scenarios.extend(type_scenarios[:remaining])

    # Trim to exact count and save
    scenarios = scenarios[: args.count]

    if not scenarios:
        logger.error("No scenarios generated. Exiting.")
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + "\n")

    _clear_checkpoint(args.output)

    # Summary
    by_type: dict[str, int] = {}
    for s in scenarios:
        t = s.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    logger.info("=" * 50)
    logger.info("EVAL SET COMPLETE: %d scenarios saved to %s", len(scenarios), args.output)
    for t, n in sorted(by_type.items()):
        logger.info("  %-30s %d", t, n)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
