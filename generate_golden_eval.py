import os
import json
import argparse
from typing import Optional

# ==============================================================================
# Provider Detection (shared logic with teacher_model_synthesis.py)
# ==============================================================================
OLLAMA_BASE_URL = "http://localhost:11434/v1"

def detect_provider(model: str) -> str:
    """Auto-detect provider from model name."""
    lower = model.lower()
    if "claude" in lower:
        return "anthropic"
    if "gpt" in lower or "o1" in lower or "o3" in lower:
        return "openai"
    return "ollama"

def build_client(provider: str, api_key: Optional[str] = None, ollama_base_url: Optional[str] = None):
    """Build the appropriate API client for the given provider."""
    if provider == "openai":
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY env var or pass --api-key.")
        return OpenAI(api_key=key)
    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError("The 'anthropic' package is required. Run: uv pip install anthropic")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY env var or pass --api-key.")
        return anthropic.Anthropic(api_key=key)
    elif provider == "ollama":
        from openai import OpenAI
        url = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
        return OpenAI(api_key="ollama", base_url=url)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# ==============================================================================
# Generation Functions
# ==============================================================================
def generate_scenarios_openai(client, model: str, system_prompt: str, domain: str, batch_size: int = 10):
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=f"Generate exactly {batch_size} complex, edge-case {domain} scenarios focusing on multi-step reasoning. Output valid JSON.",
        text={"format": {"type": "json_object"}}
    )
    data = json.loads(response.output_text)
    return data.get("scenarios", [])

def generate_scenarios_anthropic(client, model: str, system_prompt: str, domain: str, batch_size: int = 10):
    user_prompt = (
        f"Generate exactly {batch_size} complex, edge-case {domain} scenarios focusing on multi-step reasoning.\n\n"
        "You MUST respond with valid JSON only, using a single key \"scenarios\" containing an array of objects.\n"
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

def generate_scenarios_ollama(client, model: str, system_prompt: str, domain: str, batch_size: int = 10):
    """Uses Ollama's OpenAI-compatible chat completions endpoint with JSON mode."""
    user_prompt = (
        f"Generate exactly {batch_size} complex, edge-case {domain} scenarios focusing on multi-step reasoning.\n\n"
        "You MUST respond with valid JSON only, using a single key \"scenarios\" containing an array of objects.\n"
        "Each object MUST have keys: \"prompt\", \"chosen\", \"rejected\".\n"
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
def main():
    parser = argparse.ArgumentParser(description="Generate golden evaluation scenarios for domain QA")
    parser.add_argument("--model", default="gpt-4o", help="Teacher model ID (e.g., gpt-4o, claude-3-5-sonnet-20241022, llama3.1:8b)")
    parser.add_argument("--provider", default=None, help="API provider: 'openai', 'anthropic', or 'ollama' (auto-detected from model name)")
    parser.add_argument("--api-key", default=None, help="API key (falls back to env vars; not needed for Ollama)")
    parser.add_argument("--count", type=int, default=100, help="Number of scenarios to generate (default: 100)")
    parser.add_argument("--output", default="golden_eval.jsonl", help="Output file path (default: golden_eval.jsonl)")
    args = parser.parse_args()

    provider = args.provider or detect_provider(args.model)
    client = build_client(provider, args.api_key)

    # The domain is configurable via env var or defaults to "technical documentation"
    domain = os.environ.get("DOMAIN", "technical documentation")

    system_prompt = f"""You are an expert in {domain}. Your objective is to create highly complex, edge-case scenarios that test deep domain knowledge and multi-step reasoning.
You must focus purely on multi-step reasoning questions that require deep diagnostic logic.

Output exactly 10 scenarios.
Return the output strictly in JSON format using a single key "scenarios", which contains an array of objects.
Each object MUST have the following keys:
- "prompt": A detailed user query describing a complex, multi-layered issue within the domain.
- "chosen": The correct, step-by-step diagnostic and resolution process.
- "rejected": A plausible but incorrect or factually flawed resolution that might mislead a user or cause further issues.
"""

    generate_fns = {
        "openai": generate_scenarios_openai,
        "anthropic": generate_scenarios_anthropic,
        "ollama": generate_scenarios_ollama,
    }
    generate_fn = generate_fns[provider]
    batch_size = 10

    print(f"Generating {args.count} complex {domain} scenarios using {provider}:{args.model}...")

    scenarios = []
    while len(scenarios) < args.count:
        try:
            print(f"Generating batch... ({len(scenarios)}/{args.count})")
            batch = generate_fn(client, args.model, system_prompt, domain, batch_size)
            if not batch:
                continue
            scenarios.extend(batch)
        except Exception as e:
            print(f"Error generating batch: {e}")

    # Trim to exactly target_count
    scenarios = scenarios[:args.count]

    # Save to JSONL
    print(f"Saving to {args.output}...")
    with open(args.output, "w") as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + "\n")

    print(f"Successfully generated and saved {len(scenarios)} scenarios to {args.output}.")

if __name__ == "__main__":
    main()
