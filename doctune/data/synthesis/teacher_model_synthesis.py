"""
teacher_model_synthesis.py — Synthetic data generation for SFT and DPO.

Generates high-quality question-answer pairs from document chunks using a
teacher model (OpenAI, Anthropic, or Ollama). Produces both "chosen"
(correct) and "rejected" (subtly flawed) responses for DPO alignment.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, ConfigDict

from doctune.utils.pricing import compute_model_usage_cost
from doctune.utils.provider_utils import build_client, detect_provider, retry_on_rate_limit

logger = logging.getLogger(__name__)

UsageMetrics = dict[str, int]


def _build_usage(input_tokens: int | None, output_tokens: int | None) -> UsageMetrics:
    """Normalize usage values to a consistent integer metrics dict."""
    return {
        "input_tokens": max(0, int(input_tokens or 0)),
        "output_tokens": max(0, int(output_tokens or 0)),
    }


def _extract_usage_from_openai_responses(response: object) -> UsageMetrics:
    """Extract usage metrics from an OpenAI Responses API object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return _build_usage(0, 0)
    return _build_usage(
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def _extract_usage_from_openai_chat(response: object) -> UsageMetrics:
    """Extract usage metrics from OpenAI-compatible Chat Completions responses."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return _build_usage(0, 0)
    return _build_usage(
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def _extract_usage_from_anthropic(response: object) -> UsageMetrics:
    """Extract usage metrics from Anthropic responses."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return _build_usage(0, 0)
    return _build_usage(
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def _split_usage_across_pairs(total_usage: UsageMetrics, pair_count: int) -> list[UsageMetrics]:
    """Distribute shared token usage across generated SFT pairs."""
    if pair_count <= 0:
        return []

    input_base, input_remainder = divmod(total_usage["input_tokens"], pair_count)
    output_base, output_remainder = divmod(total_usage["output_tokens"], pair_count)

    distributed: list[UsageMetrics] = []
    for index in range(pair_count):
        distributed.append({
            "input_tokens": input_base + (1 if index < input_remainder else 0),
            "output_tokens": output_base + (1 if index < output_remainder else 0),
        })
    return distributed

# ==============================================================================
# Pydantic Schemas for Strict API Formatting
# ==============================================================================


class SFTPair(BaseModel):
    """A single SFT question-answer pair."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    chosen: str


class SFTResponse(BaseModel):
    """Structured response containing multiple SFT pairs."""

    qa_pairs: list[SFTPair]


class DPOResponse(BaseModel):
    """Structured response containing a rejected answer for DPO."""

    model_config = ConfigDict(frozen=True)

    rejected: str


# ==============================================================================
# JSON-mode instruction fragments (shared by Anthropic & Ollama)
# ==============================================================================
_SFT_JSON_INSTRUCTION = (
    "\n\nYou MUST respond with valid JSON only, matching this exact schema:\n"
    '{"qa_pairs": [{"prompt": "...", "chosen": "..."}]}\n'
    "Do not include any text outside the JSON object."
)

_DPO_JSON_INSTRUCTION = (
    "\n\nYou MUST respond with valid JSON only, matching this exact schema:\n"
    '{"rejected": "..."}\n'
    "Do not include any text outside the JSON object."
)


# ==============================================================================
# Synthetic Data Generator Class
# ==============================================================================


class TeacherModelSynthesizer:
    """Orchestrates synthetic data generation via a teacher LLM.

    Supports OpenAI (structured outputs), Anthropic (JSON mode), and
    Ollama (OpenAI-compatible local API).

    Args:
        domain: Subject-matter domain (e.g. "medical devices").
        api_key: Explicit API key. Falls back to env vars.
        model: Teacher model identifier.
        provider: ``"openai"``, ``"anthropic"``, or ``"ollama"``. Auto-detected if ``None``.
        ollama_base_url: Custom Ollama server URL.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        domain: str = "technical documentation",
        api_key: str | None = None,
        model: str = "gpt-4o",
        provider: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider or detect_provider(model)
        self.domain = domain

        logger.info(
            "Initializing Teacher Model Synthesizer (%s:%s) for domain: '%s'",
            self.provider, self.model, domain,
        )
        print(
            f"Initializing Teacher Model Synthesizer "
            f"({self.provider}:{self.model}) for domain: '{domain}'..."
        )

        self.client = build_client(
            self.provider,
            api_key=api_key,
            base_url=ollama_base_url,
        )

    # --------------------------------------------------------------------------
    # OpenAI implementation (structured outputs via Pydantic)
    # --------------------------------------------------------------------------
    @retry_on_rate_limit()
    def _openai_generate_sft(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[list[dict], UsageMetrics]:
        """Generate SFT pairs via OpenAI structured outputs.

        Args:
            system_prompt: System instructions for the teacher model.
            user_prompt: User-facing prompt with the document chunk.

        Returns:
            Tuple of:
            - List of dicts with ``"prompt"`` and ``"chosen"`` keys.
            - Usage metrics for the API call.
        """
        response = self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=SFTResponse,
        )
        return (
            [pair.model_dump() for pair in response.output_parsed.qa_pairs],
            _extract_usage_from_openai_responses(response),
        )

    @retry_on_rate_limit()
    def _openai_generate_dpo(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str | None, UsageMetrics]:
        """Generate a DPO rejected response via OpenAI structured outputs.

        Args:
            system_prompt: System instructions for generating flawed responses.
            user_prompt: User-facing prompt with the original QA pair.

        Returns:
            Tuple of:
            - Rejected response string, or ``None`` on failure.
            - Usage metrics for the API call.
        """
        response = self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=DPOResponse,
        )
        return (
            response.output_parsed.rejected,
            _extract_usage_from_openai_responses(response),
        )

    # --------------------------------------------------------------------------
    # JSON-mode implementation (shared by Anthropic & Ollama)
    # --------------------------------------------------------------------------
    @retry_on_rate_limit()
    def _anthropic_raw_call(
        self, system_prompt: str, user_prompt: str, temperature: float,
    ) -> tuple[str, UsageMetrics]:
        """Make a raw Anthropic API call and return text with usage."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return response.content[0].text, _extract_usage_from_anthropic(response)

    @retry_on_rate_limit()
    def _ollama_raw_call(
        self, system_prompt: str, user_prompt: str, temperature: float,
    ) -> tuple[str, UsageMetrics]:
        """Make a raw Ollama API call and return text with usage."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return content, _extract_usage_from_openai_chat(response)

    def _json_mode_call(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.3,
    ) -> tuple[str, UsageMetrics]:
        """Dispatch a JSON-mode API call to the appropriate non-OpenAI provider.

        Args:
            system_prompt: System instructions (JSON instruction already appended).
            user_prompt: User message content.
            temperature: Sampling temperature.

        Returns:
            Tuple of raw JSON response text and usage metrics.
        """
        if self.provider == "anthropic":
            return self._anthropic_raw_call(system_prompt, user_prompt, temperature)
        return self._ollama_raw_call(system_prompt, user_prompt, temperature)

    def _json_mode_generate_sft(
        self, system_prompt: str, user_prompt: str,
    ) -> tuple[list[dict], UsageMetrics]:
        """Generate SFT pairs via JSON mode (Anthropic or Ollama)."""
        raw, usage = self._json_mode_call(
            system_prompt + _SFT_JSON_INSTRUCTION, user_prompt, temperature=0.3,
        )
        parsed = SFTResponse.model_validate_json(raw)
        return [pair.model_dump() for pair in parsed.qa_pairs], usage

    def _json_mode_generate_dpo(
        self, system_prompt: str, user_prompt: str,
    ) -> tuple[str | None, UsageMetrics]:
        """Generate a DPO rejected response via JSON mode (Anthropic or Ollama)."""
        raw, usage = self._json_mode_call(
            system_prompt + _DPO_JSON_INSTRUCTION, user_prompt, temperature=0.5,
        )
        parsed = DPOResponse.model_validate_json(raw)
        return parsed.rejected, usage

    # --------------------------------------------------------------------------
    # Prompt construction
    # --------------------------------------------------------------------------
    @staticmethod
    def _build_sft_user_prompt(markdown_chunk: str) -> str:
        """Build the two-stage user prompt for SFT pair generation.

        Stage 1 asks the model to identify and state the single most specific,
        actionable claim or procedure in the chunk.  Stage 2 asks it to generate
        all questions from that named anchor, preventing question drift across
        unrelated sub-topics.

        Args:
            markdown_chunk: Enriched chunk string including the source-context
                header produced by ``DoclingManualExtractor``.

        Returns:
            Formatted user prompt string ready for the teacher model.
        """
        return (
            f'Text Chunk:\n"""{markdown_chunk}"""\n\n'
            "Step 1 — Focus selection:\n"
            "Identify the single most specific, actionable claim, step sequence, or "
            "specification in the chunk above. State it in one sentence before generating "
            "any questions. Label it: FOCUS: <your one-sentence statement>\n\n"
            "Step 2 — Question generation:\n"
            "Using ONLY the fact or procedure you named in Step 1 as your anchor, "
            "generate exactly 3 Question-Answer pairs that each approach that same "
            "anchor from a different angle:\n"
            "  • Angle A — Symptom-based: the user describes a problem or observation.\n"
            "  • Angle B — Direct action: the user asks how to perform the procedure.\n"
            "  • Angle C — Edge-case or clarification: the user asks about a boundary "
            "condition, a prerequisite, or a detail that is easy to get wrong.\n\n"
            "All three questions must be answerable from the chunk text alone and must "
            "clearly refer to the same specific claim you named in Step 1."
        )

    # --------------------------------------------------------------------------
    # Public API (provider-agnostic)
    # --------------------------------------------------------------------------
    def generate_sft_pairs(
        self,
        markdown_chunk: str,
    ) -> list[tuple[dict[str, str], UsageMetrics]]:
        """Generate diverse SFT QA pairs from a document chunk.

        Args:
            markdown_chunk: A Docling-produced markdown text chunk.

        Returns:
            List of ``(pair, usage)`` tuples where:
            - ``pair`` has ``"prompt"`` and ``"chosen"`` keys.
            - ``usage`` contains per-pair ``input_tokens`` and ``output_tokens``.
            Returns an empty list if generation fails.
        """
        system_prompt = (
            f"You are an expert technical writer and data synthesizer for {self.domain}. "
            "Your task is to read documentation and generate highly accurate, "
            "realistic user questions "
            "and their corresponding step-by-step solutions.\n\n"
            "Strict Rules:\n"
            "1. Do NOT hallucinate or use external knowledge. The answer must be "
            "derived strictly from the text.\n"
            "2. If the text lacks actionable or 'how-to' information, output an empty array.\n"
            "3. Always generate exactly 3 questions: one symptom-based, one direct-action, "
            "and one edge-case or clarification question. All three must target the same "
            "specific claim you identified in Step 1 of the user prompt.\n"
            "4. The chunk header contains a [Source Context] and, when available, "
            "a [Section] breadcrumb showing exactly where in the document this text "
            "appears. Use the most specific heading in that breadcrumb to ground "
            "both the question and the answer — prefer section-specific language "
            "over generic phrases like 'the manual' or 'the document'.\n"
            "5. If a [Section] breadcrumb is present, at least one question must "
            "be answerable only by someone who knows which section it comes from "
            "(i.e. include a section-specific detail in the question itself)."
        )
        user_prompt = self._build_sft_user_prompt(markdown_chunk)

        generate = (
            self._openai_generate_sft
            if self.provider == "openai"
            else self._json_mode_generate_sft
        )

        try:
            pairs, total_usage = generate(system_prompt, user_prompt)
            usage_per_pair = _split_usage_across_pairs(total_usage, len(pairs))
            return list(zip(pairs, usage_per_pair, strict=False))
        except json.JSONDecodeError as e:
            logger.warning("SFT generation returned invalid JSON: %s", e)
            return []
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("SFT Generation Error: %s", e)
            return []

    def generate_dpo_rejection(
        self,
        prompt: str,
        chosen: str,
    ) -> tuple[str | None, UsageMetrics]:
        """Generate a subtly flawed 'rejected' answer for DPO alignment.

        Args:
            prompt: The user question from the SFT pair.
            chosen: The correct answer from the SFT pair.

        Returns:
            Tuple of:
            - A plausible but factually incorrect response, or ``None`` on failure.
            - DPO call usage metrics.
        """
        system_prompt = (
            "You are an AI safety and alignment expert. Your task is to generate a 'rejected' "
            "response for a technical support question.\n\n"
            "Strict Rules:\n"
            "1. It must look highly plausible and confidently written.\n"
            "2. It must contain a critical factual error "
            "(e.g., wrong sequence, wrong component, or subtle hallucination).\n"
            "3. It must not be cartoonishly evil or obvious; it must force the fine-tuned model "
            "to pay close attention.\n"
        )
        user_prompt = (
            f'User Question: "{prompt}"\nTrue Answer: "{chosen}"\n\n'
            "Generate the plausible but factually incorrect 'rejected' response."
        )

        generate = (
            self._openai_generate_dpo
            if self.provider == "openai"
            else self._json_mode_generate_dpo
        )

        try:
            return generate(system_prompt, user_prompt)
        except json.JSONDecodeError as e:
            logger.warning("DPO generation returned invalid JSON: %s", e)
            return None, _build_usage(0, 0)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("DPO Generation Error: %s", e)
            return None, _build_usage(0, 0)

    def process_chunk(self, markdown_chunk: str) -> list[dict]:
        """Orchestrate the full pipeline for a single chunk: SFT → DPO.

        Args:
            markdown_chunk: A Docling-produced markdown text chunk.

        Returns:
            List of complete tuples with ``"prompt"``, ``"chosen"``, and
            ``"rejected"`` keys plus ``"metadata"``:
            ``{"model", "input_tokens", "output_tokens", "cost_usd"}``.
        """
        sft_pairs = self.generate_sft_pairs(markdown_chunk)
        if not sft_pairs:
            return []

        final_tuples: list[dict] = []
        for pair, sft_usage in sft_pairs:
            rejected_text, dpo_usage = self.generate_dpo_rejection(
                pair["prompt"], pair["chosen"],
            )
            if rejected_text:
                input_tokens = sft_usage["input_tokens"] + dpo_usage["input_tokens"]
                output_tokens = sft_usage["output_tokens"] + dpo_usage["output_tokens"]
                cost_usd = compute_model_usage_cost(
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                final_tuples.append({
                    "prompt": pair["prompt"],
                    "chosen": pair["chosen"],
                    "rejected": rejected_text,
                    "metadata": {
                        "model": self.model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": round(cost_usd, 10),
                    },
                })

        return final_tuples


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teacher Model Synthetic Data Generator")
    parser.add_argument("--model", default="gpt-4o", help="Teacher model ID")
    parser.add_argument("--provider", default=None, help="API provider (auto-detected)")
    parser.add_argument("--domain", default="technical documentation", help="Subject-matter domain")
    args = parser.parse_args()

    synthesizer = TeacherModelSynthesizer(
        domain=args.domain, model=args.model, provider=args.provider
    )

    EXAMPLE_MARKDOWN_CHUNK = (
        "### [Source Context: Product User Guide]\n\n"
        "**Clearing a Paper Jam in the ADF**\n"
        "1. Lift the document feeder cover.\n"
        "2. Gently pull the jammed paper out of the rollers.\n"
        "3. Close the document feeder cover until it snaps into place."
    )

    print("Synthesizing SFT and DPO data...")
    generated_data = synthesizer.process_chunk(EXAMPLE_MARKDOWN_CHUNK)

    for i, data in enumerate(generated_data):
        print(f"\n--- Synthetic Tuple {i + 1} ---")
        print(json.dumps(data, indent=2))
