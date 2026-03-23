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

from doctune.utils.provider_utils import build_client, detect_provider, retry_on_rate_limit

logger = logging.getLogger(__name__)

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
    def _openai_generate_sft(self, system_prompt: str, user_prompt: str) -> list[dict]:
        """Generate SFT pairs via OpenAI structured outputs.

        Args:
            system_prompt: System instructions for the teacher model.
            user_prompt: User-facing prompt with the document chunk.

        Returns:
            List of dicts with ``"prompt"`` and ``"chosen"`` keys.
        """
        response = self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=SFTResponse,
        )
        return [pair.model_dump() for pair in response.output_parsed.qa_pairs]

    @retry_on_rate_limit()
    def _openai_generate_dpo(self, system_prompt: str, user_prompt: str) -> str | None:
        """Generate a DPO rejected response via OpenAI structured outputs.

        Args:
            system_prompt: System instructions for generating flawed responses.
            user_prompt: User-facing prompt with the original QA pair.

        Returns:
            The rejected response string, or ``None`` on failure.
        """
        response = self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=DPOResponse,
        )
        return response.output_parsed.rejected

    # --------------------------------------------------------------------------
    # JSON-mode implementation (shared by Anthropic & Ollama)
    # --------------------------------------------------------------------------
    @retry_on_rate_limit()
    def _anthropic_raw_call(
        self, system_prompt: str, user_prompt: str, temperature: float,
    ) -> str:
        """Make a raw Anthropic API call and return the text content."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return response.content[0].text

    @retry_on_rate_limit()
    def _ollama_raw_call(
        self, system_prompt: str, user_prompt: str, temperature: float,
    ) -> str:
        """Make a raw Ollama API call via the OpenAI-compatible endpoint."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _json_mode_call(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.3,
    ) -> str:
        """Dispatch a JSON-mode API call to the appropriate non-OpenAI provider.

        Args:
            system_prompt: System instructions (JSON instruction already appended).
            user_prompt: User message content.
            temperature: Sampling temperature.

        Returns:
            Raw JSON text response from the model.
        """
        if self.provider == "anthropic":
            return self._anthropic_raw_call(system_prompt, user_prompt, temperature)
        return self._ollama_raw_call(system_prompt, user_prompt, temperature)

    def _json_mode_generate_sft(
        self, system_prompt: str, user_prompt: str,
    ) -> list[dict]:
        """Generate SFT pairs via JSON mode (Anthropic or Ollama)."""
        raw = self._json_mode_call(
            system_prompt + _SFT_JSON_INSTRUCTION, user_prompt, temperature=0.3,
        )
        parsed = SFTResponse.model_validate_json(raw)
        return [pair.model_dump() for pair in parsed.qa_pairs]

    def _json_mode_generate_dpo(
        self, system_prompt: str, user_prompt: str,
    ) -> str | None:
        """Generate a DPO rejected response via JSON mode (Anthropic or Ollama)."""
        raw = self._json_mode_call(
            system_prompt + _DPO_JSON_INSTRUCTION, user_prompt, temperature=0.5,
        )
        parsed = DPOResponse.model_validate_json(raw)
        return parsed.rejected

    # --------------------------------------------------------------------------
    # Public API (provider-agnostic)
    # --------------------------------------------------------------------------
    def generate_sft_pairs(self, markdown_chunk: str) -> list[dict]:
        """Generate diverse SFT QA pairs from a document chunk.

        Args:
            markdown_chunk: A Docling-produced markdown text chunk.

        Returns:
            List of dicts with ``"prompt"`` and ``"chosen"`` keys.
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
            "3. Generate questions from multiple angles (e.g., direct action, "
            "symptom-based, clarification).\n"
            "4. Always reference the specific source context in the chosen response."
        )
        user_prompt = (
            f'Text Chunk:\n"""{markdown_chunk}"""\n\n'
            'Generate 2 to 3 Question-Answer pairs.'
        )

        generate = (
            self._openai_generate_sft
            if self.provider == "openai"
            else self._json_mode_generate_sft
        )

        try:
            return generate(system_prompt, user_prompt)
        except json.JSONDecodeError as e:
            logger.warning("SFT generation returned invalid JSON: %s", e)
            return []
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("SFT Generation Error: %s", e)
            return []

    def generate_dpo_rejection(self, prompt: str, chosen: str) -> str | None:
        """Generate a subtly flawed 'rejected' answer for DPO alignment.

        Args:
            prompt: The user question from the SFT pair.
            chosen: The correct answer from the SFT pair.

        Returns:
            A plausible but factually incorrect response, or ``None`` on failure.
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
            return None
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("DPO Generation Error: %s", e)
            return None

    def process_chunk(self, markdown_chunk: str) -> list[dict]:
        """Orchestrate the full pipeline for a single chunk: SFT → DPO.

        Args:
            markdown_chunk: A Docling-produced markdown text chunk.

        Returns:
            List of complete tuples with ``"prompt"``, ``"chosen"``, and
            ``"rejected"`` keys.
        """
        sft_pairs = self.generate_sft_pairs(markdown_chunk)
        if not sft_pairs:
            return []

        final_tuples: list[dict] = []
        for pair in sft_pairs:
            rejected_text = self.generate_dpo_rejection(pair["prompt"], pair["chosen"])
            if rejected_text:
                final_tuples.append({
                    "prompt": pair["prompt"],
                    "chosen": pair["chosen"],
                    "rejected": rejected_text,
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
