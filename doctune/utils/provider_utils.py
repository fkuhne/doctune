"""
provider_utils.py — Shared API provider utilities for doctune.

Centralizes provider detection, client construction, and retry logic
for OpenAI, Anthropic, and Ollama backends.
"""

from __future__ import annotations

import functools
import os
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"

_OPENAI_KEYWORDS = {"gpt", "o1", "o3", "o4"}
_ANTHROPIC_KEYWORDS = {"claude"}


# ==============================================================================
# Provider Detection
# ==============================================================================
def detect_provider(model: str) -> str:
    """Auto-detect the API provider from a model identifier.

    Args:
        model: Model name string (e.g. ``"gpt-4o"``, ``"claude-3-5-sonnet-20241022"``,
            ``"llama3.1:8b"``).

    Returns:
        One of ``"openai"``, ``"anthropic"``, or ``"ollama"``.
    """
    lower = model.lower()
    if any(kw in lower for kw in _ANTHROPIC_KEYWORDS):
        return "anthropic"
    if any(kw in lower for kw in _OPENAI_KEYWORDS):
        return "openai"
    # If it doesn't look like a cloud model, assume Ollama (local)
    return "ollama"


# ==============================================================================
# Client Construction
# ==============================================================================
def build_client(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Build the appropriate API client for the given provider.

    Args:
        provider: One of ``"openai"``, ``"anthropic"``, or ``"ollama"``.
        api_key: Explicit API key. Falls back to environment variables
            (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``). Not needed for Ollama.
        base_url: Custom base URL (primarily for Ollama).

    Returns:
        An initialized API client (``openai.OpenAI`` or ``anthropic.Anthropic``).

    Raises:
        ValueError: If the provider is unsupported or a required API key is missing.
        ImportError: If the ``anthropic`` package is not installed.
    """
    if provider == "openai":
        # pylint: disable=import-outside-toplevel
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key=."
            )
        return OpenAI(api_key=key, base_url=base_url)

    if provider == "anthropic":
        try:
            # pylint: disable=import-outside-toplevel
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for Anthropic models. "
                "Run: uv pip install anthropic"
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key=."
            )
        return anthropic.Anthropic(api_key=key)

    if provider == "ollama":
        # pylint: disable=import-outside-toplevel
        from openai import OpenAI

        url = base_url or os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
        # api_key is required by the SDK but ignored by Ollama
        return OpenAI(api_key="ollama", base_url=url)

    raise ValueError(
        f"Unsupported provider: {provider!r}. Use 'openai', 'anthropic', or 'ollama'."
    )


# ==============================================================================
# Retry Decorator for Rate Limits
# ==============================================================================
def retry_on_rate_limit(
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Any:
    """Decorator that retries a function on API rate-limit errors.

    Uses exponential backoff (``base_delay * 2^attempt``).

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before the first retry.

    Returns:
        Decorated function with automatic retry behavior.
    """

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc: # pylint: disable=broad-exception-caught
                    # Check if this is a rate-limit error we should retry
                    if _is_rate_limit_error(exc) and attempt < max_retries:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "Rate limit hit in %s (attempt %d/%d). "
                            "Retrying in %.1fs...",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            delay,
                        )
                        print(
                            f"  [Rate limit — retrying in {delay:.0f}s "
                            f"(attempt {attempt + 1}/{max_retries})]"
                        )
                        time.sleep(delay)
                        last_exception = exc
                    else:
                        raise
            # Should not reach here, but just in case
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit error from any supported provider.

    Args:
        exc: The exception to inspect.

    Returns:
        ``True`` if the exception represents a rate-limit (HTTP 429) error.
    """
    exc_type_name = type(exc).__name__

    # OpenAI SDK
    if exc_type_name == "RateLimitError":
        return True

    # Anthropic SDK
    if exc_type_name in ("RateLimitError", "APIStatusError"):
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True

    # Generic httpx / requests
    if hasattr(exc, "response"):
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True

    return False
