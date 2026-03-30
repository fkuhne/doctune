"""
deduplicate_dataset.py — Semantic deduplication for training data.

Uses sentence-transformers embeddings and cosine similarity to filter
out near-duplicate QA pairs, ensuring dataset diversity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Return the shared sentence-transformer model, loading it on first call."""
    global _embedding_model  # noqa: PLW0603
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

_REQUIRED_KEYS = ("prompt", "chosen", "rejected")


class DatasetFilter:
    """Filters training data using schema validation and cosine deduplication.

    Maintains an in-memory vector store of accepted prompt embeddings.
    New pairs are compared against all previously accepted pairs, and
    discarded if their cosine similarity exceeds the threshold.

    Args:
        similarity_threshold: Maximum cosine similarity allowed between
            any two accepted prompts (default ``0.92``).
    """

    def __init__(self, similarity_threshold: float = 0.92) -> None:
        print("Loading Embedding Model...")
        self.model = _get_embedding_model()
        self.similarity_threshold = similarity_threshold

        self.accepted_data: list[dict] = []
        self.accepted_embeddings: torch.Tensor | None = None

    @staticmethod
    def validate_schema(qa_pair: dict) -> bool:
        """Validate that a QA tuple has the required non-empty string fields.

        Args:
            qa_pair: A dict expected to contain ``"prompt"``, ``"chosen"``,
                and ``"rejected"`` string keys.

        Returns:
            ``True`` if all required fields are present, are strings, and
            are non-empty after stripping whitespace.
        """
        return all(
            isinstance(qa_pair.get(key), str) and qa_pair[key].strip()
            for key in _REQUIRED_KEYS
        )

    def process_new_pair(self, qa_pair: dict) -> bool:
        """Validate, deduplicate, and optionally accept a new QA pair.

        Args:
            qa_pair: A dict with ``"prompt"``, ``"chosen"``, ``"rejected"`` keys.

        Returns:
            ``True`` if the pair was accepted, ``False`` if dropped.
        """
        if not self.validate_schema(qa_pair):
            logger.info("Dropped: Failed schema validation (missing keys or empty strings).")
            return False

        new_prompt = qa_pair["prompt"]
        new_embedding = self.model.encode(new_prompt, convert_to_tensor=True)

        # First item is always accepted
        if self.accepted_embeddings is None:
            self.accepted_embeddings = new_embedding.unsqueeze(0)
            self.accepted_data.append(qa_pair)
            return True

        # Cosine similarity against all previously accepted prompts
        cosine_scores = util.cos_sim(new_embedding, self.accepted_embeddings)[0]
        max_similarity: float = torch.max(cosine_scores).item()

        if max_similarity > self.similarity_threshold:
            # ── Structured audit log ───────────────────────────────────────
            matched_idx: int = int(torch.argmax(cosine_scores).item())
            matched_prompt: str = self.accepted_data[matched_idx].get("prompt", "")
            logger.info(
                "DEDUP_DROP | score=%.4f | threshold=%.2f | "
                "dropped=%r | matched=%r",
                max_similarity,
                self.similarity_threshold,
                new_prompt[:120],
                matched_prompt[:120],
            )
            # ── End audit log ──────────────────────────────────────────────
            return False

        # Accept and add to vector store
        self.accepted_embeddings = torch.cat(
            (self.accepted_embeddings, new_embedding.unsqueeze(0)), dim=0
        )
        self.accepted_data.append(qa_pair)
        return True

    def save_dataset(self, output_path: str) -> None:
        """Save the filtered dataset to a JSONL file.

        Args:
            output_path: Destination file path.
        """
        content = "\n".join(json.dumps(item) for item in self.accepted_data) + "\n"
        Path(output_path).write_text(content, encoding="utf-8")
        logger.info("Saved %d QA pairs to %s", len(self.accepted_data), output_path)
        print(f"Saved {len(self.accepted_data)} highly diverse QA pairs to {output_path}.")


class ChunkFilter:
    """Deduplicates source chunks by cosine similarity before synthesis.

    Maintains an in-memory vector store of accepted chunk embeddings.
    A chunk is rejected if its cosine similarity to any previously accepted
    chunk exceeds ``similarity_threshold``.

    This gate fires *before* teacher-model synthesis, so duplicate source
    material never reaches the API.  It is intentionally more permissive
    than the downstream prompt-level ``DatasetFilter`` (lower threshold)
    because raw chunk text scores higher surface similarity than generated
    questions, even when the underlying content is genuinely distinct.

    Args:
        similarity_threshold: Maximum cosine similarity allowed between
            any two accepted chunks (default ``0.82``).
    """

    def __init__(self, similarity_threshold: float = 0.82) -> None:
        self.model = _get_embedding_model()
        self.similarity_threshold = similarity_threshold
        self.accepted_embeddings: torch.Tensor | None = None
        self._accepted_count: int = 0
        self._rejected_count: int = 0

    def is_duplicate(self, chunk_text: str) -> bool:
        """Return ``True`` if *chunk_text* is too similar to a previously seen chunk.

        Side-effect: if the chunk is accepted (not a duplicate), its embedding
        is added to the internal vector store so future chunks are compared
        against it.

        Args:
            chunk_text: Raw markdown chunk string from the extractor.

        Returns:
            ``True`` if the chunk should be skipped, ``False`` if it is novel
            enough to proceed to synthesis.
        """
        embedding = self.model.encode(chunk_text, convert_to_tensor=True)

        if self.accepted_embeddings is None:
            self.accepted_embeddings = embedding.unsqueeze(0)
            self._accepted_count += 1
            return False

        scores = util.cos_sim(embedding, self.accepted_embeddings)[0]
        max_similarity: float = torch.max(scores).item()

        if max_similarity > self.similarity_threshold:
            logger.info(
                "Chunk duplicate dropped (similarity=%.3f > threshold=%.2f).",
                max_similarity,
                self.similarity_threshold,
            )
            self._rejected_count += 1
            return True

        self.accepted_embeddings = torch.cat(
            (self.accepted_embeddings, embedding.unsqueeze(0)), dim=0,
        )
        self._accepted_count += 1
        return False

    def log_summary(self) -> None:
        """Print accepted/rejected counts to stdout."""
        total = self._accepted_count + self._rejected_count
        print(
            f"  [Chunk dedup] {self._accepted_count}/{total} chunks passed "
            f"(threshold={self.similarity_threshold})."
        )


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    qa_filter = DatasetFilter(similarity_threshold=0.92)

    synthetic_outputs = [
        {
            "prompt": "How do I clear a paper jam in the automatic document feeder?",
            "chosen": "Lift the cover and gently pull the paper...",
            "rejected": "Yank the paper out forcefully...",
        },
        {
            "prompt": "What is the best way to fix a document feeder jam?",
            "chosen": "Open the top hatch and roll the paper out...",
            "rejected": "Turn the device upside down and shake it...",
        },
        {
            "prompt": "Why is my device not connecting to the 5GHz Wi-Fi network?",
            "chosen": "Ensure your router is broadcasting a 2.4GHz band, as "
                      "many older devices do not support 5GHz...",
            "rejected": "Devices do not use Wi-Fi, you must use a USB cable.",
        },
        {
            "prompt": "",  # Will trigger schema failure
            "chosen": "Some answer",
            "rejected": "Some wrong answer",
        },
    ]

    print("--- Starting Dataset Deduplication Pipeline ---")
    for pair in synthetic_outputs:
        qa_filter.process_new_pair(pair)

    qa_filter.save_dataset("alignment_dataset.jsonl")
