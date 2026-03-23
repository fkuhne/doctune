"""
deduplicate_dataset.py — Semantic deduplication for training data.

Uses sentence-transformers embeddings and cosine similarity to filter
out near-duplicate QA pairs, ensuring dataset diversity.
"""

from __future__ import annotations

import json
import logging

import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class DatasetFilter:
    """Filters training data using schema validation and cosine deduplication.

    Maintains an in-memory vector store of accepted prompt embeddings.
    New pairs are compared against all previously accepted pairs, and
    discarded if their cosine similarity exceeds the threshold.

    Args:
        similarity_threshold: Maximum cosine similarity allowed between
            any two accepted prompts (default ``0.85``).
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        print("Loading Embedding Model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_threshold = similarity_threshold

        self.accepted_data: list[dict] = []
        self.accepted_embeddings: torch.Tensor | None = None

    def validate_schema(self, qa_pair: dict) -> bool:
        """Validate that a QA tuple has the required non-empty fields.

        Args:
            qa_pair: A dict expected to contain ``"prompt"``, ``"chosen"``,
                and ``"rejected"`` string keys.

        Returns:
            ``True`` if all required fields are present and non-empty.
        """
        required_keys = ["prompt", "chosen", "rejected"]

        if not all(key in qa_pair for key in required_keys):
            return False

        return all(
            isinstance(qa_pair[key], str) and len(qa_pair[key].strip()) > 0
            for key in required_keys
        )

    def process_new_pair(self, qa_pair: dict) -> bool:
        """Validate, deduplicate, and optionally accept a new QA pair.

        Args:
            qa_pair: A dict with ``"prompt"``, ``"chosen"``, ``"rejected"`` keys.

        Returns:
            ``True`` if the pair was accepted, ``False`` if it was dropped
            (schema failure or too similar to an existing pair).
        """
        if not self.validate_schema(qa_pair):
            print("Dropped: Failed schema validation (missing keys or empty strings).")
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
            print(f"Dropped: Semantic similarity too high ({max_similarity:.2f}) -> '{new_prompt}'")
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
        print(f"\nSaving {len(self.accepted_data)} highly diverse QA pairs to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.accepted_data:
                f.write(json.dumps(item) + "\n")
        print("Dataset successfully compiled.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    qa_filter = DatasetFilter(similarity_threshold=0.85)

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
