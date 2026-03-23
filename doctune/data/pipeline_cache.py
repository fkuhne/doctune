"""
pipeline_cache.py — Persistent, resumable caching for the dataset pipeline.

Stores extraction chunks and synthesis results on disk so that interrupted
runs can resume from the last completed step instead of restarting.

Cache layout::

    .cache/<domain>/
        chunks_<pdf_hash>.json        — extracted markdown chunks
        synthesis_<pdf_hash>.jsonl     — append-only synthesis results
        metadata_<pdf_hash>.json       — file hash, chunk count, timestamp
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)


class PipelineCache:
    """Manages per-domain cache files for the dataset build pipeline.

    Args:
        cache_dir: Root cache directory (default ``".cache"``).
        domain: Subject-matter domain used as the subdirectory name.
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        domain: str = "technical_documentation",
    ) -> None:
        safe_domain = re.sub(r"[^a-z0-9_]", "_", domain.lower().strip())
        self.cache_path = os.path.join(cache_dir, safe_domain)
        os.makedirs(self.cache_path, exist_ok=True)
        logger.info("Pipeline cache directory: %s", self.cache_path)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _chunks_path(self, pdf_hash: str) -> str:
        return os.path.join(self.cache_path, f"chunks_{pdf_hash}.json")

    def _synthesis_path(self, pdf_hash: str) -> str:
        return os.path.join(self.cache_path, f"synthesis_{pdf_hash}.jsonl")

    def _metadata_path(self, pdf_hash: str) -> str:
        return os.path.join(self.cache_path, f"metadata_{pdf_hash}.json")

    # ------------------------------------------------------------------
    # PDF identity
    # ------------------------------------------------------------------
    @staticmethod
    def get_pdf_hash(pdf_path: str) -> str:
        """Compute a SHA-256 hash of a PDF file's contents.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Hex-digest string (first 16 characters for brevity).
        """
        sha = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 16), b""):
                sha.update(block)
        digest: str = sha.hexdigest()
        return digest[:16]

    # ------------------------------------------------------------------
    # Chunk caching (Docling extraction results)
    # ------------------------------------------------------------------
    def has_chunks(self, pdf_hash: str) -> bool:
        """Check whether cached chunks exist for a given PDF hash."""
        return os.path.isfile(self._chunks_path(pdf_hash))

    def load_chunks(self, pdf_hash: str) -> list[str]:
        """Load cached enriched chunks from disk.

        Args:
            pdf_hash: The PDF file hash.

        Returns:
            List of enriched markdown chunk strings.
        """
        path = self._chunks_path(pdf_hash)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_chunks(
        self,
        pdf_hash: str,
        chunks: list[str],
        pdf_path: str,
    ) -> None:
        """Persist extracted chunks and metadata to disk.

        Args:
            pdf_hash: The PDF file hash.
            chunks: List of enriched markdown chunk strings.
            pdf_path: Original PDF path (stored in metadata for reference).
        """
        with open(self._chunks_path(pdf_hash), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        metadata = {
            "pdf_hash": pdf_hash,
            "pdf_path": os.path.abspath(pdf_path),
            "chunk_count": len(chunks),
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(self._metadata_path(pdf_hash), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Cached %d chunks for %s (%s)", len(chunks), pdf_path, pdf_hash,
        )

    # ------------------------------------------------------------------
    # Synthesis caching (per-chunk, append-only)
    # ------------------------------------------------------------------
    def get_completed_chunk_indices(self, pdf_hash: str) -> set[int]:
        """Return the set of chunk indices that have already been synthesized.

        Args:
            pdf_hash: The PDF file hash.

        Returns:
            Set of integer chunk indices with cached results.
        """
        path = self._synthesis_path(pdf_hash)
        completed: set[int] = set()

        if not os.path.isfile(path):
            return completed

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed.add(record["chunk_index"])
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Skipping malformed synthesis cache line")

        return completed

    def append_synthesis_result(
        self,
        pdf_hash: str,
        chunk_index: int,
        results: list[dict],
    ) -> None:
        """Append synthesis results for a single chunk to the cache.

        Each call writes one JSON line, making the file safe for append
        even after an unexpected interruption.

        Args:
            pdf_hash: The PDF file hash.
            chunk_index: Zero-based index of the processed chunk.
            results: List of QA-tuple dicts produced by the synthesizer
                (may be empty if the chunk yielded no actionable data).
        """
        record = {
            "chunk_index": chunk_index,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(self._synthesis_path(pdf_hash), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all_synthesis_results(self, pdf_hash: str) -> list[dict]:
        """Load all cached synthesis QA tuples for a given PDF.

        Args:
            pdf_hash: The PDF file hash.

        Returns:
            Flat list of QA-tuple dicts from all cached chunks.
        """
        path = self._synthesis_path(pdf_hash)
        all_results: list[dict] = []

        if not os.path.isfile(path):
            return all_results

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    all_results.extend(record.get("results", []))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed synthesis cache line")

        return all_results
