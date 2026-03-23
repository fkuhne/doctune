"""
build_dataset.py — Phase 2 orchestrator for training data generation.

Scans a directory of PDFs, loads cached extraction chunks (produced by
``extract_dataset.py``), synthesizes SFT and DPO training data using a
teacher model, deduplicates via cosine similarity, and outputs a single
``alignment_dataset.jsonl`` file.

Intermediate results are persisted to ``.cache/<domain>/`` so that
interrupted runs can resume from the last completed chunk.

Usage::

    python build_dataset.py [--model MODEL] [--input-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from doctune.data.extraction.pdf_extractor import DoclingManualExtractor
from doctune.data.pipeline.pipeline_cache import PipelineCache
from doctune.data.pipeline.pipeline_utils import (
    add_common_cli_args,
    discover_pdfs,
    extract_chunks_cached,
    extract_device_context,
    init_extractor_and_cache,
)
from doctune.data.synthesis.deduplicate_dataset import DatasetFilter
from doctune.data.synthesis.teacher_model_synthesis import TeacherModelSynthesizer

logger = logging.getLogger(__name__)


@dataclass
class _BuildStats:  # pylint: disable=too-few-public-methods
    """Accumulates pipeline-wide statistics during the build loop."""

    total_chunks_processed: int = 0
    total_pairs_generated: int = 0
    skipped_chunks: int = 0
    pdf_count: int = 0

    def log_summary(self, unique_pairs: int) -> None:
        """Print the final pipeline summary banner."""
        print("\n============================================================")
        print("PIPELINE COMPLETE.")
        print(f"Total PDF Manuals Processed: {self.pdf_count}")
        print(f"Total Semantic Chunks Analyzed: {self.total_chunks_processed}")
        print(f"Total QA Pairs Generated: {self.total_pairs_generated}")
        print(f"Total Unique QA Pairs Retained: {unique_pairs}")
        if self.skipped_chunks:
            print(f"Chunks Skipped Due to Errors: {self.skipped_chunks}")
        print("============================================================")


class DatasetBuilder:  # pylint: disable=too-few-public-methods
    """Master orchestrator for the data curation pipeline.

    Coordinates PDF extraction (with caching), teacher-model synthesis,
    and semantic deduplication into a single end-to-end workflow.
    Supports persistent caching so that interrupted runs can resume from
    the last completed chunk.

    Args:
        input_dir: Directory containing PDF manuals.
        output_file: Path for the output JSONL dataset.
        model: Teacher model identifier.
        provider: API provider (auto-detected if ``None``).
        domain: Subject-matter domain for prompt context.
        extractor: Pre-initialised Docling extractor.
        cache: Optional pipeline cache (``None`` disables caching).
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_dir: str,
        output_file: str,
        model: str,
        provider: str | None,
        domain: str,
        extractor: DoclingManualExtractor | None,
        cache: PipelineCache | None,
    ) -> None:
        self.input_dir = input_dir
        self.output_file = output_file
        self.extractor = extractor
        self.cache = cache

        print("--- INITIALIZING PHASE 2 PIPELINE ---")
        self.synthesizer = TeacherModelSynthesizer(
            domain=domain,
            model=model,
            provider=provider,
        )
        self.filter = DatasetFilter(similarity_threshold=0.85)

    # ------------------------------------------------------------------
    # Cache helper
    # ------------------------------------------------------------------
    def _cache_synthesis(
        self, pdf_hash: str | None, chunk_index: int, results: list[dict],
    ) -> None:
        """Persist synthesis results to the cache if caching is enabled."""
        if self.cache is not None and pdf_hash is not None:
            self.cache.append_synthesis_result(pdf_hash, chunk_index, results)

    # ------------------------------------------------------------------
    # Main build loop
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Execute the full data curation pipeline.

        Iterates through all PDFs in ``input_dir``, extracts chunks
        (using the cache when available), synthesizes SFT/DPO pairs,
        deduplicates, and saves the result.  Individual chunk failures
        are logged and skipped without stopping the pipeline.  Caching
        ensures that completed work survives interruptions.
        """
        pdf_files = discover_pdfs(self.input_dir)

        if not pdf_files:
            print(f"CRITICAL: No PDFs found in directory '{self.input_dir}'.")
            return

        stats = _BuildStats(pdf_count=len(pdf_files))
        print(f"Found {len(pdf_files)} manuals to process. Starting pipeline...\n")

        for i, pdf_path in enumerate(pdf_files):
            device_context = extract_device_context(pdf_path)
            print("============================================================")
            print(f"Processing Manual {i + 1}/{len(pdf_files)}: {device_context}")
            print("============================================================")

            try:
                self._process_single_pdf(pdf_path, device_context, stats)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Critical error processing %s: %s", pdf_path, e)
                print(f"\nCRITICAL ERROR processing {pdf_path}: {e}")
                print("Skipping to the next manual to preserve pipeline execution...")

        stats.log_summary(unique_pairs=len(self.filter.accepted_data))
        self.filter.save_dataset(self.output_file)

    def _process_single_pdf(
        self,
        pdf_path: str,
        device_context: str,
        stats: _BuildStats,
    ) -> None:
        """Extract, synthesize, and deduplicate chunks for one PDF.

        Args:
            pdf_path: Path to the PDF file.
            device_context: Human-readable document label.
            stats: Mutable stats accumulator.
        """
        # Step 1: Extract and Chunk (with caching)
        enriched_chunks = extract_chunks_cached(
            pdf_path, device_context, self.extractor, self.cache,
        )

        if not enriched_chunks:
            print("  No chunks extracted — skipping this manual.")
            return

        # Determine which chunks were already synthesized
        pdf_hash: str | None = None
        completed_indices: set[int] = set()
        if self.cache is not None:
            pdf_hash = self.cache.get_pdf_hash(pdf_path)
            completed_indices = self.cache.get_completed_chunk_indices(pdf_hash)

            # Re-ingest previously cached synthesis results into the filter
            if completed_indices:
                cached_results = self.cache.load_all_synthesis_results(pdf_hash)
                for qa_tuple in cached_results:
                    self.filter.process_new_pair(qa_tuple)
                    stats.total_pairs_generated += 1

                print(
                    f"  [RESUMING] {len(completed_indices)}/{len(enriched_chunks)} "
                    f"chunks already processed — resuming from next."
                )

        # Step 2: Iterate through every chunk
        for j, chunk in enumerate(enriched_chunks):
            # Skip chunks already completed in a previous run
            if j in completed_indices:
                stats.total_chunks_processed += 1
                continue

            stats.total_chunks_processed += 1
            print(
                f"  -> Synthesizing chunk {j + 1}/{len(enriched_chunks)}...",
                end=" ", flush=True,
            )

            try:
                generated_tuples = self.synthesizer.process_chunk(chunk)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Chunk %d/%d failed in %s: %s",
                    j + 1, len(enriched_chunks), device_context, e,
                )
                print(f"[ERROR: {type(e).__name__} — skipped]")
                stats.skipped_chunks += 1
                self._cache_synthesis(pdf_hash, j, [])
                continue

            # Cache synthesis results immediately (even empty ones)
            self._cache_synthesis(
                pdf_hash, j, generated_tuples if generated_tuples else [],
            )

            if not generated_tuples:
                print("[Skipped: No actionable data]")
                continue

            # Step 3: Filter and Deduplicate
            kept_count = 0
            for qa_tuple in generated_tuples:
                stats.total_pairs_generated += 1
                if self.filter.process_new_pair(qa_tuple):
                    kept_count += 1

            print(f"[Generated {len(generated_tuples)} | Kept {kept_count}]")

            # Brief pause to avoid API rate limits
            time.sleep(1)


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Build training dataset from PDF manuals"
    )
    parser.add_argument("--model", default="gpt-4o", help="Teacher model ID")
    parser.add_argument("--provider", default=None, help="API provider (auto-detected)")
    parser.add_argument(
        "--output", default="alignment_dataset.jsonl", help="Output JSONL file path"
    )
    add_common_cli_args(parser)
    args = parser.parse_args()

    Path(args.input_dir).mkdir(parents=True, exist_ok=True)

    cli_extractor, cli_cache = init_extractor_and_cache(args, init_extractor=False)

    builder = DatasetBuilder(
        input_dir=args.input_dir,
        output_file=args.output,
        model=args.model,
        provider=args.provider,
        domain=args.domain,
        extractor=cli_extractor,
        cache=cli_cache,
    )

    builder.build()
