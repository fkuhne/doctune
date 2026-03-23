"""
pipeline_utils.py — Shared helpers for the dataset-building pipeline.

Provides reusable functions for PDF discovery, filename-to-context conversion,
cached chunk extraction, CLI argument registration, and extractor/cache
initialization.  Used by both ``extract_dataset.py`` and ``build_dataset.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from doctune.data.extraction.pdf_extractor import DoclingManualExtractor
from doctune.data.pipeline.pipeline_cache import PipelineCache

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------
def extract_device_context(filename: str) -> str:
    """Convert a PDF filename into a clean source-context label.

    Args:
        filename: Path to the PDF file (e.g. ``"product_user_guide.pdf"``).

    Returns:
        A title-cased context label (e.g. ``"Product User Guide"``).
    """
    stem = Path(filename).stem
    clean_name = stem.replace("_", " ").replace("-", " ").title()
    clean_name = clean_name.replace(" Manual", "").replace(" User Guide", "")
    return clean_name


def discover_pdfs(input_dir: str) -> list[str]:
    """Return a sorted list of PDF paths inside *input_dir*.

    Args:
        input_dir: Directory to scan.

    Returns:
        Sorted list of ``*.pdf`` file paths (as strings), possibly empty.
    """
    return sorted(str(p) for p in Path(input_dir).glob("*.pdf"))


# ------------------------------------------------------------------
# Cached extraction
# ------------------------------------------------------------------
def extract_chunks_cached(
    pdf_path: str,
    device_context: str,
    extractor: DoclingManualExtractor | None,
    cache: PipelineCache | None,
) -> list[str]:
    """Extract chunks from a PDF, using the cache when available.

    Args:
        pdf_path: Path to the PDF file.
        device_context: Human-readable document label.
        extractor: Initialised Docling extractor instance.
        cache: Optional :class:`PipelineCache` (``None`` disables caching).

    Returns:
        List of enriched markdown chunk strings.
    """
    pdf_hash: str | None = None

    if cache is not None:
        pdf_hash = cache.get_pdf_hash(pdf_path)

        if cache.has_chunks(pdf_hash):
            chunks = cache.load_chunks(pdf_hash)
            logger.info(
                "[CACHE HIT] Loaded %d chunks (hash: %s)", len(chunks), pdf_hash,
            )
            print(
                f"  [CACHE HIT] Loaded {len(chunks)} chunks from cache "
                f"(hash: {pdf_hash})"
            )
            return chunks

    # No cache hit — run Docling extraction
    if extractor is None:
        logger.warning("Cache miss for %s and extraction is disabled.", pdf_path)
        print(f"  [ERROR] Cache miss for {pdf_path} and extraction is disabled. Skipping.")
        return []

    enriched_chunks = extractor.process_manual(pdf_path, device_context)

    # Save to cache for future runs
    if cache is not None and enriched_chunks and pdf_hash is not None:
        cache.save_chunks(pdf_hash, enriched_chunks, pdf_path)
        print(f"  [CACHED] Saved {len(enriched_chunks)} chunks (hash: {pdf_hash})")

    return enriched_chunks


# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------
def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags shared by extraction and build scripts.

    Args:
        parser: The :class:`argparse.ArgumentParser` to extend.
    """
    parser.add_argument(
        "--input-dir", default="./manuals",
        help="Directory containing PDF manuals",
    )
    parser.add_argument(
        "--domain", default="technical documentation",
        help="Subject-matter domain",
    )
    parser.add_argument(
        "--cache-dir", default=".cache",
        help="Root directory for the pipeline cache",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable caching (fresh run every time)",
    )


def add_extraction_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags specific to the extraction step.

    Args:
        parser: The :class:`argparse.ArgumentParser` to extend.
    """
    parser.add_argument(
        "--docling-page-batch-size", type=int, default=None,
        help=(
            "Pages per Docling conversion batch. Lower values reduce memory "
            "spikes on large PDFs (default: env DOCTUNE_DOCLING_PAGE_BATCH_SIZE "
            "or 25)."
        ),
    )


def init_extractor_and_cache(
    args: argparse.Namespace,
    init_extractor: bool = True,
) -> tuple[DoclingManualExtractor | None, PipelineCache | None]:
    """Build an optional Docling extractor and optional pipeline cache from CLI args.

    Args:
        args: Parsed CLI arguments (must include the flags registered by
            :func:`add_common_cli_args`).
        init_extractor: If True, initialize the Docling extractor.

    Returns:
        Tuple of ``(extractor, cache)``.  *cache* is ``None`` when
        ``--no-cache`` was passed. *extractor* is ``None`` when
        *init_extractor* is False.
    """
    extractor = None
    if init_extractor:
        extractor = DoclingManualExtractor(
            page_batch_size=getattr(args, "docling_page_batch_size", None),
        )

    cache: PipelineCache | None = None
    if not args.no_cache:
        cache = PipelineCache(cache_dir=args.cache_dir, domain=args.domain)
        logger.info("Cache enabled: %s", cache.cache_path)
        print(f"  Cache enabled: {cache.cache_path}")
    else:
        logger.info("Cache disabled (--no-cache)")
        print("  Cache disabled (--no-cache)")

    return extractor, cache
