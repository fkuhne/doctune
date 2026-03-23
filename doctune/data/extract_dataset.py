"""
extract_dataset.py — Standalone PDF chunk extraction for the dataset pipeline.

Scans a directory of PDFs, extracts content via Docling, and persists the
resulting chunks to ``.cache/<domain>/`` for later consumption by
``build_dataset.py``.

Usage::

    python extract_dataset.py --input-dir ./manuals --domain my_product
"""

from __future__ import annotations

import argparse
import os

from doctune.data.pipeline_utils import (
    add_common_cli_args,
    add_extraction_cli_args,
    discover_pdfs,
    extract_chunks_cached,
    extract_device_context,
    init_extractor_and_cache,
)


def run_extraction(args: argparse.Namespace) -> None:
    """Execute the extraction-only pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    print("--- INITIALIZING EXTRACTION PIPELINE ---")
    extractor, cache = init_extractor_and_cache(args)

    pdf_files = discover_pdfs(args.input_dir)
    if not pdf_files:
        print(f"CRITICAL: No PDFs found in directory '{args.input_dir}'.")
        return

    print(f"Found {len(pdf_files)} manuals to extract.\n")

    total_chunks = 0
    for i, pdf_path in enumerate(pdf_files):
        device_context = extract_device_context(pdf_path)
        print("============================================================")
        print(f"Extracting Manual {i + 1}/{len(pdf_files)}: {device_context}")
        print("============================================================")

        chunks = extract_chunks_cached(pdf_path, device_context, extractor, cache)
        total_chunks += len(chunks)
        print(f"  -> {len(chunks)} chunks extracted.\n")

    print("============================================================")
    print("EXTRACTION COMPLETE.")
    print(f"Total Manuals Processed: {len(pdf_files)}")
    print(f"Total Chunks Extracted:  {total_chunks}")
    if cache is not None:
        print(f"Chunks cached in:        {cache.cache_path}")
    print("============================================================")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract PDF manuals into cached chunks for later dataset building"
        ),
    )
    add_common_cli_args(parser)
    add_extraction_cli_args(parser)
    cli_args = parser.parse_args()

    os.makedirs(cli_args.input_dir, exist_ok=True)

    run_extraction(cli_args)
