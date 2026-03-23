"""
pdf_extractor.py — Layout-aware PDF extraction via IBM Docling.

Parses PDF documents using Docling's vision models (DocLayNet) to
faithfully preserve tables, headings, and reading order. Produces
enriched markdown chunks with source context metadata.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class DoclingManualExtractor: # pylint: disable=too-few-public-methods
    """Extracts and chunks PDF documents using IBM Docling.

    Uses DocLayNet-based ML models to understand the visual structure
    of the PDF before extracting text. Chunks are produced based on the
    document's internal heading hierarchy.
    """

    def __init__(self, page_batch_size: int | None = None) -> None:
        """Initialize the Docling converter and hierarchical chunker."""
        # pylint: disable=import-outside-toplevel
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.chunking import HierarchicalChunker
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

        print("Initializing Docling Document Converter...")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        self._suppress_rapidocr_logs()
        self._document_converter_cls = DocumentConverter
        self._pdf_format_option_cls = PdfFormatOption
        self._input_format_enum = InputFormat
        self._pdf_pipeline_options_cls = PdfPipelineOptions
        self._rapid_ocr_options_cls = RapidOcrOptions
        self._accelerator_options_cls = AcceleratorOptions
        self.converter = self._build_converter()
        self._suppress_rapidocr_logs()
        self.chunker = HierarchicalChunker()
        # Process large PDFs in bounded page windows to reduce native-memory spikes.
        if page_batch_size is not None:
            self.page_batch_size = max(1, page_batch_size)
        else:
            env_val = os.getenv("DOCTUNE_DOCLING_PAGE_BATCH_SIZE", "25")
            try:
                self.page_batch_size = max(1, int(env_val))
            except ValueError:
                logger.warning(
                    "Invalid DOCTUNE_DOCLING_PAGE_BATCH_SIZE=%r. Using default 25.",
                    env_val,
                )
                self.page_batch_size = 25

        retry_env = os.getenv("DOCTUNE_DOCLING_RETRY_ATTEMPTS", "3")
        backoff_env = os.getenv("DOCTUNE_DOCLING_RETRY_BACKOFF_SECONDS", "1.0")
        try:
            self.retry_attempts = max(1, int(retry_env))
        except ValueError:
            logger.warning(
                "Invalid DOCTUNE_DOCLING_RETRY_ATTEMPTS=%r. Using default 3.",
                retry_env,
            )
            self.retry_attempts = 3

        try:
            self.retry_backoff_seconds = max(0.0, float(backoff_env))
        except ValueError:
            logger.warning(
                "Invalid DOCTUNE_DOCLING_RETRY_BACKOFF_SECONDS=%r. Using default 1.0.",
                backoff_env,
            )
            self.retry_backoff_seconds = 1.0

    def _resolve_docling_device(self) -> str:
        """Resolve OCR device preference with safe GPU-to-CPU fallback."""
        pref = os.getenv("DOCTUNE_DOCLING_USE_GPU", "auto").strip().lower()

        if pref in {"0", "false", "no", "cpu"}:
            return "cpu"

        if pref in {"1", "true", "yes", "gpu"}:
            pref = "cuda:0"

        try:
            import torch

            has_cuda = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count()) if has_cuda else 0
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.warning("Could not inspect CUDA availability: %s", e)
            has_cuda = False
            device_count = 0

        if pref.startswith("cuda"):
            if has_cuda:
                if pref == "cuda":
                    return "cuda:0"
                if ":" in pref:
                    suffix = pref.split(":", maxsplit=1)[1]
                    if suffix.isdigit() and int(suffix) < device_count:
                        return pref
                    logger.warning(
                        "Requested %s but only %d CUDA device(s) are visible. "
                        "Falling back to cpu.",
                        pref,
                        device_count,
                    )
                    return "cpu"
                return "cuda:0"

            logger.warning(
                "CUDA was requested via DOCTUNE_DOCLING_USE_GPU=%s but is not "
                "available in this Python environment. Falling back to cpu.",
                pref,
            )
            return "cpu"

        if pref == "auto":
            return "cuda:0" if has_cuda else "cpu"

        logger.warning(
            "Unknown DOCTUNE_DOCLING_USE_GPU value %r. Using auto device selection.",
            pref,
        )
        return "cuda:0" if has_cuda else "cpu"

    def _build_converter(self) -> Any:
        """Build Docling converter with explicit RapidOCR torch backend settings."""
        device = self._resolve_docling_device()

        ocr_options = self._rapid_ocr_options_cls(
            backend="torch",
            print_verbose=False,
            rapidocr_params={"Global.log_level": "error"},
        )
        pipeline_options = self._pdf_pipeline_options_cls(
            do_ocr=True,
            ocr_options=ocr_options,
            accelerator_options=self._accelerator_options_cls(device=device, num_threads=1),
        )
        converter = self._document_converter_cls(
            format_options={
                self._input_format_enum.PDF: self._pdf_format_option_cls(
                    pipeline_options=pipeline_options,
                ),
            }
        )

        print(f"  Docling OCR device: {device} (RapidOCR backend: torch)")
        return converter

    def _suppress_rapidocr_logs(self) -> None:
        """Mute verbose RapidOCR informational logs across logger variants."""
        candidate_names = ["RapidOCR", "rapidocr"]
        for logger_name in candidate_names:
            log_obj = logging.getLogger(logger_name)
            log_obj.setLevel(logging.ERROR)
            log_obj.propagate = False
            log_obj.disabled = True

        # Some RapidOCR builds create module-specific logger names; disable those too.
        for logger_name, logger_obj in logging.root.manager.loggerDict.items():
            if "rapidocr" not in logger_name.lower():
                continue
            if not isinstance(logger_obj, logging.Logger):
                continue
            logger_obj.setLevel(logging.ERROR)
            logger_obj.propagate = False
            logger_obj.disabled = True

    def _reset_converter(self, reason: str) -> None:
        """Reinitialize Docling converter to recover from native pipeline failures."""
        logger.warning("Reinitializing Docling converter: %s", reason)
        self.converter = self._build_converter()
        self._suppress_rapidocr_logs()

    def _get_page_count(self, pdf_path: str) -> int | None:
        """Return page count using a lightweight PDF parser, if available."""
        # pylint: disable=import-outside-toplevel
        try:
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(pdf_path)
            page_count = len(doc)
            del doc
            return page_count
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.warning("Could not read page count for %s: %s", pdf_path, e)
            return None

    def _convert_range_with_fallback(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int,
    ) -> list[Any]:
        """Convert a page range, recursively splitting on failure."""
        if start_page > end_page:
            return []

        last_error_label: str | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                conv_result = self.converter.convert(
                    pdf_path,
                    page_range=(start_page, end_page),
                    raises_on_error=False,
                )

                status_name = getattr(getattr(conv_result, "status", None), "name", "")
                errors = getattr(conv_result, "errors", []) or []
                has_document = getattr(conv_result, "document", None) is not None

                if has_document and status_name == "SUCCESS":
                    return [conv_result]

                # PARTIAL_SUCCESS often means a handful of pages failed while most
                # pages are still usable. Keep the partial result to avoid expensive
                # whole-range retries that mostly repeat OCR initialization.
                if has_document and status_name == "PARTIAL_SUCCESS":
                    logger.warning(
                        "Using partial conversion for pages %d-%d (errors=%d).",
                        start_page,
                        end_page,
                        len(errors),
                    )
                    return [conv_result]

                last_error_label = f"status={status_name or 'UNKNOWN'} errors={len(errors)}"
                logger.warning(
                    "Conversion attempt %d/%d for pages %d-%d returned non-success %s.",
                    attempt,
                    self.retry_attempts,
                    start_page,
                    end_page,
                    last_error_label,
                )
            except Exception as e: # pylint: disable=broad-exception-caught
                last_error_label = type(e).__name__
                logger.warning(
                    "Conversion attempt %d/%d failed for pages %d-%d: %s",
                    attempt,
                    self.retry_attempts,
                    start_page,
                    end_page,
                    e,
                )

            if attempt < self.retry_attempts:
                # Converter reset is costly and spammy; reserve it for hard failures.
                if last_error_label and not last_error_label.startswith("status=PARTIAL_SUCCESS"):
                    self._reset_converter(
                        f"retrying pages {start_page}-{end_page} after {last_error_label}"
                    )
                sleep_for = self.retry_backoff_seconds * attempt
                if sleep_for > 0:
                    time.sleep(sleep_for)

        if start_page == end_page:
            logger.warning(
                "Skipping page %d in %s after %d attempts (%s)",
                start_page,
                pdf_path,
                self.retry_attempts,
                last_error_label or "unknown failure",
            )
            print(
                f"  [WARN] Skipping failing page {start_page} "
                f"after {self.retry_attempts} attempts ({last_error_label or 'failure'})"
            )
            return []

        mid = (start_page + end_page) // 2
        logger.warning(
            "Range %d-%d still failed after %d attempts (%s). "
            "Retrying split ranges %d-%d and %d-%d.",
            start_page,
            end_page,
            self.retry_attempts,
            last_error_label or "unknown failure",
            start_page,
            mid,
            mid + 1,
            end_page,
        )
        self._reset_converter(
            f"splitting range {start_page}-{end_page} after repeated failures"
        )
        left_results = self._convert_range_with_fallback(pdf_path, start_page, mid)
        right_results = self._convert_range_with_fallback(pdf_path, mid + 1, end_page)
        return left_results + right_results

    def process_manual(self, pdf_path: str, device_context: str) -> list[str]:
        """Parse a PDF and return enriched markdown chunks.

        Uses Docling's vision models to identify tables, headers, and reading
        order, then chunks by heading hierarchy and injects source metadata.

        Args:
            pdf_path: Absolute or relative path to the PDF file.
            device_context: Human-readable name of the document source
                (e.g. ``"Product User Guide"``).

        Returns:
            List of markdown-formatted text chunks, each prefixed with a
            source context header. Chunks shorter than 100 characters are
            filtered out. Returns an empty list on failure.

        Raises:
            FileNotFoundError: If *pdf_path* does not exist (logged, not raised).
        """
        # Validate file existence before calling Docling
        if not os.path.exists(pdf_path):
            logger.error("PDF not found: %s", pdf_path)
            print(f"ERROR: PDF not found: {pdf_path}")
            return []

        if not pdf_path.lower().endswith(".pdf"):
            logger.warning("File does not have .pdf extension: %s", pdf_path)

        print(f"\nAnalyzing Structural Layout from: {pdf_path}...")

        page_count = self._get_page_count(pdf_path)
        if page_count is None:
            print(
                "  [WARN] Could not determine page count. "
                "Falling back to full-document conversion."
            )
            try:
                conversion_results = [
                    self.converter.convert(pdf_path, raises_on_error=False)
                ]
            except PermissionError:
                logger.error("Permission denied reading PDF: %s", pdf_path)
                print(f"ERROR: Permission denied reading PDF: {pdf_path}")
                return []
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.error(
                    "PDF parsing failed for %s: %s: %s",
                    pdf_path, type(e).__name__, e,
                )
                print(f"ERROR: PDF parsing failed for {pdf_path}: {type(e).__name__}: {e}")
                return []
        else:
            conversion_results: list[Any] = []
            print(
                f"  Processing {page_count} pages in batches of "
                f"{self.page_batch_size}..."
            )
            for start_page in range(1, page_count + 1, self.page_batch_size):
                end_page = min(start_page + self.page_batch_size - 1, page_count)
                print(f"  -> Converting pages {start_page}-{end_page}...")
                try:
                    range_results = self._convert_range_with_fallback(
                        pdf_path,
                        start_page,
                        end_page,
                    )
                except PermissionError:
                    logger.error("Permission denied reading PDF: %s", pdf_path)
                    print(f"ERROR: Permission denied reading PDF: {pdf_path}")
                    return []

                conversion_results.extend(range_results)

            if not conversion_results:
                logger.error("No pages could be converted for %s", pdf_path)
                print(f"ERROR: No pages could be converted for {pdf_path}")
                return []

        # 2. Semantic Chunking + 3. Metadata Injection & Filtering
        final_dataset_chunks: list[str] = []
        for conv_result in conversion_results:
            try:
                chunks = self.chunker.chunk(conv_result.document)
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.warning("Chunking failed for a converted segment in %s: %s", pdf_path, e)
                continue

            for chunk in chunks:
                raw_text = chunk.text

                # Filter out tiny, useless chunks (isolated page numbers, logos, etc.)
                if len(raw_text.strip()) < 100:
                    continue

                enriched_chunk = (
                    f"### [Source Context: {device_context}]\n\n"
                    f"{raw_text}\n"
                )
                final_dataset_chunks.append(enriched_chunk)

        print(f"Extraction complete. Yielded {len(final_dataset_chunks)} high-fidelity chunks.")
        return final_dataset_chunks


# ==============================================================================
# CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    extractor = DoclingManualExtractor()

    TARGET_PDF = "example_manual.pdf"
    SOURCE_NAME = "Product User Guide"

    if os.path.exists(TARGET_PDF):
        enriched_chunks = extractor.process_manual(TARGET_PDF, SOURCE_NAME)

        if enriched_chunks:
            print("\n--- PREVIEW OF DOCLING CHUNK 1 ---")
            print(enriched_chunks[0])
            print("----------------------------------")
    else:
        print(f"Awaiting PDF file: {TARGET_PDF}")
