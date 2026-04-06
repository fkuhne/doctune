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

# Maximum tokens per chunk sent to the teacher model.
# HybridChunker enforces this ceiling; see __init__ for rationale.
_MAX_CHUNK_TOKENS: int = 350

# Minimum character length for a chunk to be kept.
# Filters page numbers, isolated captions, and lone headings (~20 tokens).
_MIN_CHUNK_CHARS: int = 100

# Tokenizer used by HybridChunker for token counting only (not for embeddings).
_CHUNKER_TOKENIZER: str = "BAAI/bge-small-en-v1.5"


class DoclingManualExtractor: # pylint: disable=too-few-public-methods
    """Extracts and chunks PDF documents using IBM Docling.

    Uses DocLayNet-based ML models to understand the visual structure
    of the PDF before extracting text. Chunks are produced based on the
    document's internal heading hierarchy.
    """

    # ------------------------------------------------------------------
    # Helpers: logging & environment
    # ------------------------------------------------------------------

    @staticmethod
    def _get_env_numeric(
        var_name: str,
        default: int | float,
        cast: type = int,
        min_value: int | float = 0,
    ) -> int | float:
        """Read a numeric value from an environment variable with safe fallback."""
        raw = os.getenv(var_name, str(default))
        try:
            return max(min_value, cast(raw))
        except ValueError:
            logger.warning("Invalid %s=%r. Using default %s.", var_name, raw, default)
            return default

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, page_batch_size: int | None = None) -> None:
        """Initialize the Docling converter and hierarchical chunker."""
        # pylint: disable=import-outside-toplevel
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.chunking import HybridChunker
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
        self.chunker = HybridChunker(
            tokenizer=_CHUNKER_TOKENIZER,
            max_tokens=_MAX_CHUNK_TOKENS,
            merge_peers=True,  # merges tiny sibling chunks up to the limit
        )

        # Process large PDFs in bounded page windows to reduce native-memory spikes.
        self.page_batch_size = (
            max(1, page_batch_size)
            if page_batch_size is not None
            else int(self._get_env_numeric(
                "DOCTUNE_DOCLING_PAGE_BATCH_SIZE", 25, int, 1,
            ))
        )
        self.retry_attempts = int(self._get_env_numeric(
            "DOCTUNE_DOCLING_RETRY_ATTEMPTS", 3, int, 1,
        ))
        self.retry_backoff_seconds = float(self._get_env_numeric(
            "DOCTUNE_DOCLING_RETRY_BACKOFF_SECONDS", 1.0, float, 0.0,
        ))

    def _resolve_docling_device(self) -> str:
        """Resolve OCR device preference with safe GPU-to-CPU fallback."""
        pref = os.getenv("DOCTUNE_DOCLING_USE_GPU", "auto").strip().lower()

        if pref in {"0", "false", "no", "cpu"}:
            return "cpu"

        # Normalise common GPU aliases to a canonical CUDA string.
        _GPU_ALIASES = {"1", "true", "yes", "gpu", "cuda"}  # noqa: N806
        if pref in _GPU_ALIASES:
            pref = "cuda:0"

        # Detect CUDA availability once.
        try:
            import torch  # pylint: disable=import-outside-toplevel

            has_cuda = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if has_cuda else 0
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Could not inspect CUDA availability: %s", exc)
            has_cuda, device_count = False, 0

        auto_device = "cuda:0" if has_cuda else "cpu"

        if not pref.startswith("cuda"):
            if pref != "auto":
                logger.warning(
                    "Unknown DOCTUNE_DOCLING_USE_GPU value %r. "
                    "Using auto device selection.",
                    pref,
                )
            return auto_device

        # pref starts with "cuda" — validate it.
        if not has_cuda:
            logger.warning(
                "CUDA was requested via DOCTUNE_DOCLING_USE_GPU=%s but is not "
                "available in this Python environment. Falling back to cpu.",
                pref,
            )
            return "cpu"

        # Validate device index (e.g. 'cuda:1').
        if ":" in pref:
            idx = pref.split(":", maxsplit=1)[1]
            if idx.isdigit() and int(idx) < device_count:
                return pref
            logger.warning(
                "Requested %s but only %d CUDA device(s) are visible. "
                "Falling back to cpu.",
                pref,
                device_count,
            )
            return "cpu"

        return "cuda:0"

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

    @staticmethod
    def _suppress_rapidocr_logs() -> None:
        """Mute verbose RapidOCR informational logs across logger variants."""
        def _silence(log_obj: logging.Logger) -> None:
            log_obj.setLevel(logging.ERROR)
            log_obj.propagate = False
            log_obj.disabled = True

        # Silence the two well-known logger names first (creates them if needed).
        for name in ("RapidOCR", "rapidocr"):
            _silence(logging.getLogger(name))

        # Catch any module-specific loggers created by RapidOCR at import time.
        for name, obj in logging.root.manager.loggerDict.items():
            if "rapidocr" in name.lower() and isinstance(obj, logging.Logger):
                _silence(obj)

    @staticmethod
    def _build_section_breadcrumb(chunk: object) -> str:
        """Build a ' > '-joined breadcrumb from Docling chunk heading metadata.

        Args:
            chunk: A Docling ``DocChunk`` produced by ``HybridChunker`` or
                ``HierarchicalChunker``.  The method is defensive against
                missing or malformed metadata.

        Returns:
            A breadcrumb string such as ``"Chapter 3: Connectivity > Wi-Fi Setup"``,
            or an empty string if no heading metadata is available.
        """
        try:
            headings: list[str] = getattr(chunk.meta, "headings", None) or []
            return " > ".join(h.strip() for h in headings if h and h.strip())
        except Exception:  # noqa: BLE001
            return ""

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
                if last_error_label:
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
                return []
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.error(
                    "PDF parsing failed for %s: %s: %s",
                    pdf_path, type(e).__name__, e,
                )
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
                    return []

                conversion_results.extend(range_results)

            if not conversion_results:
                logger.error("No pages could be converted for %s", pdf_path)
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

                # Floor: ~20 tokens. Filters page numbers, isolated captions, lone headings.
                # Ceiling is enforced upstream by HybridChunker at _MAX_CHUNK_TOKENS.
                if len(raw_text.strip()) < _MIN_CHUNK_CHARS:
                    continue

                breadcrumb = self._build_section_breadcrumb(chunk)
                section_tag = (
                    f" [Section: {breadcrumb}]" if breadcrumb else ""
                )
                enriched_chunk = (
                    f"### [Source Context: {device_context}]{section_tag}\n\n"
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
