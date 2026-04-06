# `doctune/data/extraction`

This package is **Stage 1** of the doctune data pipeline. Its sole responsibility is
turning raw PDF manuals into clean, enriched markdown chunks that later stages
(diversity selection, deduplication, and teacher-model synthesis) can consume.

```
PDF files  ──►  DoclingManualExtractor  ──►  list[str]  ──►  PipelineCache
                (layout-aware OCR)           (markdown chunks)
```

---

## Why a dedicated extraction package?

PDF extraction is the most computationally expensive step in the pipeline. Isolating
it into its own package means:

- It can be run **independently** (`extract_dataset.py`) to warm the cache before a
  full build, without touching synthesis or deduplication.
- The `DoclingManualExtractor` class stays a **pure, testable unit** with no
  knowledge of downstream concerns.
- Configuration knobs (batch size, retry policy, GPU device) are controlled by
  environment variables, keeping the class usable from code and from the CLI alike.

---

## Files

| File | Role |
|---|---|
| `pdf_extractor.py` | Core extraction class + module-level constants |
| `extract_dataset.py` | CLI entry point for extraction-only runs |
| `__init__.py` | Package API surface |

---

## `pdf_extractor.py`

### Module-level constants

| Constant | Value | Purpose |
|---|---|---|
| `_MAX_CHUNK_TOKENS` | `350` | Token ceiling passed to `HybridChunker`. Keeps chunks within the teacher model's context budget. |
| `_MIN_CHUNK_CHARS` | `100` | Character floor applied after chunking. Filters page numbers, lone headings, and isolated captions (~20 tokens). |
| `_CHUNKER_TOKENIZER` | `"BAAI/bge-small-en-v1.5"` | Lightweight tokenizer used by `HybridChunker` **for token counting only** — not for embeddings. Centralised here so `HybridChunker` and any future token-counting logic share a single source of truth. |

---

### `DoclingManualExtractor`

The main public class. It wraps [IBM Docling](https://github.com/DS4SD/docling) and
exposes a single high-level method (`process_manual`) that accepts a PDF path and
returns a list of enriched markdown strings.

Docling uses the **DocLayNet** vision model to understand the visual layout of each
page (tables, headings, columns, reading order) before extracting text. This is
significantly more faithful than naive text extraction for technical manuals.

#### Construction — `__init__(page_batch_size=None)`

Initialises the Docling converter and the `HybridChunker`. All heavy Docling imports
are deferred to call time (inside `__init__`) to avoid long import times when the
module is loaded but not immediately used.

**Key instance attributes:**

| Attribute | Source | Default | Description |
|---|---|---|---|
| `page_batch_size` | arg / `DOCTUNE_DOCLING_PAGE_BATCH_SIZE` | `25` | Pages processed per Docling call. Smaller values reduce native-memory spikes on large PDFs. |
| `retry_attempts` | `DOCTUNE_DOCLING_RETRY_ATTEMPTS` | `3` | How many times a failing page range is retried before splitting or skipping. |
| `retry_backoff_seconds` | `DOCTUNE_DOCLING_RETRY_BACKOFF_SECONDS` | `1.0` | Linear backoff multiplier between retries (attempt × backoff). |
| `converter` | built by `_build_converter()` | — | The live Docling `DocumentConverter` instance. |
| `chunker` | `HybridChunker` | — | Splits converted documents into token-bounded chunks. |

---

#### Public method — `process_manual(pdf_path, device_context) → list[str]`

The only method external code should call. It orchestrates the full
extract → chunk → enrich pipeline for a single PDF.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `pdf_path` | `str` | Absolute or relative path to the PDF file. |
| `device_context` | `str` | Human-readable label injected into every chunk header (e.g. `"Product User Guide"`). Downstream models use this to attribute answers to a source. |

**Returns:** `list[str]` — Each element is a markdown string of the form:

```
### [Source Context: Product User Guide] [Section: Chapter 3 > Wi-Fi Setup]

<raw chunk text>
```

**Behaviour:**
1. Validates that the file exists and has a `.pdf` extension.
2. Reads the page count via `pypdfium2` (fast, no OCR).
3. If the page count is known, processes the PDF in `page_batch_size`-page windows
   to bound native memory usage.
4. If the page count cannot be determined, falls back to a single full-document
   conversion.
5. Each converted segment is passed to `HybridChunker`, which splits text at heading
   boundaries and merges sibling chunks up to `_MAX_CHUNK_TOKENS`.
6. Chunks shorter than `_MIN_CHUNK_CHARS` are discarded.
7. Each surviving chunk is enriched with a `### [Source Context: …]` header and an
   optional `[Section: …]` breadcrumb derived from the heading hierarchy.

Returns `[]` on any unrecoverable error (logged at `ERROR` level).

---

#### Private helpers

| Method | Signature | Description |
|---|---|---|
| `_get_env_numeric` | `(var_name, default, cast, min_value) → int\|float` | Reads a numeric config value from an environment variable with clamping and safe fallback on invalid input. Used in `__init__` to populate `page_batch_size`, `retry_attempts`, and `retry_backoff_seconds`. |
| `_resolve_docling_device` | `() → str` | Determines the OCR compute device (`"cpu"` or `"cuda:N"`) by inspecting the `DOCTUNE_DOCLING_USE_GPU` env var and cross-checking CUDA availability via `torch`. Defaults to `auto` (uses GPU if available). Falls back to CPU gracefully. |
| `_build_converter` | `() → DocumentConverter` | Instantiates the Docling `DocumentConverter` with RapidOCR on the torch backend and the resolved compute device. Called once at init and again after each converter reset. |
| `_suppress_rapidocr_logs` | `() → None` | Silences all RapidOCR loggers (both well-known names and any module-specific loggers created at import time) to prevent INFO-level noise from flooding pipeline output. |
| `_build_section_breadcrumb` | `(chunk) → str` | Extracts the heading hierarchy from a Docling `DocChunk`'s `.meta.headings` list and joins it with ` > ` separators (e.g. `"Chapter 3 > Wi-Fi Setup"`). Returns `""` if metadata is absent or malformed. |
| `_reset_converter` | `(reason) → str` | Tears down and rebuilds the Docling converter. Called when a page-range retry hard-fails, to recover from native pipeline corruption without restarting the whole process. |
| `_get_page_count` | `(pdf_path) → int\|None` | Uses `pypdfium2` to read the page count cheaply (no OCR). Returns `None` if `pypdfium2` is unavailable or the file is unreadable. |
| `_convert_range_with_fallback` | `(pdf_path, start_page, end_page) → list` | Converts a page range with retry + exponential-like backoff. If all retries are exhausted, **recursively bisects** the range and retries each half independently. Single failing pages are skipped and logged. Accepts both `SUCCESS` and `PARTIAL_SUCCESS` statuses as usable results. |

---

## `extract_dataset.py`

A standalone CLI script for running the extraction step in isolation, without
triggering synthesis or deduplication. Useful for warming the chunk cache before a
full pipeline run, or for debugging extraction quality on a specific PDF set.

### `run_extraction(args)`

Orchestrates the extraction-only pipeline:

1. Validates that `args.input_dir` exists (exits with a `CRITICAL` message if not).
2. Initialises `DoclingManualExtractor` and `PipelineCache` via `init_extractor_and_cache`.
3. Discovers all `*.pdf` files in `args.input_dir` via `discover_pdfs`.
4. For each PDF, resolves a `device_context` label from the filename and calls
   `extract_chunks_cached`, which uses the cache when available and falls back to
   live Docling extraction.
5. Prints a summary of total manuals processed and chunks extracted.

### CLI flags

Inherits all flags from `add_common_cli_args` and `add_extraction_cli_args`
(defined in `pipeline/pipeline_utils.py`):

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | `./manuals` | Directory containing PDF manuals |
| `--domain` | `technical documentation` | Domain label, used as the cache subdirectory name |
| `--cache-dir` | `.cache` | Root directory for the pipeline cache |
| `--no-cache` | `False` | Disable caching; forces a fresh extraction run |
| `--chunk-sim-threshold` | `0.82` | Cosine similarity floor for chunk-level deduplication (passed through; not used in extraction) |
| `--pair-sim-threshold` | `0.92` | Cosine similarity floor for prompt-level deduplication (passed through; not used in extraction) |
| `--docling-page-batch-size` | env / `25` | Pages per Docling batch; overrides `DOCTUNE_DOCLING_PAGE_BATCH_SIZE` |

**Example:**

```bash
python -m doctune.data.extraction.extract_dataset \
    --input-dir ./manuals \
    --domain my_product \
    --docling-page-batch-size 10
```

---

## `__init__.py`

Exposes `DoclingManualExtractor` as the package's public API:

```python
from doctune.data.extraction import DoclingManualExtractor
```

---

## Environment variables

All env vars are read with safe numeric clamping via `_get_env_numeric`. Invalid
values fall back to defaults with a warning log.

| Variable | Default | Description |
|---|---|---|
| `DOCTUNE_DOCLING_PAGE_BATCH_SIZE` | `25` | Pages per conversion batch |
| `DOCTUNE_DOCLING_RETRY_ATTEMPTS` | `3` | Max retry attempts per page range (min 1) |
| `DOCTUNE_DOCLING_RETRY_BACKOFF_SECONDS` | `1.0` | Linear backoff base in seconds |
| `DOCTUNE_DOCLING_USE_GPU` | `auto` | Device selection: `auto`, `cpu`, `cuda`, `cuda:N` |

---

## Data flow diagram

```
┌──────────────┐    pypdfium2     ┌─────────────────┐
│   PDF file   │ ──page count──► │                 │
│  (*.pdf)     │                 │  DoclingManual  │
│              │ ──OCR batches──►│  Extractor      │
└──────────────┘   (Docling +    │                 │
                   RapidOCR)     └────────┬────────┘
                                          │
                                  HybridChunker
                                  (token-bounded,
                                   heading-aware)
                                          │
                                 ┌────────▼────────┐
                                 │  list[str]      │
                                 │  enriched chunk │
                                 │  markdown       │
                                 └────────┬────────┘
                                          │
                                   PipelineCache
                                   (.cache/<domain>/
                                    chunks_<hash>.json)
                                          │
                                 ┌────────▼────────┐
                                 │  build_dataset  │
                                 │  (Stage 2+)     │
                                 └─────────────────┘
```
