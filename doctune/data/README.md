# `doctune/data`

This package contains the complete **data curation pipeline** for doctune — the
workflow that transforms raw PDF product manuals into a high-quality
`alignment_dataset.jsonl` file ready for SFT and DPO fine-tuning.

The pipeline is split into three focused sub-packages, each with its own README:

| Sub-package | Stage | Responsibility |
|---|---|---|
| [`extraction/`](extraction/README.md) | Stage 1 | Layout-aware PDF parsing → enriched markdown chunks |
| [`pipeline/`](pipeline/README.md) | Stage 2 | Orchestration, caching, and CLI entry point for the full end-to-end run |
| [`synthesis/`](synthesis/README.md) | Stages 2b–2d | Diversity selection, deduplication, teacher-model synthesis |

---

## End-to-end pipeline overview

```
PDF manuals (./manuals/)
        │
        │  Stage 1 — extraction/
        ▼
DoclingManualExtractor          IBM Docling + DocLayNet OCR
        │                       Layout-aware: preserves tables, headings, reading order
        │                       Batched page processing + retry/split-on-failure
        │
        ▼
PipelineCache                   Chunks persisted to .cache/<domain>/chunks_<hash>.json
        │                       SHA-256 keyed; future runs load from disk instantly
        │
        │  Stage 2 — pipeline/ orchestrates ──────► synthesis/ does the work
        ▼
DiversitySelector  (optional)   Greedy farthest-first selection on late-chunked
        │                       jina-embeddings-v3 vectors; keeps the N% most
        │                       semantically varied chunks per document
        ▼
ChunkFilter                     Cosine dedup of source chunks (threshold 0.82)
        │                       Prevents near-duplicate source material reaching the API
        ▼
TeacherModelSynthesizer         Calls OpenAI / Anthropic / Ollama
        │                       Two-stage prompt: focus-selection → 3-angle QA generation
        │                       Immediately generates a DPO rejected response per pair
        │
        ▼
PipelineCache                   Results appended to synthesis_<hash>.jsonl (append-only)
        │                       Interrupted runs resume from the last committed chunk
        ▼
DatasetFilter                   Cosine dedup of generated prompts (threshold 0.92)
        │                       Drops near-duplicate questions across all documents
        ▼
alignment_dataset.jsonl         { prompt, chosen, rejected, metadata }
```

---

## Running the pipeline

### Option A — Two-stage (recommended for large corpora)

Run extraction first to warm the cache, then run synthesis separately:

```bash
# Stage 1: extract and cache all chunks
python -m doctune.data.extraction.extract_dataset \
    --input-dir ./manuals \
    --domain my_product \
    --docling-page-batch-size 10

# Stage 2: synthesize from the warmed cache (no Docling needed)
python -m doctune.data.pipeline.build_dataset \
    --input-dir ./manuals \
    --domain my_product \
    --model gpt-4o \
    --output ./data/alignment_dataset.jsonl \
    --diversity-ratio 0.65 \
    --log-level INFO
```

### Option B — Single command

Pass both stages together. Docling is initialized alongside synthesis (slower startup):

```bash
python -m doctune.data.pipeline.build_dataset \
    --input-dir ./manuals \
    --domain my_product \
    --model gpt-4o \
    --output ./data/alignment_dataset.jsonl
```

> **Note:** `build_dataset.py` does not initialise the Docling extractor by default —
> it assumes the cache is pre-warmed. If running single-command, the extractor must
> be explicitly passed (or you can patch `init_extractor=True` in the CLI).

---

## Key design principles

### Resumability
Every completed chunk is written to the cache **immediately**. An interrupted run
(rate limit, crash, timeout) resumes from the next un-cached chunk with zero rework.

### Cost efficiency
Three gates fire **before** the teacher-model API is called:
1. **Resume filter** — already-synthesized chunks are skipped entirely.
2. **`DiversitySelector`** — reduces the chunk set to the most semantically varied
   subset (default 70%), cutting API calls proportionally.
3. **`ChunkFilter`** — drops source chunks that are near-duplicates of each other,
   preventing redundant synthesis.

### Auditability
- Synthesis spend can be reviewed at any time with `report_synthesis_spend.py`.
- The `DatasetFilter` emits structured `DEDUP_DROP` audit log lines at `INFO` level —
  enable with `--log-level INFO --log-file dedup.log` to analyse threshold calibration.

---

## Cache layout

```
.cache/<domain>/
    chunks_<hash>.json          Extracted markdown chunks for one PDF
    synthesis_<hash>.jsonl      Synthesis results (append-only, one line per chunk)
    metadata_<hash>.json        Provenance: original path, chunk count, timestamp
```

The hash is the first 16 characters of the PDF's SHA-256 digest. Renaming a PDF does
not invalidate its cache; changing its content does.

---

## Configuration reference

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DOCTUNE_DOCLING_PAGE_BATCH_SIZE` | `25` | Pages per Docling conversion batch |
| `DOCTUNE_DOCLING_RETRY_ATTEMPTS` | `3` | Max retry attempts per failing page range |
| `DOCTUNE_DOCLING_RETRY_BACKOFF_SECONDS` | `1.0` | Linear backoff base in seconds |
| `DOCTUNE_DOCLING_USE_GPU` | `auto` | OCR device: `auto`, `cpu`, `cuda`, `cuda:N` |

### Key thresholds (CLI flags)

| Flag | Default | Affects |
|---|---|---|
| `--diversity-ratio` | `0.7` | Fraction of chunks kept after diversity selection |
| `--chunk-sim-threshold` | `0.82` | `ChunkFilter` cosine ceiling |
| `--pair-sim-threshold` | `0.92` | `DatasetFilter` cosine ceiling |

---

## Sub-package READMEs

Each sub-package has its own detailed README covering all classes, methods, constants,
and internal helpers:

- [`extraction/README.md`](extraction/README.md) — `DoclingManualExtractor`, batched
  conversion, GPU fallback, `extract_dataset.py` CLI
- [`pipeline/README.md`](pipeline/README.md) — `DatasetBuilder`, `PipelineCache`,
  `pipeline_utils`, gate ordering, `build_dataset.py` CLI
- [`synthesis/README.md`](synthesis/README.md) — `LateChunker`, `DiversitySelector`,
  `ChunkFilter`, `DatasetFilter`, `TeacherModelSynthesizer`, `report_synthesis_spend`
