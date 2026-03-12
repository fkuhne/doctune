# Data Card: Domain QA Alignment Dataset

## Dataset Overview

| Field | Value |
|---|---|
| **Name** | `alignment_dataset.jsonl` |
| **Format** | JSONL (one JSON object per line) |
| **Purpose** | SFT and DPO training for domain-specific QA |
| **Generation Method** | Synthetic (GPT-4o Teacher Model) |
| **Status** | Schema defined; dataset not yet generated |

## Source Data

### Origin
- **Input:** User-provided PDF documents placed in `./manuals/`
- **Extraction:** IBM Docling (layout-aware parsing with table preservation)
- **Synthesis:** OpenAI GPT-4o via structured outputs (Pydantic schemas)

### License Guidance
> [!IMPORTANT]
> Users are responsible for ensuring they have the right to use their source PDF documents for model training. Consider the following before generating data:
> - Are the PDFs publicly available or proprietary?
> - Does the document license permit derivative works?
> - Are there any PII or confidential content concerns?

## Schema

Every row in the dataset strictly follows this structure:

```json
{
  "prompt": "A realistic user question derived from the source text",
  "chosen": "The factually correct, step-by-step answer grounded in the source",
  "rejected": "A plausible but subtly incorrect answer (for DPO alignment)"
}
```

### Field Descriptions

| Field | Type | Constraints |
|---|---|---|
| `prompt` | `string` | Non-empty. Derived from source text, multi-angle (symptom, action, clarification). |
| `chosen` | `string` | Non-empty. Must reference source context. Factually correct. |
| `rejected` | `string` | Non-empty. Must mirror chosen style. Contains exactly one injected flaw. |

### Flaw Types in Rejected Responses
- **Wrong Sequence:** Steps presented in incorrect order
- **Wrong Component:** Correct procedure applied to wrong part
- **Subtle Hallucination:** Invented menu paths, settings, or specifications

## Quality Controls

| Control | Implementation |
|---|---|
| Schema validation | All rows must have non-null `prompt`, `chosen`, `rejected` |
| Semantic deduplication | Cosine similarity > 0.85 → discard (via `all-MiniLM-L6-v2`) |
| Chunk size limit | Max 1,000 tokens per source chunk |
| Chunk minimum | Chunks < 100 characters are filtered out |

## Statistics (Placeholder)

> [!NOTE]
> These fields will be populated after the first dataset generation run.

| Metric | Value |
|---|---|
| Total rows | — |
| Total source PDFs processed | — |
| Unique prompts retained | — |
| Duplicate prompts discarded | — |
| Avg prompt length (tokens) | — |
| Avg chosen length (tokens) | — |
| Avg rejected length (tokens) | — |

## Known Limitations

- **100% synthetic:** No human-curated QA pairs; quality is bounded by GPT-4o capabilities
- **Domain bias:** Question diversity is limited by the content of the source PDFs
- **English only:** System prompts and synthesis are configured for English text

## Recommended Usage

> **💻 No GPU required.** The data generation pipeline runs entirely on CPU (Docling + sentence-transformers + OpenAI API). You can generate the dataset locally on macOS or Linux.

1. Install base dependencies: `pip install -e "."` (or run `bash local_setup.sh`)
2. Place your PDF documents in `./manuals/`
3. Set `OPENAI_API_KEY` environment variable
4. Run `python build_dataset.py`
5. Inspect `examples/sample_dataset.jsonl` for expected format
6. Transfer the generated `.jsonl` files to your GPU environment for training (Phases 3–6)

