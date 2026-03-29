# Implementation Report: Late Chunking with Docling and jina-embeddings-v3
### For Synthetic Dataset Generation and LLM Fine-Tuning

---

## Executive Summary

Late chunking is an embedding technique designed to preserve global document context within individual segments. Traditional pipelines split text into chunks *before* embedding, which causes **contextual drift** — a segment loses awareness of information that appears elsewhere in the document. Late chunking reverses this: the entire document is encoded first, producing a token-level hidden-state sequence, and chunk vectors are derived *afterward* by pooling over the relevant token spans.

This report describes a full pipeline that pairs **IBM's Docling** document parser with the **`jinaai/jina-embeddings-v3`** model to implement late chunking at scale. The primary goal is the construction of high-quality synthetic datasets for fine-tuning language models, where contextual fidelity of training examples directly affects downstream model quality.

**Pipeline overview:**

```
PDF / DOCX / HTML
      │
      ▼
 [Docling Parser]         ← structured extraction: headings, tables, lists
      │
      ▼
 [Full-text reconstruction + character offsets]
      │
      ▼
 [jina-embeddings-v3]     ← whole document → token-level hidden states [seq, 1024]
      │
      ▼
 [Boundary-aware mean-pooling per chunk]
      │
      ▼
 [Enriched chunk records: text + embedding + metadata]
      │
      ├──▶ [QA pair generation via LLM]
      ├──▶ [DPO / preference pair generation]
      └──▶ [Deduplication → JSONL / ShareGPT / DPO export]
```

---

## 1. Conceptual Framework: The "Late" Advantage

### 1.1 The Problem with Naïve Chunking

Standard retrieval-augmented generation (RAG) pipelines embed chunks in isolation. Consider a 40-page technical manual:

- **Chapter 1** introduces "Reactor Model XR-7" and its safety specifications.
- **Chapter 5** describes a "Safety Valve" procedure.

When the Chapter 5 paragraph is chunked and embedded in isolation, its vector has no knowledge of "XR-7". A query about "XR-7 pressure relief" may fail to retrieve it, even though the chunk is directly relevant. This is contextual drift. The problem compounds in technical, legal, and scientific documents where terminology is defined once and used throughout.

### 1.2 How Late Chunking Solves This

Late chunking processes the full document through the transformer's self-attention mechanism before any segmentation occurs. The process has three stages:

1. **Global Encoding:** The entire document (up to the model's context window) is tokenized and passed through the model. Bidirectional self-attention allows every token's representation to be influenced by every other token in the document.

2. **Token-Level Hidden States:** Rather than pooling the output into a single document-level vector (the standard CLS or mean-pool approach), the raw `last_hidden_state` tensor is retained — one vector per input token, shaped `[sequence_length, embedding_dim]`.

3. **Boundary-Aware Pooling:** Structural boundaries identified by a document parser are mapped to token indices. The final vector for each chunk is computed by mean-pooling the token embeddings within those index boundaries and L2-normalizing the result.

This technique was formalized in *"Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models"* (Günther et al., 2024) [1] and is specifically designed for long-context embedding models like `jina-embeddings-v3`.

### 1.3 Comparison: Late Chunking vs. Standard Chunking

| Property | Standard Chunking | Late Chunking |
|---|---|---|
| Context awareness | Chunk-local only | Full document via self-attention |
| Computational cost | Low (embed each chunk) | Higher (embed full document) |
| Cross-reference fidelity | Poor | High |
| Document length limit | Chunk size | Model context window |
| Supports long docs | Always | Requires sliding window |

### 1.4 Why This Matters for Synthetic Dataset Generation

When constructing synthetic training pairs (e.g., question-answer or instruction-response pairs) from document chunks, the quality of the chunk's contextual signal directly affects:

- **Retrieval fidelity**: whether the right chunk is found during data mining or similarity-based deduplication
- **Coherence of the generated example**: a contextually-rich chunk produces more specific and accurate synthetic questions, as the LLM receives heading hierarchy and full contextual grounding
- **Cross-reference preservation**: technical documents often define terms in one section and use them in another; late chunking carries those definitions into the embedding, producing more faithful training examples
- **Dataset diversity**: embedding-based clustering of late-chunked vectors captures semantic diversity more accurately than character-count heuristics, enabling principled selection of representative training examples

---

## 2. Docling Integration: Structured Document Parsing

IBM's **Docling** library (v2+) converts complex document formats (PDF, DOCX, PPTX, HTML) into a structured `DoclingDocument` object. Critically for late chunking, it provides **character-offset-aware chunks** and preserves the document's logical hierarchy (headings, tables, figures, lists), which are used to define chunk boundaries and to enrich the LLM prompts during dataset generation [2].

### 2.1 Installation

```bash
pip install docling transformers torch einops
```

For GPU-accelerated PDF parsing (recommended for large documents):
```bash
pip install "docling[pdf]"
```

For the full pipeline including the Anthropic client:
```bash
pip install anthropic scikit-learn numpy
```

### 2.2 Parsing a Document with Docling

```python
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# Initialize the converter
converter = DocumentConverter()

# Convert a document — accepts PDF, DOCX, PPTX, HTML, Markdown, etc.
result = converter.convert("path/to/your/document.pdf")
doc = result.document

# Use HybridChunker for semantically-aware boundaries.
# It respects headings, paragraphs, tables, and list boundaries.
# Setting max_tokens against the same tokenizer used for embedding
# guarantees chunks never exceed the token budget.
chunker = HybridChunker(tokenizer="jinaai/jina-embeddings-v3", max_tokens=256)

chunks = list(chunker.chunk(doc))

for chunk in chunks:
    print(f"Text: {chunk.text[:80]}...")
    print(f"Metadata: {chunk.meta}")  # includes page, heading context, bbox
```

Docling's `HybridChunker` is particularly valuable because:
- It is **tokenizer-aware**: `max_tokens` is enforced against the same tokenizer used for embedding.
- It preserves **heading context** in `chunk.meta`, which is prepended to the chunk text to enrich the embedding and the generation prompt.
- It handles **tables** as atomic units, preventing mid-table splits.
- It marks **figure captions** separately, avoiding polluting the text stream with bare base64 blobs.

### 2.3 Extracting the Full Document Text and Chunk Offsets

For late chunking, you need both the full concatenated document text and the character offsets of each chunk within it.

```python
def extract_chunks_with_offsets(chunks):
    """
    Reconstruct the full document text and compute character offsets
    for each chunk within it.

    Returns:
        full_text (str): the complete document text
        chunk_spans (list of tuples): (start_char, end_char) for each chunk
        chunk_texts (list of str): the raw text of each chunk
    """
    chunk_texts = [chunk.text for chunk in chunks]

    # Reconstruct document text by joining chunks with single newlines.
    # The separator length must be tracked for accurate offset computation.
    separator = "\n"
    full_text = separator.join(chunk_texts)

    chunk_spans = []
    cursor = 0
    for text in chunk_texts:
        start = cursor
        end = cursor + len(text)
        chunk_spans.append((start, end))
        cursor = end + len(separator)

    return full_text, chunk_spans, chunk_texts
```

> **Note:** This approach joins chunks with a separator. An alternative is to use `doc.export_to_markdown()` as the canonical full text, then use `chunk.meta` offsets to map back. The join approach is simpler and less sensitive to Docling version changes; use the markdown export approach when byte-perfect alignment with the original layout is required.

### 2.4 Heading-Enriched Text for Embedding

Prepending the heading hierarchy to each chunk's text before tokenization has been shown to improve retrieval quality (Anthropic, 2024 [4]; Docling documentation). This is especially valuable in documents where section bodies repeat the same vocabulary across different contexts (e.g., "Performance" in a networking chapter vs. a financial chapter).

```python
def enrich_chunk_text(chunk) -> str:
    """
    Prepend the heading context from Docling metadata to the chunk text.
    This creates a richer string for embedding without modifying the
    original chunk text stored in the record.
    """
    headings = []
    if hasattr(chunk.meta, "headings") and chunk.meta.headings:
        headings = chunk.meta.headings
    elif hasattr(chunk.meta, "doc_items"):
        # Fallback: walk doc_items for parent heading labels
        for item in chunk.meta.doc_items:
            if hasattr(item, "label") and str(item.label) == "section_header":
                headings.append(item.text if hasattr(item, "text") else "")

    if headings:
        prefix = " > ".join(h for h in headings if h)
        return f"{prefix}\n\n{chunk.text}"
    return chunk.text
```

---

## 3. Late Chunking Implementation

### 3.1 Model and Tokenizer Setup

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import unicodedata

MODEL_ID = "jinaai/jina-embeddings-v3"
MAX_TOKENS = 8192

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

The `jina-embeddings-v3` model uses a LoRA-based adapter architecture with task-specific heads [3]. It requires `trust_remote_code=True` because the adapter selection logic is implemented in a custom `modeling.py` file on the Hugging Face Hub. The embedding dimension is **1,024** and the context window is **8,192 tokens**, corresponding to approximately 6,000–7,000 words depending on vocabulary overlap.

### 3.2 Text Normalization

Apply Unicode normalization before tokenization to prevent subtle alignment drift between Docling's output and the tokenizer's expectations:

```python
def normalize_text(text: str) -> str:
    """
    NFKC normalization folds compatibility characters (curly quotes,
    em-dashes, ligatures) into their ASCII equivalents, which is
    consistent with what most BPE tokenizers expect.
    """
    return unicodedata.normalize("NFKC", text)
```

Call this on both the `full_text` and each `chunk.text` *before* computing offsets, so the character positions remain consistent.

### 3.3 Generating Token-Level Embeddings

```python
def get_token_embeddings(full_text: str, task: str = "retrieval.passage"):
    """
    Encodes the full document and returns the raw token-level hidden states.

    Args:
        full_text: The complete document text (normalized).
        task: Jina v3 LoRA adapter task. Use 'retrieval.passage' for document
              chunks destined for a retrieval index.

    Returns:
        token_embeddings: Tensor of shape [seq_len, 1024]
        encodings: The tokenizer output, including offset_mapping
    """
    full_text = normalize_text(full_text)

    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        return_offsets_mapping=True,  # Critical for char-to-token alignment
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        # Pass the task identifier to activate the correct LoRA adapter.
        # This is supported by Jina v3's custom modeling.py.
        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=task,
        )

    # Shape: [1, seq_len, 1024] — remove the batch dimension
    token_embeddings = model_output.last_hidden_state.squeeze(0)

    return token_embeddings, encodings
```

### 3.4 Mapping Character Offsets to Token Indices

This is the most fragile step in the pipeline. Tokenizers use subword algorithms (BPE, WordPiece) that do not align with character or word boundaries. The `return_offsets_mapping=True` flag returns a list of `(char_start, char_end)` tuples for each token, which allows exact mapping.

> **Bug note (original report):** The original condition `if t_end <= char_end` missed tokens whose character range partially extends past `char_end` — a common occurrence at span boundaries where a multi-character subword token straddles the boundary. The corrected condition uses `t_start < char_end`, meaning: *include any token that begins before the span ends*.

```python
def char_span_to_token_span(
    char_start: int,
    char_end: int,
    offset_mapping: list,
) -> tuple[int, int]:
    """
    Maps a character-level span to token indices using the tokenizer's
    offset mapping.

    Args:
        char_start: Start character index in the full document string.
        char_end: Exclusive end character index in the full document string.
        offset_mapping: List of (token_char_start, token_char_end) tuples
                        from the tokenizer.

    Returns:
        (token_start, token_end): Inclusive start, exclusive end token indices.

    Raises:
        ValueError: If no tokens are found for the given character span.
    """
    token_start, token_end = None, None

    for idx, (t_start, t_end) in enumerate(offset_mapping):
        # Skip special tokens (CLS, SEP, PAD) which have offset (0, 0).
        # Use idx > 0 guard to preserve the actual first token at position 0.
        if t_start == 0 and t_end == 0 and idx != 0:
            continue

        # First token whose start falls within or at the span start
        if t_start >= char_start and token_start is None:
            token_start = idx

        # Last token whose start is strictly before the span end.
        # This correctly handles subword tokens that straddle the boundary.
        if token_start is not None and t_start < char_end:
            token_end = idx + 1  # exclusive end

    if token_start is None or token_end is None:
        raise ValueError(
            f"No tokens found for char span ({char_start}, {char_end}). "
            "Check for document truncation or normalization mismatch."
        )

    return token_start, token_end
```

### 3.5 Pooling and Normalizing Chunk Embeddings

```python
def pool_chunk_embedding(
    token_embeddings: torch.Tensor,
    token_start: int,
    token_end: int,
) -> torch.Tensor:
    """
    Produces a single L2-normalized embedding for a chunk by mean-pooling
    its constituent token embeddings.

    Args:
        token_embeddings: Tensor of shape [seq_len, 1024]
        token_start: Inclusive start token index.
        token_end: Exclusive end token index.

    Returns:
        Normalized chunk embedding of shape [1024]
    """
    chunk_tokens = token_embeddings[token_start:token_end, :]  # [chunk_len, 1024]
    pooled = chunk_tokens.mean(dim=0)                          # [1024]
    normalized = F.normalize(pooled, p=2, dim=0)               # [1024]
    return normalized
```

### 3.6 Full Pipeline: Docling → Late Chunking

```python
def late_chunk_document(document_path: str) -> list[dict]:
    """
    Full pipeline: parse a document with Docling, apply late chunking,
    and return a list of enriched chunk records.

    Each record contains:
        - text:          Raw chunk text
        - enriched_text: Heading-prefixed text (used for embedding)
        - embedding:     Late-chunked vector (numpy array, shape [1024])
        - metadata:      Docling metadata (page, heading hierarchy, bbox)
        - token_span:    (start, end) token indices in the full document
        - char_span:     (start, end) character indices in the full document
    """
    # --- Stage 1: Docling Parsing ---
    converter = DocumentConverter()
    result = converter.convert(document_path)
    chunker = HybridChunker(tokenizer=MODEL_ID, max_tokens=256)
    chunks = list(chunker.chunk(result.document))

    if not chunks:
        return []

    # --- Stage 2: Reconstruct Full Text and Offsets ---
    # Enrich each chunk's text with heading context before reconstruction
    enriched_texts = [normalize_text(enrich_chunk_text(c)) for c in chunks]
    raw_texts = [normalize_text(c.text) for c in chunks]

    separator = "\n"
    full_text = separator.join(enriched_texts)

    char_spans = []
    cursor = 0
    for text in enriched_texts:
        char_spans.append((cursor, cursor + len(text)))
        cursor += len(text) + len(separator)

    # --- Stage 3: Token-Level Encoding ---
    token_embeddings, encodings = get_token_embeddings(full_text)
    offset_mapping = encodings["offset_mapping"].squeeze(0).tolist()

    # --- Stage 4: Map Spans and Pool ---
    records = []
    for i, (chunk, char_span, enriched, raw) in enumerate(
        zip(chunks, char_spans, enriched_texts, raw_texts)
    ):
        char_start, char_end = char_span
        try:
            token_start, token_end = char_span_to_token_span(
                char_start, char_end, offset_mapping
            )
        except ValueError as e:
            print(f"Warning: Skipping chunk {i} — {e}")
            continue

        embedding = pool_chunk_embedding(token_embeddings, token_start, token_end)

        records.append({
            "text": raw,
            "enriched_text": enriched,
            "embedding": embedding.cpu().numpy(),
            "metadata": chunk.meta.model_dump() if hasattr(chunk.meta, "model_dump") else {},
            "token_span": (token_start, token_end),
            "char_span": char_span,
        })

    return records
```

### 3.7 Sliding Window for Long Documents

`jina-embeddings-v3` supports up to 8,192 tokens per call — roughly 6,000–7,000 words. For longer documents, a **sliding window strategy** approximates full-document context:

1. Split the document into overlapping windows of ~7,500 tokens with ~500-token overlap.
2. Apply late chunking independently within each window.
3. For chunks in the overlap region, use the embedding from the window where the chunk is most central (i.e., farthest from any window boundary), to maximize contextual coverage.

```python
def late_chunk_long_document(
    full_text: str,
    char_spans: list[tuple[int, int]],
    window_tokens: int = 7500,
    overlap_tokens: int = 500,
) -> list[torch.Tensor]:
    """
    Applies late chunking to a document longer than the model's context window
    using an overlapping sliding window.

    Returns a list of normalized embeddings, one per chunk span.
    """
    # Tokenize the full text without truncation to get total length
    full_encoding = tokenizer(
        full_text, return_offsets_mapping=True, truncation=False
    )
    all_offsets = full_encoding["offset_mapping"]  # list of (char_start, char_end)
    total_tokens = len(all_offsets)

    # Build windows as token-index ranges
    windows = []
    start = 0
    while start < total_tokens:
        end = min(start + window_tokens, total_tokens)
        windows.append((start, end))
        if end == total_tokens:
            break
        start += window_tokens - overlap_tokens

    # For each window, record the character range it covers
    window_char_ranges = []
    for (wt_start, wt_end) in windows:
        wc_start = all_offsets[wt_start][0]
        wc_end = all_offsets[wt_end - 1][1]
        window_char_ranges.append((wc_start, wc_end))

    # Assign each chunk to the window where it is most central
    chunk_to_window = []
    for (cs, ce) in char_spans:
        chunk_mid = (cs + ce) / 2
        best_window = min(
            range(len(windows)),
            key=lambda wi: abs(chunk_mid - sum(window_char_ranges[wi]) / 2),
        )
        chunk_to_window.append(best_window)

    # Encode each window and pool assigned chunks
    embeddings = [None] * len(char_spans)

    for wi, (wt_start, wt_end) in enumerate(windows):
        # Collect chunk indices assigned to this window
        assigned = [ci for ci, w in enumerate(chunk_to_window) if w == wi]
        if not assigned:
            continue

        wc_start, wc_end = window_char_ranges[wi]
        window_text = full_text[wc_start:wc_end]

        token_embs, enc = get_token_embeddings(window_text)
        offset_map = enc["offset_mapping"].squeeze(0).tolist()

        for ci in assigned:
            cs, ce = char_spans[ci]
            # Adjust character offsets relative to window start
            local_cs = cs - wc_start
            local_ce = ce - wc_start
            try:
                ts, te = char_span_to_token_span(local_cs, local_ce, offset_map)
                embeddings[ci] = pool_chunk_embedding(token_embs, ts, te)
            except ValueError as e:
                print(f"Warning: chunk {ci} skipped in window {wi} — {e}")

    return embeddings
```

---

## 4. Synthetic Dataset Construction

### 4.1 Dataset Schema

For instruction fine-tuning, each training example should follow a standard schema:

```python
{
    "instruction": "Explain the purpose of the Safety Valve in the XR-7 system.",
    "input": "",          # optional document excerpt as context
    "output": "The Safety Valve in the XR-7 system...",
    "source_chunk": "...",    # the original chunk text
    "source_doc": "manual.pdf",
    "chunk_embedding": [...], # for deduplication and quality filtering
}
```

This Alpaca-style schema [7] is compatible with most fine-tuning frameworks (LLaMA-Factory, Axolotl, Unsloth). The `chunk_embedding` field is stripped before training; it exists only during curation.

### 4.2 Generating Question-Answer Pairs per Chunk

Because each chunk's embedding carries full-document context, using the Docling heading hierarchy as a prompt prefix produces significantly more specific and accurate synthetic questions.

```python
import anthropic
import json

client = anthropic.Anthropic()

def generate_qa_pairs(chunk_record: dict, num_pairs: int = 3) -> list[dict]:
    """
    Given a late-chunked record, generate synthetic QA pairs using Claude.
    Heading context from Docling grounds the LLM prompt.
    """
    heading_context = chunk_record["metadata"].get("headings", [])
    heading_str = " > ".join(heading_context) if heading_context else "Document"

    prompt = f"""You are a technical dataset curator. Given the document section below,
generate {num_pairs} diverse question-answer pairs suitable for fine-tuning a language model.

Section location: {heading_str}

Section text:
\"\"\"
{chunk_record['text']}
\"\"\"

Requirements:
- Questions should be answerable solely from the section text
- Vary question types: factual, procedural, conceptual
- Answers should be complete and self-contained
- Respond ONLY in JSON: {{"pairs": [{{"question": "...", "answer": "..."}}]}}

Respond with only the JSON object, no preamble."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.content[0].text
    # Strip markdown fences if the model wraps the JSON
    content = content.strip().removeprefix("```json").removesuffix("```").strip()
    data = json.loads(content)

    return [
        {
            "instruction": pair["question"],
            "output": pair["answer"],
            "source_chunk": chunk_record["text"],
            "source_doc": chunk_record["metadata"].get("filename", "unknown"),
            "chunk_embedding": chunk_record["embedding"].tolist(),
        }
        for pair in data["pairs"]
    ]
```

### 4.3 Generating DPO / Preference Pairs

For training with **Direct Preference Optimization (DPO)** [8], you need *chosen* (good) and *rejected* (poor) response pairs. Late-chunked embeddings make this straightforward: generate two answers per question at different temperatures, then use a judge prompt to assign preference labels.

```python
def generate_dpo_pairs(chunk_record: dict, num_pairs: int = 2) -> list[dict]:
    """
    Generate preference pairs for DPO training.
    Strategy: generate two candidate answers per question (high vs. low temperature),
    then use a judge call to assign chosen/rejected.
    """
    heading_context = chunk_record["metadata"].get("headings", [])
    heading_str = " > ".join(heading_context) if heading_context else "Document"

    question_prompt = f"""Generate a single challenging question about the following section.
The question must be answerable from the text.

Section: {heading_str}
Text: \"\"\"{chunk_record['text']}\"\"\"

Respond with only the question."""

    # Generate the question
    q_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        messages=[{"role": "user", "content": question_prompt}],
    )
    question = q_resp.content[0].text.strip()

    answer_prompt = f"""Answer the following question using only the provided text.
Text: \"\"\"{chunk_record['text']}\"\"\"
Question: {question}
Answer:"""

    # Generate a high-quality answer (chosen candidate)
    good_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": answer_prompt}],
    )
    good_answer = good_resp.content[0].text.strip()

    # Generate a degraded answer by truncating or paraphrasing poorly
    # In practice, use a weaker model or higher temperature for the rejected answer.
    # Here we demonstrate using a shorter, less complete response.
    bad_prompt = answer_prompt + " Give a brief, incomplete answer in one sentence."
    bad_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        messages=[{"role": "user", "content": bad_prompt}],
    )
    bad_answer = bad_resp.content[0].text.strip()

    return [{
        "prompt": question,
        "chosen": good_answer,
        "rejected": bad_answer,
        "source_chunk": chunk_record["text"],
        "source_doc": chunk_record["metadata"].get("filename", "unknown"),
        "chunk_embedding": chunk_record["embedding"].tolist(),
    }]
```

### 4.4 Deduplication via Embedding Similarity

Because document sections often repeat information (abstracts, summaries, appendices), deduplicate the final dataset using cosine similarity on chunk embeddings before writing the output file.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_by_embedding(records: list[dict], threshold: float = 0.92) -> list[dict]:
    """
    Remove near-duplicate records based on chunk embedding cosine similarity.
    Retains the first occurrence of any cluster of similar chunks.

    A threshold of 0.92 is a conservative starting point; lower it to 0.85
    for more aggressive deduplication in repetitive corpora.
    """
    if not records:
        return records

    embeddings = np.array([r["chunk_embedding"] for r in records])
    sim_matrix = cosine_similarity(embeddings)

    keep = []
    discarded = set()

    for i in range(len(records)):
        if i in discarded:
            continue
        keep.append(records[i])
        for j in range(i + 1, len(records)):
            if sim_matrix[i, j] >= threshold:
                discarded.add(j)

    print(f"Deduplication: {len(records)} → {len(keep)} records retained.")
    return keep
```

### 4.5 Dataset Quality Filtering

Beyond deduplication, apply heuristic filters to remove low-signal training examples before export. Poor chunks yield poor synthetic pairs regardless of how good the embedding is.

```python
def quality_filter(records: list[dict], min_chars: int = 80, max_chars: int = 2000) -> list[dict]:
    """
    Remove records that are likely to produce low-quality training examples:
    - Too short: boilerplate, lone headings, page numbers
    - Too long: may exceed the fine-tuning context window
    - High digit ratio: tables of raw numbers with little semantic content
    - High special-character ratio: corrupted OCR output
    """
    def is_valid(r):
        text = r["text"]
        if not (min_chars <= len(text) <= max_chars):
            return False
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        if digit_ratio > 0.4:
            return False
        special_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1)
        if special_ratio > 0.3:
            return False
        return True

    filtered = [r for r in records if is_valid(r)]
    print(f"Quality filter: {len(records)} → {len(filtered)} records retained.")
    return filtered
```

### 4.6 Exporting in Standard Fine-Tuning Formats

```python
import json

def export_jsonl(records: list[dict], output_path: str):
    """Export in Alpaca-style JSONL format (LLaMA-Factory, Axolotl compatible)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            entry = {
                "instruction": record["instruction"],
                "input": "",
                "output": record["output"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def export_sharegpt(records: list[dict], output_path: str):
    """Export in ShareGPT conversation format (Axolotl, LLaMA-Factory compatible)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            entry = {
                "conversations": [
                    {"from": "human", "value": record["instruction"]},
                    {"from": "gpt", "value": record["output"]},
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def export_dpo_jsonl(records: list[dict], output_path: str):
    """Export in DPO format compatible with TRL's DPOTrainer."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            entry = {
                "prompt": record["prompt"],
                "chosen": record["chosen"],
                "rejected": record["rejected"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

### 4.7 Batch Processing Pipeline

For processing a corpus of documents, wrap the single-document pipeline in a batch loop with progress tracking and error isolation:

```python
from pathlib import Path
from tqdm import tqdm

def build_dataset_from_corpus(
    document_paths: list[str],
    output_path: str,
    qa_pairs_per_chunk: int = 3,
    dedup_threshold: float = 0.92,
) -> None:
    """
    End-to-end pipeline: parse → late-chunk → generate QA → deduplicate → export.
    """
    all_records = []

    for doc_path in tqdm(document_paths, desc="Processing documents"):
        try:
            chunk_records = late_chunk_document(doc_path)
            chunk_records = quality_filter(chunk_records)

            for chunk_record in chunk_records:
                # Tag source document in metadata for provenance tracking
                chunk_record["metadata"]["filename"] = Path(doc_path).name
                try:
                    pairs = generate_qa_pairs(chunk_record, num_pairs=qa_pairs_per_chunk)
                    all_records.extend(pairs)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  QA generation failed for chunk in {doc_path}: {e}")
                    continue

        except Exception as e:
            print(f"Failed to process {doc_path}: {e}")
            continue

    print(f"\nTotal records before dedup: {len(all_records)}")
    all_records = deduplicate_by_embedding(all_records, threshold=dedup_threshold)

    export_jsonl(all_records, output_path)
    print(f"Dataset written to {output_path} ({len(all_records)} examples)")
```

---

## 5. Dataset Quality Evaluation

Before using the synthetic dataset for fine-tuning, measure several quality dimensions. This section introduces lightweight metrics that do not require human annotators.

### 5.1 Lexical Diversity

High lexical diversity indicates that the dataset covers a broad vocabulary and avoids repetitive phrasing. Compute the **type-token ratio (TTR)** across all instruction fields:

```python
from collections import Counter
import re

def lexical_diversity(records: list[dict]) -> float:
    """
    Compute the type-token ratio (TTR) of the instruction field.
    Values above 0.6 indicate good diversity for corpora under 10k examples.
    For large corpora, use the moving-average TTR (MATTR) instead.
    """
    tokens = []
    for r in records:
        tokens.extend(re.findall(r"\w+", r["instruction"].lower()))
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)
```

### 5.2 Embedding-Based Semantic Coverage

Use the chunk embeddings to check whether the dataset covers the full semantic space of the source corpus. A high-coverage dataset will have cluster centroids spread across the embedding space:

```python
from sklearn.cluster import KMeans

def semantic_coverage_score(records: list[dict], n_clusters: int = 20) -> float:
    """
    Fit K-Means on chunk embeddings and compute the average intra-cluster
    distance as a proxy for semantic coverage. Lower mean distance = tighter
    clusters = better coverage of distinct topics.
    """
    embeddings = np.array([r["chunk_embedding"] for r in records])
    if len(embeddings) < n_clusters:
        return float("nan")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(embeddings)
    labels = km.labels_
    centers = km.cluster_centers_

    intra_dists = []
    for k in range(n_clusters):
        cluster_pts = embeddings[labels == k]
        if len(cluster_pts) == 0:
            continue
        dists = np.linalg.norm(cluster_pts - centers[k], axis=1)
        intra_dists.append(dists.mean())

    return float(np.mean(intra_dists))
```

### 5.3 Answer Length Distribution

Short answers may indicate incomplete generation; very long answers may include hallucinated content. Check the distribution:

```python
def answer_length_stats(records: list[dict]) -> dict:
    lengths = [len(r["output"].split()) for r in records]
    return {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p5": float(np.percentile(lengths, 5)),
        "p95": float(np.percentile(lengths, 95)),
    }
```

For reference, high-quality instruction datasets (Alpaca, WizardLM) typically have answer lengths in the 50–300 word range. Answers below 10 words or above 500 words are candidates for manual review.

### 5.4 Faithfulness Check (LLM-as-Judge)

For critical datasets, run an automated faithfulness evaluation using an LLM judge. The judge verifies that each answer is grounded in the source chunk:

```python
def faithfulness_score(records: list[dict], sample_size: int = 50) -> float:
    """
    Sample records and ask Claude to judge whether the answer is
    supported by the source chunk. Returns the fraction judged faithful.
    """
    sample = records[:sample_size]
    faithful = 0

    for r in sample:
        prompt = f"""Given the source text and the generated answer, answer 'YES' if the
answer is fully supported by the source text, or 'NO' if it contains information
not present in the source.

Source text: \"\"\"{r['source_chunk']}\"\"\"

Answer: \"\"\"{r['output']}\"\"\"

Is the answer faithful to the source? Answer only YES or NO."""

        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = resp.content[0].text.strip().upper()
        if verdict.startswith("YES"):
            faithful += 1

    return faithful / len(sample)
```

---

## 6. Key Implementation Considerations

### 6.1 Task-Specific Adapters in jina-embeddings-v3

Jina v3 uses **LoRA-based task adapters** to shift the embedding space for different use cases [3]. Passing the wrong task yields suboptimal embeddings.

| Task Identifier | Use Case |
|---|---|
| `retrieval.query` | Short queries in a retrieval system |
| `retrieval.passage` | Document chunks indexed for retrieval |
| `text-matching` | Semantic similarity, paraphrase detection |
| `classification` | Text classification tasks |
| `separation` | Topic clustering, outlier detection |

For this pipeline, use `retrieval.passage` when encoding chunks for indexing, and `retrieval.query` if you later query the index at inference time. When computing embedding-based deduplication within the dataset builder, `text-matching` may give marginally better similarity estimates.

### 6.2 Token Alignment Edge Cases

When using Docling and the Jina tokenizer together, watch for these common misalignment sources:

- **Unicode normalization**: Docling may normalize certain characters (curly quotes, em-dashes) differently from what the tokenizer expects. Use `unicodedata.normalize("NFKC", text)` before tokenization (included in `normalize_text()` above).
- **Whitespace handling**: BPE tokenizers encode leading spaces as part of the token. If your chunk separator adds or removes whitespace relative to the source text, offsets will drift. Always reconstruct `full_text` consistently from `enriched_texts`, not from `chunk.text` directly.
- **Table cells**: Docling's Markdown pipe-table serialization inserts characters not present in the original text. Consider serializing tables as CSV or plain text, or excluding table chunks from the embedding (they are often better handled by structured extraction tools).
- **Truncation**: If the full document exceeds `MAX_TOKENS`, chunks near the end of the document will be silently dropped by the tokenizer. The `ValueError` raised by `char_span_to_token_span` when no tokens are found for a span is the intended signal for this case; handle it with the `continue` guard shown in `late_chunk_document`.

### 6.3 Memory and Throughput

| Config | Approx. VRAM | Notes |
|---|---|---|
| CPU only | N/A | Feasible but slow (~10–30 s/doc) |
| GPU, 8 GB VRAM | Borderline | Max ~4k tokens reliably |
| GPU, 16 GB VRAM | Comfortable | Full 8k context window |
| GPU, 24 GB+ VRAM | Optimal | Full 8k context + batched documents |

For large-scale dataset construction, serialize embeddings to disk (e.g., `.npy` arrays or a vector store like LanceDB or Qdrant) rather than keeping them in memory between documents. A 1,000-document corpus at 8k tokens per document produces roughly 1,000 × 8,192 × 1,024 × 4 bytes ≈ **32 GB** of token embeddings if kept in memory simultaneously; process one document at a time and serialize chunk embeddings immediately.

---

## 7. Limitations and Alternatives

### 7.1 Limitations of Late Chunking

- **Fixed context ceiling**: Documents longer than 8,192 tokens cannot be fully context-aware in a single pass. The sliding window strategy approximates global context but does not guarantee it.
- **Computational cost**: Processing a full document at once is significantly more expensive than processing small chunks. For a 1,000-document corpus, budget 2–5× the embedding time of traditional chunking.
- **Model dependency**: The technique is only beneficial with long-context embedding models that output meaningful token-level representations. Smaller models (e.g., `all-MiniLM-L6-v2`) lack the context length and representational capacity to benefit.
- **Synthetic data quality ceiling**: Late chunking improves *retrieval* of the relevant chunk for dataset generation, but the synthetic QA quality is ultimately bounded by the LLM generator's capability and the source document quality. Noisy or poorly structured documents (e.g., scanned PDFs with OCR errors) degrade output regardless of embedding quality.

### 7.2 Complementary Approaches

| Technique | When to Use |
|---|---|
| **Contextual retrieval** (Anthropic, 2024) [4] | Prepend an LLM-generated summary of the document to each chunk before embedding. Simpler but requires an LLM call per chunk. Orthogonal to late chunking — both can be applied together. |
| **Proposition indexing** (Chen et al., 2023) [5] | Decompose text into atomic factual propositions before embedding. Increases retrieval precision but loses longer reasoning chains. Best for fact-dense corpora. |
| **Parent-child chunking** | Store small chunks for retrieval but return the full parent section for generation. Complementary to late chunking, not a replacement. |
| **Self-Instruct / Evol-Instruct** (Wang et al., 2022; Xu et al., 2023) [9][10] | Use the LLM to iteratively rewrite and complicate existing instructions. Increases dataset complexity but requires careful quality filtering. |
| **RAGAS** (Es et al., 2023) [11] | Framework for automatic RAG evaluation. Can be used to evaluate whether the retrieval step (powered by late-chunked embeddings) actually improves downstream answer quality. |

---

## 8. References

1. **Günther, M., et al.** (2024). *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*. JinaAI Technical Report. [arXiv:2409.04701](https://arxiv.org/abs/2409.04701)

2. **Auer, P., et al.** (2024). *Docling Technical Report*. IBM Research. [arXiv:2408.09869](https://arxiv.org/abs/2408.09869)

3. **Sturua, S., et al.** (2024). *jina-embeddings-v3: Multilingual Embeddings With Task LoRA*. JinaAI Technical Report. [arXiv:2409.10173](https://arxiv.org/abs/2409.10173)

4. **Anthropic** (2024). *Introducing Contextual Retrieval*. Anthropic Research Blog. [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval)

5. **Chen, T., et al.** (2023). *Dense X Retrieval: What Retrieval Granularity Should We Use?* [arXiv:2312.06648](https://arxiv.org/abs/2312.06648)

6. **Hu, E., et al.** (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

7. **Taori, R., et al.** (2023). *Alpaca: A Strong, Replicable Instruction-Following Model*. Stanford Center for Research on Foundation Models. [crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)

8. **Rafailov, R., et al.** (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

9. **Wang, Y., et al.** (2022). *Self-Instruct: Aligning Language Models with Self-Generated Instructions*. [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)

10. **Xu, C., et al.** (2023). *WizardLM: Empowering Large Language Models to Follow Complex Instructions*. [arXiv:2304.12244](https://arxiv.org/abs/2304.12244)

11. **Es, S., et al.** (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)

12. **Raffel, C., et al.** (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) — Background on seq2seq fine-tuning objectives.

---

*Report version: 3.0 | Covers Docling v2, jina-embeddings-v3, transformers ≥ 4.40, anthropic-sdk ≥ 0.28*
