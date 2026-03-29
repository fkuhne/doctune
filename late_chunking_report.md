# Implementation Report: Late Chunking with Docling and jina-embeddings-v3
### For Synthetic Dataset Generation and LLM Fine-Tuning

---

## Executive Summary

Late chunking is an embedding technique designed to preserve global document context within individual segments. Traditional pipelines split text into chunks *before* embedding, which causes **contextual drift** — a segment loses awareness of information that appears elsewhere in the document. Late chunking reverses this: the entire document is encoded first, producing a token-level hidden-state sequence, and chunk vectors are derived *afterward* by pooling over the relevant token spans.

This report describes a full pipeline that pairs **IBM's Docling** document parser with the **`jinaai/jina-embeddings-v3`** model to implement late chunking at scale. The primary goal is the construction of high-quality synthetic datasets for fine-tuning language models, where contextual fidelity of training examples directly affects model quality.

---

## 1. Conceptual Framework: The "Late" Advantage

### 1.1 The Problem with Naive Chunking

Standard retrieval-augmented generation (RAG) pipelines embed chunks in isolation. Consider a 40-page technical manual:

- **Chapter 1** introduces "Reactor Model XR-7" and its safety specifications.
- **Chapter 5** describes a "Safety Valve" procedure.

When the Chapter 5 paragraph is chunked and embedded in isolation, its vector has no knowledge of "XR-7". A query about "XR-7 pressure relief" may fail to retrieve it, even though the chunk is directly relevant. This is contextual drift.

### 1.2 How Late Chunking Solves This

Late chunking processes the full document through the transformer's self-attention mechanism before any segmentation occurs. The process has three stages:

1. **Global Encoding:** The entire document (up to the model's context window) is tokenized and passed through the model. The bidirectional self-attention allows every token's representation to be influenced by every other token in the document.

2. **Token-Level Hidden States:** Rather than pooling the output into a single document-level vector (the standard CLS or mean-pool approach), the raw `last_hidden_state` tensor is retained — one embedding vector per input token, shaped `[sequence_length, embedding_dim]`.

3. **Boundary-Aware Pooling:** Structural boundaries identified by a document parser are mapped to token indices. The final vector for each chunk is computed by mean-pooling the token embeddings within those index boundaries and L2-normalizing the result.

This technique was formalized in the paper *"Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models"* (Günther et al., 2024, JinaAI) and is specifically designed for long-context embedding models like `jina-embeddings-v3`.

### 1.3 Why This Matters for Synthetic Dataset Generation

When constructing synthetic training pairs (e.g., question-answer or instruction-response pairs) from document chunks, the quality of the chunk vector directly affects:

- **Retrieval fidelity**: whether the right chunk is found during data mining or similarity search
- **Coherence of the generated example**: a contextually-rich chunk produces more coherent and accurate synthetic questions
- **Cross-reference preservation**: technical documents often define terms in one section and use them in another; late chunking carries these definitions through

---

## 2. Docling Integration: Structured Document Parsing

IBM's **Docling** library (v2+) converts complex document formats (PDF, DOCX, PPTX, HTML) into a structured `DoclingDocument` object. Critically for late chunking, it provides **character-offset-aware chunks** and preserves the document's logical hierarchy (headings, tables, figures, lists), which are used to define the chunk boundaries.

### 2.1 Installation

```bash
pip install docling transformers torch einops
```

For GPU-accelerated PDF parsing (recommended for large documents):
```bash
pip install docling[pdf]
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

# Use HybridChunker for semantically-aware boundaries
# It respects headings, paragraphs, tables, and list boundaries
chunker = HybridChunker(tokenizer="jinaai/jina-embeddings-v3", max_tokens=256)

# Produce chunks with metadata and text content
chunks = list(chunker.chunk(doc))

for chunk in chunks:
    print(f"Text: {chunk.text[:80]}...")
    print(f"Metadata: {chunk.meta}")  # includes page, heading context, bbox
```

Docling's `HybridChunker` is particularly valuable because:
- It is **tokenizer-aware**: you can configure `max_tokens` against the same tokenizer used for embedding, guaranteeing chunks never exceed a token budget.
- It preserves **heading context** in `chunk.meta`, which can be prepended to the chunk text to further enrich the embedding.
- It handles **tables** as atomic units, preventing mid-table splits.

### 2.3 Extracting the Full Document Text and Chunk Offsets

For late chunking, you need both the full concatenated document text and the character (or token) offsets of each chunk within that text.

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

> **Note:** This approach joins chunks with a separator. An alternative is to use Docling's `doc.export_to_markdown()` as the canonical full text, then use `chunk.meta` offsets to map back. Choose the approach that gives cleaner alignment for your document types.

---

## 3. Late Chunking Implementation

### 3.1 Model and Tokenizer Setup

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

MODEL_ID = "jinaai/jina-embeddings-v3"
MAX_TOKENS = 8192

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

The `jina-embeddings-v3` model uses a LoRA-based adapter architecture with task-specific heads. It requires `trust_remote_code=True` because the adapter selection logic is implemented in the model's custom `modeling.py` file hosted on the Hugging Face Hub.

### 3.2 Generating Token-Level Embeddings

```python
def get_token_embeddings(full_text: str, task: str = "retrieval.passage"):
    """
    Encodes the full document and returns the raw token-level hidden states.
    
    Args:
        full_text: The complete document text.
        task: Jina v3 adapter task. Use 'retrieval.passage' for document chunks
              destined for a retrieval index.
    
    Returns:
        token_embeddings: Tensor of shape [seq_len, embedding_dim]
        encodings: The tokenizer output (needed for offset mapping)
    """
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
        # Pass the task identifier to activate the correct LoRA adapter
        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=task,
        )
    
    # Shape: [1, seq_len, 1024] — remove the batch dimension
    token_embeddings = model_output.last_hidden_state.squeeze(0)
    
    return token_embeddings, encodings
```

### 3.3 Mapping Character Offsets to Token Indices

This is the most fragile step in the pipeline. Tokenizers use subword algorithms (BPE, WordPiece) that do not align with character or word boundaries. The `return_offsets_mapping=True` flag returns a list of `(char_start, char_end)` tuples for each token, which allows exact mapping.

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
        char_end: End character index in the full document string.
        offset_mapping: List of (token_char_start, token_char_end) tuples
                        from the tokenizer.

    Returns:
        (token_start, token_end): Inclusive start, exclusive end token indices.
    
    Raises:
        ValueError: If no tokens are found for the given character span.
    """
    token_start, token_end = None, None

    for idx, (t_start, t_end) in enumerate(offset_mapping):
        # Skip special tokens (CLS, SEP, PAD) which have offset (0, 0)
        if t_start == 0 and t_end == 0:
            continue
        if t_start >= char_start and token_start is None:
            token_start = idx
        if t_end <= char_end:
            token_end = idx + 1  # exclusive end

    if token_start is None or token_end is None:
        raise ValueError(
            f"No tokens found for char span ({char_start}, {char_end}). "
            "This may indicate a truncated document."
        )

    return token_start, token_end
```

### 3.4 Pooling and Normalizing Chunk Embeddings

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
        token_embeddings: Tensor of shape [seq_len, embedding_dim]
        token_start: Inclusive start token index.
        token_end: Exclusive end token index.

    Returns:
        Normalized chunk embedding of shape [embedding_dim]
    """
    chunk_tokens = token_embeddings[token_start:token_end, :]  # [chunk_len, dim]
    pooled = chunk_tokens.mean(dim=0)                          # [dim]
    normalized = F.normalize(pooled, p=2, dim=0)               # [dim]
    return normalized
```

### 3.5 Full Pipeline: Docling → Late Chunking

```python
def late_chunk_document(document_path: str) -> list[dict]:
    """
    Full pipeline: parse a document with Docling, apply late chunking,
    and return a list of enriched chunk records.

    Each record contains:
        - text: The raw chunk text
        - embedding: Late-chunked vector (numpy array)
        - metadata: Docling metadata (page number, parent heading, etc.)
        - token_span: (start, end) token indices in the full document
    """
    # --- Stage 1: Docling Parsing ---
    converter = DocumentConverter()
    result = converter.convert(document_path)
    chunker = HybridChunker(tokenizer=MODEL_ID, max_tokens=256)
    chunks = list(chunker.chunk(result.document))

    if not chunks:
        return []

    # --- Stage 2: Reconstruct Full Text and Offsets ---
    full_text, char_spans, chunk_texts = extract_chunks_with_offsets(chunks)

    # --- Stage 3: Token-Level Encoding ---
    token_embeddings, encodings = get_token_embeddings(full_text)
    offset_mapping = encodings["offset_mapping"].squeeze(0).tolist()

    # --- Stage 4: Map Spans and Pool ---
    records = []
    for i, (chunk, char_span) in enumerate(zip(chunks, char_spans)):
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
            "text": chunk.text,
            "embedding": embedding.cpu().numpy(),
            "metadata": chunk.meta.model_dump() if hasattr(chunk.meta, "model_dump") else {},
            "token_span": (token_start, token_end),
            "char_span": char_span,
        })

    return records
```

---

## 4. Synthetic Dataset Construction

### 4.1 Dataset Schema

For instruction fine-tuning, each training example should follow a standard schema:

```python
{
    "instruction": "Explain the purpose of the Safety Valve in the XR-7 system.",
    "input": "",  # optional document excerpt as context
    "output": "The Safety Valve in the XR-7 system...",
    "source_chunk": "...",    # the original chunk text
    "source_doc": "manual.pdf",
    "chunk_embedding": [...], # for deduplication and quality filtering
}
```

### 4.2 Generating Question-Answer Pairs per Chunk

After collecting late-chunked records, you can use a capable LLM (e.g., Claude, GPT-4, or a locally hosted Llama-3) to synthesize QA pairs from each chunk. The key insight is that because the chunk embedding carries full-document context, you can include the **document heading hierarchy** from Docling's metadata as a prefix, making the generated questions significantly more specific and accurate.

```python
import anthropic

client = anthropic.Anthropic()

def generate_qa_pairs(chunk_record: dict, num_pairs: int = 3) -> list[dict]:
    """
    Given a late-chunked record, generate synthetic QA pairs using Claude.
    """
    # Use Docling's heading context to ground the prompt
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
- Respond in JSON: {{"pairs": [{{"question": "...", "answer": "..."}}]}}

Respond with only the JSON object, no preamble."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    
    import json
    content = response.content[0].text
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

### 4.3 Deduplication via Embedding Similarity

Because document sections often repeat information (abstracts, summaries, appendices), it is important to deduplicate the final dataset. Use cosine similarity on the chunk embeddings to detect near-duplicate training examples before writing the final dataset file.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_by_embedding(records: list[dict], threshold: float = 0.92) -> list[dict]:
    """
    Remove near-duplicate records based on chunk embedding cosine similarity.
    Retains the first occurrence of any cluster of similar chunks.
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

### 4.4 Exporting in Standard Fine-Tuning Formats

```python
import json

def export_jsonl(records: list[dict], output_path: str):
    """Export dataset in Alpaca-style JSONL format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            entry = {
                "instruction": record["instruction"],
                "input": "",
                "output": record["output"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def export_sharegpt(records: list[dict], output_path: str):
    """Export dataset in ShareGPT conversation format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            entry = {
                "conversations": [
                    {"from": "human", "value": record["instruction"]},
                    {"from": "gpt", "value": record["output"]},
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

---

## 5. Key Implementation Considerations

### 5.1 Task-Specific Adapters in jina-embeddings-v3

Jina v3 uses **LoRA-based task adapters** to shift the embedding space for different use cases. Passing the wrong task will yield suboptimal embeddings.

| Task Identifier | Use Case |
|---|---|
| `retrieval.query` | Short queries in a retrieval system |
| `retrieval.passage` | Document chunks indexed for retrieval |
| `text-matching` | Semantic similarity, paraphrase detection |
| `classification` | Text classification tasks |
| `separation` | Topic clustering, outlier detection |

For this pipeline, use `retrieval.passage` for encoding chunks and `retrieval.query` if you later query the index.

### 5.2 Handling Documents Longer Than 8,192 Tokens

`jina-embeddings-v3` supports up to 8,192 tokens per call — roughly 6,000–7,000 words depending on vocabulary. For longer documents, apply a **sliding window strategy**:

1. Split the document into overlapping windows of ~7,500 tokens with ~500-token overlap.
2. Apply late chunking independently to each window.
3. For chunks that fall within the overlap region, take the embedding from the window where the chunk is most central (lowest distance from the midpoint of the window), to maximize contextual coverage.

This is a known limitation of the technique: global context is approximated, not guaranteed, for very long documents.

### 5.3 Token Alignment Edge Cases

When using Docling and the Jina tokenizer together, watch for these common misalignment sources:

- **Unicode normalization**: Docling may normalize certain characters (curly quotes, em-dashes) differently than the tokenizer expects. Preprocess with `text = unicodedata.normalize("NFKC", text)` before tokenizing.
- **Whitespace handling**: The BPE tokenizer used by Jina encodes leading spaces as part of the token. If your chunk separator adds or removes whitespace relative to the source text, offsets will drift. Always reconstruct `full_text` consistently.
- **Table cells**: Docling's table serialization (e.g., Markdown pipe tables) may insert characters not present in the original text. Consider serializing tables separately or using Docling's CSV export for table content.

### 5.4 Memory and Throughput

| Config | Approx. VRAM | Notes |
|---|---|---|
| CPU only | N/A | Feasible but slow (~10–30 sec/doc) |
| GPU, 8GB VRAM | Borderline | Max ~4k tokens reliably |
| GPU, 16GB VRAM | Comfortable | Full 8k context window |
| GPU, 24GB+ VRAM | Optimal | Full 8k context + batched documents |

For large-scale dataset construction, process documents in batches and serialize embeddings to disk (e.g., `.npy` files or a vector store like LanceDB or Qdrant) rather than keeping them in memory.

---

## 6. Limitations and Alternatives

### 6.1 Limitations of Late Chunking

- **Fixed context ceiling**: Documents longer than 8,192 tokens cannot be fully context-aware in a single pass.
- **Computational cost**: Processing a full document at once is significantly more expensive than processing small chunks. For a 1,000-document corpus, budget 2–5x the embedding time of traditional chunking.
- **Model dependency**: The technique is only beneficial with long-context embedding models that output meaningful token-level representations. Smaller models (e.g., `all-MiniLM`) are unsuitable.

### 6.2 Complementary Approaches

| Technique | When to Use |
|---|---|
| **Contextual retrieval** (Anthropic, 2024) | Prepend an LLM-generated summary of the document to each chunk before embedding. Simpler but requires an LLM call per chunk. |
| **Proposition indexing** (Chen et al., 2023) | Decompose text into atomic factual propositions before embedding. Increases precision but loses longer reasoning chains. |
| **Parent-child chunking** | Store small chunks for retrieval but return the full parent section for generation. Complementary to late chunking, not a replacement. |

---

## 7. References

1. **Günther, M., et al.** (2024). *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*. JinaAI Technical Report. [arXiv:2409.04701](https://arxiv.org/abs/2409.04701)

2. **Auer, P., et al.** (2024). *Docling Technical Report*. IBM Research. [arXiv:2408.09869](https://arxiv.org/abs/2408.09869)

3. **Sturua, S., et al.** (2024). *jina-embeddings-v3: Multilingual Embeddings With Task LoRA*. JinaAI Technical Report. [arXiv:2409.10173](https://arxiv.org/abs/2409.10173)

4. **Anthropic** (2024). *Introducing Contextual Retrieval*. Anthropic Research Blog. [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval)

5. **Chen, T., et al.** (2023). *Dense X Retrieval: What Retrieval Granularity Should We Use?* [arXiv:2312.06648](https://arxiv.org/abs/2312.06648)

6. **Hu, E., et al.** (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

---

*Report version: 2.0 | Covers Docling v2, jina-embeddings-v3, transformers ≥ 4.40*
