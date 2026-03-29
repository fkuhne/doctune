# Implementation Report: Late Chunking with jina-embeddings-v3

### Executive Summary
Late chunking is a method designed to preserve global context within document segments. Traditional chunking strategies split text into pieces before embedding, which often results in "contextual drift"—where a segment loses the overarching meaning of the document. Late chunking addresses this by embedding the entire document first and only then segmenting the resulting token-level embeddings. Using the `jinaai/jina-embeddings-v3` model, this process leverages an 8,192-token context window and sophisticated self-attention to ensure every chunk remains contextually grounded.

---

### 1. Conceptual Framework: The "Late" Advantage
In standard pipelines, a paragraph is isolated from its source document before being vectorized. If a manual describes a "Safety Valve" in Chapter 5, the vector for that paragraph might not "know" it refers to a specific reactor model mentioned in Chapter 1.

Late chunking reverses this flow:
1.  **Global Encoding:** The entire text (up to 8k tokens) is processed by the model. Through self-attention, the embedding for every single token is influenced by every other token in the document.
2.  **Token-Level Hidden States:** Instead of pooling the document into a single summary vector, the model outputs a sequence of embeddings—one for each token.
3.  **Boundary-Based Pooling:** Semantic boundaries (derived from a parser or layout analyzer) are applied to the token embeddings. The final vector for a chunk is the mean-pool of the specific tokens within those boundaries.

---

### 2. Technical Implementation Guide

#### Prerequisites
Implementation requires `transformers`, `torch`, and `einops`. The Jina v3 model uses a Mixture-of-Experts (MoE) architecture that requires `trust_remote_code=True`.

```bash
pip install transformers torch einops
```

#### Local Implementation Logic
To implement late chunking locally, you must bypass the model's default "pooled" output and access the last_hidden_state.

```python
from transformers import AutoModel, AutoTokenizer
import torch

# 1. Load the model and tokenizer
model_id = "jinaai/jina-embeddings-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

# 2. Tokenize the full document
text = "Your full document text here..."
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=8192)

# 3. Generate token-level embeddings
with torch.no_generated():
    model_output = model(**inputs)
    # This tensor contains embeddings for every token
    # Shape: [batch_size, sequence_length, 1024]
    token_embeddings = model_output.last_hidden_state
```

#### Mapping Boundaries to Tokens

The critical step is identifying which token indices correspond to your text chunks. If using a parser that provides character offsets, you must map those offsets to the tokenizer's output.

```python
def pool_late_chunk(token_embeddings, token_start_idx, token_end_idx):
    """
    Pools token-level embeddings into a single chunk vector.
    """
    # Slice the embeddings for the specific chunk
    chunk_tokens = token_embeddings[:, token_start_idx:token_end_idx, :]
    
    # Mean pooling across the token dimension
    chunk_vector = torch.mean(chunk_tokens, dim=1)
    
    # Normalize the resulting vector
    return torch.nn.functional.normalize(chunk_vector, p=2, dim=1)
```

3. Key Considerations for Implementation

## Task-Specific Adapters
Jina v3 supports specialized adapters. For document parsing and semantic segmenting, the retrieval.passage task is the most appropriate setting to ensure the embedding space is optimized for factual density.

## Token Alignment
When using a parser like Docling, the output is often in Markdown or JSON. You must ensure that the "chunks" identified by the parser are correctly aligned with the indices in the tokenizer's input_ids. Using the tokenizer's char_to_token() method is the most reliable way to find these boundaries.

## Computational Costs
Because the entire 8k context is processed at once, memory usage is higher than processing small individual chunks. For massive document sets, it is recommended to use a GPU with at least 16GB of VRAM to handle the full attention matrix of the 8,192-token sequence comfortably.

4. Why Use This for Dataset Generation?
For synthetic data generation and LLM fine-tuning, late chunking ensures that the "lessons" extracted from a document are not fragmented. When a model learns from a chunk, that chunk’s vector already contains the "DNA" of the entire document, leading to higher-quality training pairs and more coherent synthetic examples.