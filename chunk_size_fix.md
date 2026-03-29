# Chunk size fix — `pdf_extractor.py` + `teacher_model_synthesis.py`

**Phase 1 · Task 1 of 3**
Target: enforce a 350-token ceiling on chunks fed to the teacher model.

---

## `pdf_extractor.py`

### 1. Import — line 66

Replace `HierarchicalChunker` with `HybridChunker`.

```diff
-from docling.chunking import HierarchicalChunker
+from docling.chunking import HybridChunker
```

### 2. Module-level constant — after imports, before the class (line 18)

Add immediately after the `logger` definition on line 16.

```diff
 logger = logging.getLogger(__name__)
+
+# Maximum tokens per chunk sent to the teacher model.
+# HybridChunker enforces this ceiling; see __init__ for rationale.
+_MAX_CHUNK_TOKENS: int = 350
```

### 3. Chunker instantiation — line 83 (inside `__init__`)

```diff
-        self.chunker = HierarchicalChunker()
+        self.chunker = HybridChunker(
+            tokenizer="BAAI/bge-small-en-v1.5",  # lightweight; used only for token counting
+            max_tokens=_MAX_CHUNK_TOKENS,
+            merge_peers=True,  # merges tiny sibling chunks up to the limit
+        )
```

> **`tokenizer` note:** `BAAI/bge-small-en-v1.5` is used here purely for token counting
> during chunking. It gives a close approximation of GPT-4 / Claude token counts for
> English technical text. When jina-embeddings-v3 is wired in (Phase 4), swap this
> string to `"jinaai/jina-embeddings-v3"` — `_MAX_CHUNK_TOKENS` carries through unchanged.

> **`merge_peers` note:** Without this flag, a heading with a one-sentence body produces
> its own chunk that passes the 100-character floor filter but carries almost no useful
> signal for synthesis. `merge_peers=True` joins adjacent sibling chunks up to the token
> ceiling before emitting them.

### 4. Minimum character filter comment — line 403 (inside `process_manual`)

Behavior is unchanged; the comment makes the two-sided contract explicit.

```diff
-                if len(raw_text.strip()) < 100:
-                    continue
+                # Floor: ~20 tokens. Filters page numbers, isolated captions, lone headings.
+                # Ceiling is enforced upstream by HybridChunker at _MAX_CHUNK_TOKENS.
+                if len(raw_text.strip()) < 100:
+                    continue
```

---

## `teacher_model_synthesis.py`

### 5. User prompt — lines 332–335 (inside `generate_sft_pairs`)

Add a focus instruction so the teacher model targets distinct specific claims
rather than generating paraphrased summary questions over the whole chunk.

```diff
-        user_prompt = (
-            f'Text Chunk:\n"""{markdown_chunk}"""\n\n'
-            'Generate 2 to 3 Question-Answer pairs.'
-        )
+        user_prompt = (
+            f'Text Chunk:\n"""{markdown_chunk}"""\n\n'
+            "Identify the most specific actionable claim, step, or fact in this chunk. "
+            "Generate 2 to 3 Question-Answer pairs, each targeting a *distinct* piece of "
+            "information. Do not ask the same question twice in different words."
+        )
```

---

## Validation — temporary diagnostic block

Add inside `process_manual` in `pdf_extractor.py`, immediately after the chunk
loop closes (after line 410), to measure the token distribution before and after:

```python
# --- TEMPORARY DIAGNOSTIC — remove after validation ---
try:
    from transformers import AutoTokenizer as _Tok
    _tok = _Tok.from_pretrained("BAAI/bge-small-en-v1.5")
    token_counts = [len(_tok.encode(c)) for c in final_dataset_chunks]
    if token_counts:
        print(f"\n--- Chunk token distribution ---")
        print(f"  Count : {len(token_counts)}")
        print(f"  Min   : {min(token_counts)}")
        print(f"  Max   : {max(token_counts)}")
        print(f"  Mean  : {sum(token_counts) / len(token_counts):.0f}")
        print(f"  >350  : {sum(1 for t in token_counts if t > _MAX_CHUNK_TOKENS)} chunks above ceiling")
except Exception:
    pass
# --- END DIAGNOSTIC ---
```

**Expected output after the fix** on a typical technical manual:

| Metric | Before (no ceiling) | After (`_MAX_CHUNK_TOKENS = 350`) |
|---|---|---|
| Max tokens | 1 000 – 3 000+ | ≤ 350 |
| Mean tokens | 400 – 800 | 200 – 280 |
| Chunks above ceiling | many | 0 |
| Chunk count per doc | lower | higher (more, smaller chunks) |

---

## Dependency check

`HybridChunker` with a HuggingFace tokenizer requires `transformers` and a small
model download (~90 MB for `BAAI/bge-small-en-v1.5`). Verify it is in your environment:

```bash
uv pip install transformers
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5'); print('OK')"
```

The model is cached to `~/.cache/huggingface/` after the first download and is not
re-downloaded on subsequent runs.
