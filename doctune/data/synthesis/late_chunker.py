"""
late_chunker.py — Reusable late-chunking encoder for jina-embeddings-v3.

Encodes whole documents through the transformer's self-attention mechanism,
then pools per-chunk embeddings from the token-level hidden states.  This
preserves full-document context within every chunk vector — a Chapter 5
"Safety Valve" chunk carries knowledge of the "XR-7" defined in Chapter 1.

For documents exceeding the 8 192-token context window, an overlapping
sliding-window strategy assigns each chunk to the window where it sits
most centrally (farthest from any boundary) to maximise contextual coverage.

This module has **no doctune imports** — only stdlib, numpy, and torch.
Both ``diversity_selector.py`` and the future dedup replacement (Phase 4
Task 3) import from here, keeping a strictly layered dependency graph::

    late_chunker.py  ← diversity_selector.py
                     ← deduplicate_dataset.py  (future)

Usage::

    from doctune.data.synthesis.late_chunker import LateChunker

    chunker = LateChunker()
    embeddings = chunker.encode(["chunk one …", "chunk two …"])
    # embeddings.shape == (2, 1024), each row L2-normalised
"""

from __future__ import annotations

import logging
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_ID       = "jinaai/jina-embeddings-v3"
_MAX_TOKENS     = 8192
_WINDOW_TOKENS  = 7500   # tokens per sliding window
_OVERLAP_TOKENS = 500    # overlap between adjacent windows
_EMBED_DIM      = 1024   # jina-embeddings-v3 output dimension
_SEPARATOR      = "\n"   # used when reconstructing a document from chunks


# ---------------------------------------------------------------------------
# Pure helper functions (no model dependency)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """NFKC-normalise *text* to ensure tokenizer alignment.

    Folds compatibility characters (curly quotes, em-dashes, ligatures)
    into their canonical equivalents, which is consistent with what most
    BPE tokenizers expect.
    """
    return unicodedata.normalize("NFKC", text)


def reconstruct_full_text(
    norm_chunks: list[str],
) -> tuple[str, list[tuple[int, int]]]:
    """Join normalised chunks and return the concatenated text + char spans.

    Args:
        norm_chunks: Already-normalised chunk strings.

    Returns:
        A ``(full_text, char_spans)`` tuple where *char_spans* is a list of
        ``(start_char, end_char)`` pairs — one per chunk — giving each
        chunk's character-level position inside *full_text*.
    """
    full_text  = _SEPARATOR.join(norm_chunks)
    char_spans: list[tuple[int, int]] = []
    cursor     = 0
    for chunk in norm_chunks:
        char_spans.append((cursor, cursor + len(chunk)))
        cursor += len(chunk) + len(_SEPARATOR)
    return full_text, char_spans


def char_span_to_token_span(
    char_start:     int,
    char_end:       int,
    offset_mapping: list[tuple[int, int]],
) -> tuple[int, int]:
    """Map a character span to inclusive-start / exclusive-end token indices.

    Uses the tokenizer's ``offset_mapping`` (list of per-token
    ``(char_start, char_end)`` tuples) for exact alignment.

    Args:
        char_start: Start character index in the document string.
        char_end:   Exclusive end character index.
        offset_mapping: Per-token character ranges from the tokenizer.

    Returns:
        ``(token_start, token_end)`` — inclusive start, exclusive end.

    Raises:
        ValueError: If no tokens fall within the character span (typically
            caused by document truncation or a normalisation mismatch).
    """
    token_start: int | None = None
    token_end:   int | None = None

    for idx, (t_start, t_end) in enumerate(offset_mapping):
        # Skip special tokens (CLS, SEP, PAD) whose offset is (0, 0),
        # but preserve the genuine first token at position 0.
        if t_start == 0 and t_end == 0 and idx != 0:
            continue

        if t_start >= char_start and token_start is None:
            token_start = idx

        # Include any token whose start is strictly before span end.
        # This correctly handles subword tokens straddling the boundary.
        if token_start is not None and t_start < char_end:
            token_end = idx + 1

    if token_start is None or token_end is None:
        raise ValueError(
            f"No tokens for char span ({char_start}, {char_end}). "
            "Check for truncation or normalization mismatch."
        )
    return token_start, token_end


def pool_chunk(
    token_embeddings: torch.Tensor,
    token_start:      int,
    token_end:        int,
) -> torch.Tensor:
    """Mean-pool and L2-normalise token embeddings for one chunk.

    Args:
        token_embeddings: Shape ``[seq_len, embed_dim]``.
        token_start: Inclusive start token index.
        token_end:   Exclusive end token index.

    Returns:
        Unit-norm embedding of shape ``[embed_dim]``.
    """
    span   = token_embeddings[token_start:token_end, :]
    pooled = span.mean(dim=0)
    return F.normalize(pooled, p=2, dim=0)


def pool_all_spans(
    token_embeddings: torch.Tensor,
    offset_mapping:   list[tuple[int, int]],
    char_spans:       list[tuple[int, int]],
) -> np.ndarray:
    """Pool embeddings for every chunk in a single-window document.

    Args:
        token_embeddings: Shape ``[seq_len, embed_dim]``.
        offset_mapping:   Per-token character ranges.
        char_spans:       Per-chunk ``(char_start, char_end)`` pairs.

    Returns:
        Numpy array of shape ``[n_chunks, embed_dim]`` (float32).
    """
    n   = len(char_spans)
    arr = np.zeros((n, _EMBED_DIM), dtype=np.float32)
    for i, (cs, ce) in enumerate(char_spans):
        try:
            ts, te  = char_span_to_token_span(cs, ce, offset_mapping)
            arr[i]  = pool_chunk(token_embeddings, ts, te).cpu().numpy()
        except ValueError as exc:
            logger.warning("Chunk %d skipped: %s", i, exc)
    return arr


def assign_chunks_to_windows(
    char_spans:        list[tuple[int, int]],
    window_char_ranges: list[tuple[int, int]],
) -> list[int]:
    """Assign each chunk to the window where it sits most centrally.

    "Most central" means the chunk's midpoint is closest to the window's
    midpoint, which maximises contextual coverage from the transformer's
    bidirectional attention.

    This is a **pure function** with no model dependency — pass synthetic
    char spans and window ranges for unit testing.

    Args:
        char_spans: ``(char_start, char_end)`` for each chunk.
        window_char_ranges: ``(char_start, char_end)`` for each window.

    Returns:
        List of window indices, one per chunk.
    """
    assignments: list[int] = []
    for cs, ce in char_spans:
        chunk_mid = (cs + ce) / 2
        best = min(
            range(len(window_char_ranges)),
            key=lambda wi: abs(
                chunk_mid - (window_char_ranges[wi][0] + window_char_ranges[wi][1]) / 2
            ),
        )
        assignments.append(best)
    return assignments


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LateChunker:
    """Lazy-loaded jina-embeddings-v3 encoder with late-chunking support.

    Handles both single-pass and sliding-window encoding transparently.
    The model is not downloaded or moved to GPU until the first call to
    :meth:`encode`, avoiding ~2.2 GB memory cost when encoding is unused.

    Args:
        model_id: HuggingFace model identifier.  Override only for testing
            an alternative long-context embedding model.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        device:   str | None = None,
    ) -> None:
        self.model_id = model_id
        self._target_device = device
        self._device:    torch.device | None = None
        self._tokenizer  = None
        self._model      = None

        #: Token count from the most recent :meth:`encode` call.
        #: Lets consumers detect sliding-window usage without re-tokenizing.
        self.last_token_count: int = 0

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Download and load model + tokenizer on first use."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        logger.info("Loading %s for late chunking...", self.model_id)
        print(f"  [LateChunker] Loading {self.model_id}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self._model.eval()

        if self._target_device:
            self._device = torch.device(self._target_device)
        else:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self._model = self._model.to(self._device)
        logger.info("LateChunker loaded on %s.", self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        chunks: list[str],
        *,
        task: str = "retrieval.passage",
    ) -> np.ndarray:
        """Encode chunks via late chunking.

        Normalises every chunk, reconstructs the full document text,
        and encodes it through jina-embeddings-v3 to obtain token-level
        hidden states.  Chunk embeddings are then derived by mean-pooling
        over each chunk's token span and L2-normalising.

        For documents exceeding the 8 192-token context window, a
        sliding-window strategy is used automatically.

        Args:
            chunks: Chunk strings (may include heading prefixes).
            task: Jina v3 LoRA adapter task.  ``"retrieval.passage"``
                for indexing, ``"text-matching"`` for dedup similarity.

        Returns:
            Numpy array of shape ``[n_chunks, 1024]`` (float32),
            each row L2-normalised.
        """
        self._ensure_loaded()

        if not chunks:
            return np.empty((0, _EMBED_DIM), dtype=np.float32)

        norm_chunks = [normalize(c) for c in chunks]
        full_text, char_spans = reconstruct_full_text(norm_chunks)

        total_tokens = self.count_tokens(full_text)
        self.last_token_count = total_tokens

        if total_tokens > _MAX_TOKENS:
            return self._encode_sliding_window(full_text, char_spans, task=task)
        return self._encode_single(full_text, char_spans, task=task)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text* without truncation."""
        self._ensure_loaded()
        return len(
            self._tokenizer(
                text, truncation=False, add_special_tokens=True,
            )["input_ids"]
        )

    # ------------------------------------------------------------------
    # Encoding — single pass
    # ------------------------------------------------------------------

    def _encode_single(
        self,
        full_text:  str,
        char_spans: list[tuple[int, int]],
        *,
        task: str,
    ) -> np.ndarray:
        """Encode the full document in one pass and pool per chunk."""
        token_embeddings, offset_mapping = self._get_token_embeddings(
            full_text, task=task,
        )
        return pool_all_spans(token_embeddings, offset_mapping, char_spans)

    # ------------------------------------------------------------------
    # Encoding — sliding window
    # ------------------------------------------------------------------

    def _encode_sliding_window(
        self,
        full_text:  str,
        char_spans: list[tuple[int, int]],
        *,
        task: str,
    ) -> np.ndarray:
        """Encode a long document via overlapping windows.

        Each chunk is assigned to the window where it sits most centrally
        (farthest from any window boundary), maximising contextual coverage.
        """
        # Tokenize without truncation to get the complete offset map
        full_enc = self._tokenizer(
            full_text,
            return_offsets_mapping=True,
            truncation=False,
            add_special_tokens=False,
        )
        all_offsets: list[tuple[int, int]] = full_enc["offset_mapping"]
        total_tokens = len(all_offsets)

        # Build (token_start, token_end) windows
        windows: list[tuple[int, int]] = []
        start = 0
        while start < total_tokens:
            end = min(start + _WINDOW_TOKENS, total_tokens)
            windows.append((start, end))
            if end == total_tokens:
                break
            start += _WINDOW_TOKENS - _OVERLAP_TOKENS

        # Window character ranges
        window_char_ranges: list[tuple[int, int]] = []
        for wt_start, wt_end in windows:
            wc_start = all_offsets[wt_start][0]
            wc_end   = all_offsets[wt_end - 1][1]
            window_char_ranges.append((wc_start, wc_end))

        # Assign each chunk to its most central window
        chunk_to_window = assign_chunks_to_windows(char_spans, window_char_ranges)

        # Encode each window; pool assigned chunks
        n = len(char_spans)
        embeddings: list[np.ndarray | None] = [None] * n

        for wi, (_wt_start, _wt_end) in enumerate(windows):
            assigned = [ci for ci, w in enumerate(chunk_to_window) if w == wi]
            if not assigned:
                continue

            wc_start, wc_end = window_char_ranges[wi]
            window_text = full_text[wc_start:wc_end]

            token_embs, offset_map = self._get_token_embeddings(
                window_text, task=task,
            )

            for ci in assigned:
                cs, ce = char_spans[ci]
                # Adjust character offsets relative to window start
                local_cs = cs - wc_start
                local_ce = ce - wc_start
                try:
                    ts, te = char_span_to_token_span(local_cs, local_ce, offset_map)
                    emb = pool_chunk(token_embs, ts, te)
                    embeddings[ci] = emb.cpu().numpy()
                except ValueError as exc:
                    logger.warning(
                        "Chunk %d skipped in window %d: %s", ci, wi, exc,
                    )

        # Fill gaps with zero vectors (should be rare)
        arr = np.zeros((n, _EMBED_DIM), dtype=np.float32)
        for i, emb in enumerate(embeddings):
            if emb is not None:
                arr[i] = emb
        return arr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token_embeddings(
        self,
        text: str,
        *,
        task: str,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Encode *text* and return token-level hidden states + offset map.

        .. important::

           ``offset_mapping`` is **popped** from the tokenizer output dict
           before the forward pass.  Jina's custom ``modeling.py`` raises
           ``TypeError`` on unexpected keyword arguments, and
           ``offset_mapping`` is not a model input.
        """
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKENS,
            return_offsets_mapping=True,
        )
        # Pop offset_mapping BEFORE passing to the model — Jina's custom
        # forward() does not accept this key and will raise TypeError.
        offset_mapping: list[tuple[int, int]] = enc.pop(
            "offset_mapping"
        ).squeeze(0).tolist()

        input_ids      = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
            )

        token_embeddings = out.last_hidden_state.squeeze(0)  # [seq, 1024]
        return token_embeddings, offset_mapping
