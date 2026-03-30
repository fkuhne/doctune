"""
diversity_selector.py — Late-chunked corpus diversity selection.

Ranks document chunks by semantic distinctiveness using jina-embeddings-v3
with late chunking. Selects the most informative subset for teacher-model
synthesis, reducing API cost without sacrificing coverage.

Late chunking encodes the full document through the transformer before
pooling per-chunk embeddings, so each vector carries full-document context.
This means "Chapter 5: Safety Valve" knows about "XR-7" from Chapter 1,
producing embeddings that reflect cross-referential meaning rather than
surface vocabulary alone.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_ID         = "jinaai/jina-embeddings-v3"
_MAX_TOKENS       = 8192
_WINDOW_TOKENS    = 7500   # tokens per sliding window
_OVERLAP_TOKENS   = 500    # overlap between adjacent windows
_EMBED_DIM        = 1024   # jina-embeddings-v3 output dimension
_SEPARATOR        = "\n"   # separator used when reconstructing full text


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """Outcome of a single diversity selection pass.

    Attributes:
        selected_chunks: Ordered list of selected chunk strings, most
            diverse first.
        selected_indices: Original indices of selected chunks in the
            input list, in selection order.
        embeddings: Late-chunked embeddings for ALL input chunks,
            shape ``[n_chunks, 1024]``. Retained for downstream use
            (e.g. cross-document dedup in Phase 4 Task 3).
        dropped_count: Number of chunks not selected.
        used_sliding_window: Whether the document exceeded the model's
            context window and required windowed encoding.
    """
    selected_chunks:  list[str]
    selected_indices: list[int]
    embeddings:       np.ndarray
    dropped_count:    int
    used_sliding_window: bool = False
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DiversitySelector:
    """Select the most semantically diverse chunks from a document using
    late-chunked jina-embeddings-v3 embeddings.

    The selector is lazy-loaded: the model is not downloaded or moved to
    GPU until the first call to ``select()``. This avoids the ~2.2 GB memory
    cost on runs that disable diversity selection via ``--no-diversity``.

    Args:
        model_id: HuggingFace model identifier. Override only if you want
            to test an alternative long-context embedding model.
        diversity_ratio: Fraction of chunks to keep, in (0.0, 1.0].
            Default 0.7 keeps the 70% most diverse chunks. A document with
            40 chunks yields 28 selected; with 10 chunks yields 7.
        min_chunks: Minimum number of chunks to select regardless of ratio.
            Prevents very short documents from being reduced to 1–2 chunks.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.
    """

    def __init__(
        self,
        model_id:         str   = _MODEL_ID,
        diversity_ratio:  float = 0.7,
        min_chunks:       int   = 5,
        device:           str | None = None,
    ) -> None:
        self.model_id        = model_id
        self.diversity_ratio = diversity_ratio
        self.min_chunks      = min_chunks

        self._device: torch.device | None = None
        self._tokenizer = None
        self._model     = None
        self._target_device = device  # stored; resolved on first use

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer on first use."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        logger.info("Loading %s for diversity selection...", self.model_id)
        print(f"  [DiversitySelector] Loading {self.model_id}...")

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
        logger.info("DiversitySelector loaded on %s.", self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, chunks: list[str]) -> SelectionResult:
        """Select the most diverse subset of chunks via late chunking.

        Encodes the full document (with sliding window for long documents),
        then applies greedy farthest-first selection to maximise pairwise
        cosine distance across the selected set.

        Args:
            chunks: Enriched chunk strings as produced by
                ``DoclingManualExtractor.process_manual`` — each already
                contains the ``[Source Context]`` and ``[Section]`` header.

        Returns:
            ``SelectionResult`` with selected chunks, their indices, and
            all embeddings (for downstream use).
        """
        self._ensure_loaded()

        if not chunks:
            return SelectionResult([], [], np.empty((0, _EMBED_DIM)), 0)

        k = max(self.min_chunks, round(len(chunks) * self.diversity_ratio))
        k = min(k, len(chunks))  # never ask for more than we have

        # Normalize all chunk texts
        norm_chunks = [_normalize(c) for c in chunks]

        # Reconstruct full document text and compute char spans
        full_text, char_spans = _reconstruct_full_text(norm_chunks)

        # Encode — single pass or sliding window
        total_tokens = self._count_tokens(full_text)
        used_window  = total_tokens > _MAX_TOKENS

        if used_window:
            embeddings = self._encode_sliding_window(full_text, char_spans)
        else:
            embeddings = self._encode_single(full_text, char_spans)

        # Greedy farthest-first diversity selection
        selected_indices = _greedy_farthest_first(embeddings, k)

        selected_chunks = [chunks[i] for i in selected_indices]
        dropped_count   = len(chunks) - k

        logger.info(
            "DiversitySelector: %d → %d chunks selected (ratio=%.2f, window=%s).",
            len(chunks), k, self.diversity_ratio, used_window,
        )

        return SelectionResult(
            selected_chunks  = selected_chunks,
            selected_indices = selected_indices,
            embeddings       = embeddings,
            dropped_count    = dropped_count,
            used_sliding_window = used_window,
            stats = {
                "total_chunks":    len(chunks),
                "selected_chunks": k,
                "diversity_ratio": self.diversity_ratio,
                "total_tokens":    total_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Encoding — single pass
    # ------------------------------------------------------------------

    def _encode_single(
        self,
        full_text:  str,
        char_spans: list[tuple[int, int]],
    ) -> np.ndarray:
        """Encode the full document in one pass and pool per chunk."""
        token_embeddings, offset_mapping = self._get_token_embeddings(full_text)
        return _pool_all_spans(token_embeddings, offset_mapping, char_spans)

    # ------------------------------------------------------------------
    # Encoding — sliding window
    # ------------------------------------------------------------------

    def _encode_sliding_window(
        self,
        full_text:  str,
        char_spans: list[tuple[int, int]],
    ) -> np.ndarray:
        """Encode a long document via overlapping windows.

        Each chunk is assigned to the window where it sits most centrally
        (farthest from any window boundary), maximising contextual coverage.
        """
        # Tokenize without truncation to get total offset map
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
        chunk_to_window: list[int] = []
        for cs, ce in char_spans:
            chunk_mid = (cs + ce) / 2
            best = min(
                range(len(windows)),
                key=lambda wi: abs(
                    chunk_mid - (window_char_ranges[wi][0] + window_char_ranges[wi][1]) / 2
                ),
            )
            chunk_to_window.append(best)

        # Encode each window; pool assigned chunks
        n = len(char_spans)
        embeddings: list[np.ndarray | None] = [None] * n

        for wi, (wt_start, wt_end) in enumerate(windows):
            assigned = [ci for ci, w in enumerate(chunk_to_window) if w == wi]
            if not assigned:
                continue

            wc_start, wc_end = window_char_ranges[wi]
            window_text = full_text[wc_start:wc_end]

            token_embs, offset_map = self._get_token_embeddings(window_text)

            for ci in assigned:
                cs, ce = char_spans[ci]
                local_cs = cs - wc_start
                local_ce = ce - wc_start
                try:
                    ts, te = _char_span_to_token_span(local_cs, local_ce, offset_map)
                    emb = _pool_chunk(token_embs, ts, te)
                    embeddings[ci] = emb.cpu().numpy()
                except ValueError as exc:
                    logger.warning("Chunk %d skipped in window %d: %s", ci, wi, exc)

        # Fill any gaps with zero vectors (should be rare)
        arr = np.zeros((n, _EMBED_DIM), dtype=np.float32)
        for i, emb in enumerate(embeddings):
            if emb is not None:
                arr[i] = emb
        return arr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token_embeddings(
        self, text: str,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Encode text and return token-level hidden states + offset map."""
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKENS,
            return_offsets_mapping=True,
        )
        offset_mapping: list[tuple[int, int]] = enc.pop(
            "offset_mapping"
        ).squeeze(0).tolist()

        input_ids      = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task="retrieval.passage",
            )

        token_embeddings = out.last_hidden_state.squeeze(0)  # [seq, 1024]
        return token_embeddings, offset_mapping

    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text* without truncation."""
        return len(
            self._tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no model dependency)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """NFKC-normalize text to ensure tokenizer alignment."""
    return unicodedata.normalize("NFKC", text)


def _reconstruct_full_text(
    norm_chunks: list[str],
) -> tuple[str, list[tuple[int, int]]]:
    """Join chunks with a separator and return the text + char spans."""
    full_text  = _SEPARATOR.join(norm_chunks)
    char_spans = []
    cursor     = 0
    for chunk in norm_chunks:
        char_spans.append((cursor, cursor + len(chunk)))
        cursor += len(chunk) + len(_SEPARATOR)
    return full_text, char_spans


def _char_span_to_token_span(
    char_start:     int,
    char_end:       int,
    offset_mapping: list[tuple[int, int]],
) -> tuple[int, int]:
    """Map a character span to inclusive-start / exclusive-end token indices."""
    token_start, token_end = None, None
    for idx, (t_start, t_end) in enumerate(offset_mapping):
        if t_start == 0 and t_end == 0 and idx != 0:
            continue  # skip special tokens
        if t_start >= char_start and token_start is None:
            token_start = idx
        if token_start is not None and t_start < char_end:
            token_end = idx + 1
    if token_start is None or token_end is None:
        raise ValueError(
            f"No tokens for char span ({char_start}, {char_end}). "
            "Check for truncation or normalization mismatch."
        )
    return token_start, token_end


def _pool_chunk(
    token_embeddings: torch.Tensor,
    token_start:      int,
    token_end:        int,
) -> torch.Tensor:
    """Mean-pool and L2-normalize token embeddings for one chunk."""
    span   = token_embeddings[token_start:token_end, :]
    pooled = span.mean(dim=0)
    return F.normalize(pooled, p=2, dim=0)


def _pool_all_spans(
    token_embeddings: torch.Tensor,
    offset_mapping:   list[tuple[int, int]],
    char_spans:       list[tuple[int, int]],
) -> np.ndarray:
    """Pool embeddings for all chunks in a single-window document."""
    n    = len(char_spans)
    arr  = np.zeros((n, _EMBED_DIM), dtype=np.float32)
    for i, (cs, ce) in enumerate(char_spans):
        try:
            ts, te  = _char_span_to_token_span(cs, ce, offset_mapping)
            arr[i]  = _pool_chunk(token_embeddings, ts, te).cpu().numpy()
        except ValueError as exc:
            logger.warning("Chunk %d skipped: %s", i, exc)
    return arr


def _greedy_farthest_first(embeddings: np.ndarray, k: int) -> list[int]:
    """Select k indices that maximise minimum pairwise cosine distance.

    The Gonzalez algorithm: seed with the chunk whose L2 norm is largest
    (most "extreme" in embedding space), then repeatedly add the chunk
    farthest from all already-selected chunks.  O(n * k) time.

    Args:
        embeddings: Unit-norm embeddings, shape ``[n, dim]``.
        k: Number of chunks to select.

    Returns:
        List of selected indices in selection order (most diverse first).
    """
    n = len(embeddings)
    if k >= n:
        return list(range(n))

    # Seed: chunk with the largest L2 norm (already unit-norm after L2
    # normalisation in _pool_chunk, so this picks the first anchor
    # deterministically without randomness).
    seed = int(np.argmax(np.linalg.norm(embeddings, axis=1)))
    selected = [seed]

    # min_dist[i] = cosine distance from chunk i to its nearest selected chunk
    # cosine distance = 1 - cosine_similarity (embeddings are unit-norm,
    # so cosine_sim = dot product)
    min_dist = 1.0 - embeddings @ embeddings[seed]  # shape [n]
    min_dist[seed] = -1.0  # exclude seed from future selection

    for _ in range(k - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

        # Update min distances: new chunk may be closer to some unselected chunks
        new_dists = 1.0 - embeddings @ embeddings[next_idx]
        min_dist  = np.minimum(min_dist, new_dists)
        min_dist[next_idx] = -1.0

    return selected
