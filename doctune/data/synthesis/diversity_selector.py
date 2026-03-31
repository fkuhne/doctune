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

Encoding primitives live in ``late_chunker.py``; this module owns only the
selection logic (greedy farthest-first) and the ``SelectionResult`` container.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from doctune.data.synthesis.late_chunker import LateChunker, _EMBED_DIM, _MAX_TOKENS, _MODEL_ID

logger = logging.getLogger(__name__)


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

    The selector is lazy-loaded: the underlying ``LateChunker`` model is
    not downloaded or moved to GPU until the first call to ``select()``.
    This avoids the ~2.2 GB memory cost on runs that disable diversity
    selection via ``--no-diversity``.

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
        self.diversity_ratio = diversity_ratio
        self.min_chunks      = min_chunks
        self._chunker        = LateChunker(model_id=model_id, device=device)

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
        if not chunks:
            return SelectionResult([], [], np.empty((0, _EMBED_DIM)), 0)

        k = max(self.min_chunks, round(len(chunks) * self.diversity_ratio))
        k = min(k, len(chunks))  # never ask for more than we have

        # Encode via late chunking (single pass or sliding window)
        embeddings = self._chunker.encode(chunks)

        # LateChunker.encode() caches the token count — use it to detect
        # whether the sliding window was needed (avoids re-tokenizing)
        total_tokens = self._chunker.last_token_count
        used_window  = total_tokens > _MAX_TOKENS

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

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
    # normalisation in pool_chunk, so this picks the first anchor
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
