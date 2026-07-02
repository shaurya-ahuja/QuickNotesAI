"""
QuickNotes-AI Re-ranking Service
Cross-encoder re-ranking to sharpen retrieval relevance after hybrid search.
100% Local - No Data Leaves Your Device

This module defines a small, swappable interface (`BaseReranker`) so the
cross-encoder used here can later be replaced with a fine-tuned / LoRA model
without touching the RAG engine. Implement `BaseReranker`, register it via
`set_default_reranker()`, and the pipeline picks it up transparently.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

# The cross-encoder ships inside sentence-transformers (already a core
# dependency), but the model weights are downloaded on first use. We treat the
# reranker as *optional* at runtime: if the import or the model load fails, the
# RAG engine falls back to hybrid retrieval without re-ranking.
try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    CROSS_ENCODER_AVAILABLE = False


class BaseReranker(ABC):
    """
    Abstract re-ranker interface.

    A reranker scores each candidate document against the query and returns a
    list of relevance scores (higher = more relevant), aligned to the input
    order. Swap in a custom implementation (e.g. a fine-tuned LoRA cross-encoder
    or a hosted API) by subclassing this and calling `set_default_reranker()`.
    """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the reranker can currently produce scores."""
        raise NotImplementedError

    @abstractmethod
    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        """
        Score each document against the query.

        Args:
            query: The user's search query.
            documents: Candidate passages to score.

        Returns:
            A list of floats, one per document, aligned to `documents`.
            Higher scores indicate greater relevance.
        """
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker backed by a HuggingFace/PyTorch model.

    Unlike the bi-encoder used for dense retrieval (which embeds query and
    document independently), a cross-encoder reads the (query, document) pair
    jointly, producing a far more accurate relevance signal at the cost of
    speed. It is therefore applied only to the small candidate set returned by
    hybrid retrieval, not the whole corpus.
    """

    #: Small, fast, strong reranker trained on MS MARCO passage ranking.
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu") -> None:
        """
        Args:
            model_name: HuggingFace cross-encoder model id.
            device: Torch device. Defaults to CPU to match the transcription
                service and avoid Apple-Silicon MPS dispatch issues.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self._model: Optional["CrossEncoder"] = None
        self._load_failed = False

    @property
    def is_available(self) -> bool:
        """True unless the library is missing or the model failed to load."""
        return CROSS_ENCODER_AVAILABLE and not self._load_failed

    def _load_model(self) -> None:
        """Lazily load the cross-encoder; disable self on failure."""
        if self._model is not None or self._load_failed:
            return
        if not CROSS_ENCODER_AVAILABLE:
            self._load_failed = True
            return
        try:
            print(f"Loading cross-encoder reranker: {self.model_name}...")
            self._model = CrossEncoder(self.model_name, device=self.device)
            print("Reranker loaded!")
        except Exception as exc:  # pragma: no cover - runtime/network guard
            print(f"Failed to load reranker ({exc}); falling back to hybrid-only.")
            self._load_failed = True

    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        """
        Score candidates with the cross-encoder.

        Returns an empty list if the model is unavailable, signalling the
        caller to keep the pre-rerank ordering.
        """
        if not documents:
            return []

        self._load_model()
        if self._model is None:
            return []

        pairs = [[query, doc] for doc in documents]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:  # pragma: no cover - inference guard
            print(f"Reranker inference failed ({exc}); falling back to hybrid-only.")
            self._load_failed = True
            return []

        # `predict` may return a numpy array; normalise to plain floats.
        result = [float(s) for s in scores]

        # Some model/library version combinations (e.g. certain BERT cross-encoder
        # checkpoints under newer transformers releases) return NaN/inf from the
        # forward pass despite loading cleanly. Sorting by NaN is undefined, so we
        # treat any non-finite output as a failure and fall back to hybrid order.
        if any(not math.isfinite(s) for s in result):
            print(
                f"Reranker produced non-finite scores for model '{self.model_name}'; "
                "disabling reranker and falling back to hybrid-only. "
                "This usually indicates a sentence-transformers/transformers version "
                "mismatch — pin a compatible pair or choose a different reranker model."
            )
            self._load_failed = True
            return []

        return result


# ── Swappable singleton ──────────────────────────────────────────────────────
_default_reranker: Optional[BaseReranker] = None


def get_default_reranker() -> BaseReranker:
    """Get (or lazily create) the process-wide default reranker."""
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = CrossEncoderReranker()
    return _default_reranker


def set_default_reranker(reranker: BaseReranker) -> None:
    """
    Override the default reranker.

    Call this at startup to plug in a fine-tuned / LoRA cross-encoder or a
    remote reranking API without modifying the RAG engine.
    """
    global _default_reranker
    _default_reranker = reranker
