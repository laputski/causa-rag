"""Cross-encoder reranker adapter.

Three interchangeable implementations behind the ``Reranker`` protocol:
- ``CrossEncoderRerankerLocal`` — in-process sentence-transformers cross-encoder
  (default; no external server). Requires the optional ``[reranker]`` extra.
- ``CrossEncoderReranker`` — calls a separate reranker-server over HTTP (prod).
- ``CrossEncoderRerankerStub`` — token-overlap scoring for unit tests / air-gap.
"""
from __future__ import annotations

from typing import Any

import httpx

from core.models import ScoredChunk

# Default multilingual (ru/be) cross-encoder — small, CPU-friendly.
DEFAULT_LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerUnavailable(RuntimeError):
    """Raised when an optional reranker dependency is not installed."""


class CrossEncoderReranker:
    """Calls a running cross-encoder reranker HTTP server."""

    reranker_id = "cross_encoder"

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []

        payload = {
            "query": query,
            "texts": [sc.chunk.text for sc in candidates],
        }
        response = httpx.post(
            f"{self._base_url}/rerank",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        scores: list[float] = response.json()["scores"]

        reranked = [
            ScoredChunk(chunk=sc.chunk, score=score, retriever_id=self.reranker_id)
            for sc, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked


class CrossEncoderRerankerLocal:
    """In-process cross-encoder via sentence-transformers (no external server).

    The model is loaded lazily on first ``rerank`` call so that importing this
    module (and starting the gateway) never requires the optional dependency.
    Call :meth:`is_available` to check before registering.
    """

    reranker_id = "cross_encoder_local"

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    @staticmethod
    def is_available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:  # pragma: no cover - exercised via is_available
                raise RerankerUnavailable(
                    "sentence-transformers not installed; "
                    "install the optional extra: pip install '.[reranker]'"
                ) from exc
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, sc.chunk.text) for sc in candidates]
        scores = model.predict(pairs)
        reranked = [
            ScoredChunk(chunk=sc.chunk, score=float(score), retriever_id=self.reranker_id)
            for sc, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked


class CrossEncoderRerankerStub:
    """In-memory stub: rescores by token overlap between query and chunk text."""

    reranker_id = "cross_encoder_stub"

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        query_tokens = set(query.lower().split())
        reranked = [
            ScoredChunk(
                chunk=sc.chunk,
                score=len(query_tokens & set(sc.chunk.text.lower().split())) / max(len(query_tokens), 1),
                retriever_id=self.reranker_id,
            )
            for sc in candidates
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked
