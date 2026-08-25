"""HybridRetriever — merges dense (Qdrant) and sparse (OpenSearch) results.

Two merge strategies:
  - RRF (Reciprocal Rank Fusion) — default, parameter-free
  - weighted — explicit alpha for dense, (1-alpha) for sparse
"""
from __future__ import annotations

from typing import Any

from core.models import ScoredChunk


def _rrf_score(rank: int, k: int = 60) -> float:
    """Standard RRF formula: 1 / (k + rank), rank is 1-based."""
    return 1.0 / (k + rank)


class HybridRetriever:
    """Combines a dense and a sparse retriever into a single ranked list.

    Both retrievers must expose `.retrieve(query, k, filters, **kwargs)`.
    The dense retriever also accepts `query_vector` kwarg.
    """

    retriever_id = "hybrid"

    def __init__(
        self,
        dense_retriever: Any,
        sparse_retriever: Any,
        embedder: Any,
        merge: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        if merge not in ("rrf", "weighted"):
            raise ValueError(f"merge must be 'rrf' or 'weighted', got {merge!r}")
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._embedder = embedder
        self._merge = merge
        self._alpha = alpha
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        query_vec = self._embedder.embed([query])[0]
        fetch_k = max(k * 2, 20)

        dense_results = self._dense.retrieve(
            query=query, k=fetch_k, filters=filters, query_vector=query_vec
        )
        sparse_results = self._sparse.retrieve(query=query, k=fetch_k, filters=filters)

        # tag pre-merge scores into metadata for tracing
        dense_score_map = {sc.chunk.chunk_id: sc.score for sc in dense_results}
        sparse_score_map = {sc.chunk.chunk_id: sc.score for sc in sparse_results}

        if self._merge == "rrf":
            merged = self._merge_rrf(dense_results, sparse_results, k)
        else:
            merged = self._merge_weighted(dense_results, sparse_results, k)

        # annotate merged chunks with pre-merge scores
        for sc in merged:
            cid = sc.chunk.chunk_id
            sc.chunk.metadata["dense_score"] = round(dense_score_map.get(cid, 0.0), 4)
            sc.chunk.metadata["sparse_score"] = round(sparse_score_map.get(cid, 0.0), 4)

        return merged

    def _merge_rrf(
        self,
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk],
        k: int,
    ) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        by_id: dict[str, ScoredChunk] = {}
        ranks: dict[str, int] = {}

        for rank, sc in enumerate(dense, start=1):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank, self._rrf_k)
            by_id[cid] = sc

        for rank, sc in enumerate(sparse, start=1):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank, self._rrf_k)
            if cid not in by_id:
                by_id[cid] = sc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        for i, (cid, _) in enumerate(ranked, start=1):
            ranks[cid] = i

        result = [
            ScoredChunk(chunk=by_id[cid].chunk, score=score, retriever_id=self.retriever_id)
            for cid, score in ranked
        ]
        for sc in result:
            sc.chunk.metadata["rrf_rank"] = ranks.get(sc.chunk.chunk_id, 0)
        return result

    def _merge_weighted(
        self,
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk],
        k: int,
    ) -> list[ScoredChunk]:
        def _normalise(results: list[ScoredChunk]) -> dict[str, float]:
            if not results:
                return {}
            max_s = max(s.score for s in results) or 1.0
            return {s.chunk.chunk_id: s.score / max_s for s in results}

        d_norm = _normalise(dense)
        s_norm = _normalise(sparse)
        by_id: dict[str, ScoredChunk] = {sc.chunk.chunk_id: sc for sc in dense + sparse}

        all_ids = set(d_norm) | set(s_norm)
        scores = {
            cid: self._alpha * d_norm.get(cid, 0.0) + (1 - self._alpha) * s_norm.get(cid, 0.0)
            for cid in all_ids
        }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            ScoredChunk(chunk=by_id[cid].chunk, score=score, retriever_id=self.retriever_id)
            for cid, score in ranked
        ]
