"""HybridRetriever — merges dense (Qdrant) and sparse (OpenSearch) results.

Two merge strategies:
  - RRF (Reciprocal Rank Fusion) — default, parameter-free
  - weighted — explicit alpha for dense, (1-alpha) for sparse
"""
from __future__ import annotations

import time
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

    #: Fills a `timings` mapping when the caller passes one, so what each
    #: half cost is measured where it happens. The pipeline used to infer
    #: the split from the identifier carried by the merged results, and
    #: those carry "hybrid": the condition was false on every hybrid run
    #: ever recorded, so the whole retrieval time was written down as the
    #: dense half's, the sparse half's time was zero, and the merge's was a
    #: quarter of the total, which nobody had measured. A detector looking
    #: for a stage that ran and reported no time found it.
    reports_stage_timings = True

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

    def for_corpus(
        self,
        corpus_id: str,
        realm_id: str | None = None,
        resources: dict[str, dict[str, Any] | None] | None = None,
    ) -> Any:
        """A copy of this whose halves read `corpus_id`.

        See `core.interfaces.BoundToACorpus`. A wrapper carries no corpus of
        its own, so it rebinds what it is made of and keeps how it fuses:
        binding a corpus and choosing a fusion are two acts, and doing both
        here would let one undo the other.

        A half that cannot be rebound comes back as it was, which is how a
        stub and a store behave alike under this.
        """
        halves = [
            half.for_corpus(corpus_id, realm_id, resources)
            if hasattr(half, "for_corpus") else half
            for half in (self._dense, self._sparse)
        ]
        if halves == [self._dense, self._sparse]:
            return self
        return type(self)(
            dense_retriever=halves[0], sparse_retriever=halves[1],
            embedder=self._embedder, merge=self._merge,
            alpha=self._alpha, rrf_k=self._rrf_k,
        )

    def with_fusion(
        self, merge: str | None = None, alpha: float | None = None, rrf_k: int | None = None,
    ) -> Any:
        """A copy of this fusing as asked, around the same two halves.

        See `core.interfaces.Fusing`. Built around the halves it already has
        instead of new ones, so this composes with a corpus binding that has
        already happened.

        The fusion constant is carried and never defaulted. Rebuilding
        without it reset it to sixty, so asking for a different weight
        silently undid a different constant set anywhere upstream: measured,
        a retriever built with rrf_k=10 came back with 60.

        An unknown strategy raises in the constructor, and this returns the
        retriever as built instead: a whole run should not be lost to one
        mistyped field, and the run records what it used.
        """
        wanted = (
            merge or self._merge,
            self._alpha if alpha is None else alpha,
            self._rrf_k if rrf_k is None else rrf_k,
        )
        if wanted == (self._merge, self._alpha, self._rrf_k):
            return self
        try:
            return type(self)(
                dense_retriever=self._dense, sparse_retriever=self._sparse,
                embedder=self._embedder,
                merge=wanted[0], alpha=wanted[1], rrf_k=wanted[2],
            )
        except Exception:
            return self

    def retrieve(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        # A mapping the caller owns, filled here and never held on this
        # object: one retriever answers many queries at once, and a field
        # on it would report whichever query finished last.
        timings = kwargs.get("timings")
        query_vec = self._embedder.embed([query])[0]
        fetch_k = max(k * 2, 20)

        started = time.perf_counter()
        dense_results = self._dense.retrieve(
            query=query, k=fetch_k, filters=filters, query_vector=query_vec
        )
        dense_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        sparse_results = self._sparse.retrieve(query=query, k=fetch_k, filters=filters)
        sparse_ms = (time.perf_counter() - started) * 1000

        # tag pre-merge scores into metadata for tracing
        dense_score_map = {sc.chunk.chunk_id: sc.score for sc in dense_results}
        sparse_score_map = {sc.chunk.chunk_id: sc.score for sc in sparse_results}

        started = time.perf_counter()
        if self._merge == "rrf":
            merged = self._merge_rrf(dense_results, sparse_results, k)
        else:
            merged = self._merge_weighted(dense_results, sparse_results, k)
        merge_ms = (time.perf_counter() - started) * 1000

        if timings is not None:
            timings.update({
                "dense_ms": round(dense_ms, 1),
                "sparse_ms": round(sparse_ms, 1),
                "merge_ms": round(merge_ms, 1),
                "n_dense": len(dense_results),
                "n_sparse": len(sparse_results),
            })

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
