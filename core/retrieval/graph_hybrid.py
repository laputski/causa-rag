"""GraphHybridRetriever — merges graph + vector/BM25 results."""
from __future__ import annotations

from typing import Any

from core.models import ScoredChunk


class GraphHybridRetriever:
    """Combines a GraphRetriever with a vector/BM25 HybridRetriever via score fusion.

    Graph chunks are weighted by ``graph_weight``, dense/sparse by
    ``(1 - graph_weight)``. Deduplication is by chunk_id.
    """

    retriever_id = "graph_hybrid"

    def __init__(
        self,
        graph_retriever: Any,
        base_retriever: Any,
        graph_weight: float = 0.4,
        hops: int = 1,
    ) -> None:
        self._graph = graph_retriever
        self._base = base_retriever
        self._graph_weight = graph_weight
        self._hops = hops

    def for_corpus(
        self,
        corpus_id: str,
        realm_id: str | None = None,
        resources: dict[str, dict[str, Any] | None] | None = None,
    ) -> Any:
        """A copy of this whose base half reads `corpus_id`.

        See `core.interfaces.BoundToACorpus`. The graph is left alone: it has
        no corpus partitioning at all, one graph regardless, so there is
        nothing here to bind. How far it walks and how much say it gets are
        carried, and chosen by `with_graph` below.
        """
        base = (self._base.for_corpus(corpus_id, realm_id, resources)
                if hasattr(self._base, "for_corpus") else self._base)
        if base is self._base:
            return self
        return type(self)(
            graph_retriever=self._graph, base_retriever=base,
            graph_weight=self._graph_weight, hops=self._hops,
        )

    def with_graph(
        self, graph_weight: float | None = None, hops: int | None = None,
    ) -> Any:
        """A copy of this walking as asked. See `core.interfaces.WalkingAGraph`.

        These two are the only parameters the graph point has, and neither
        had a field on a configuration at all: the graph pipeline could be
        chosen and could not be varied, so two graph runs could not differ in
        anything a person had set. A pipeline nobody can vary is one on which
        no failure can be staged by a setting.
        """
        wanted = (
            self._graph_weight if graph_weight is None else graph_weight,
            self._hops if hops is None else hops,
        )
        if wanted == (self._graph_weight, self._hops):
            return self
        return type(self)(
            graph_retriever=self._graph, base_retriever=self._base,
            graph_weight=wanted[0], hops=wanted[1],
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,  # e.g. query_vector — forwarded to the base retriever
    ) -> list[ScoredChunk]:
        graph_result = self._graph.retrieve_graph(
            query, k=k, hops=self._hops, filters=filters
        )
        # Forward retriever-specific kwargs (query_vector) to the base retriever,
        # which (e.g. QdrantRetriever) may require them.
        base_results = self._base.retrieve(query, k=k, filters=filters, **kwargs)

        merged: dict[str, ScoredChunk] = {}

        for sc in graph_result.scored_chunks:
            cid = sc.chunk.chunk_id
            # Surface the per-source score in metadata so the trace and the
            # detectors see the real signal (otherwise dense_score reads 0 and the
            # stub-embedder detector false-positives on graph runs).
            sc.chunk.metadata.setdefault("graph_score", sc.score)
            adjusted = ScoredChunk(
                chunk=sc.chunk,
                score=sc.score * self._graph_weight,
                retriever_id=sc.retriever_id,
            )
            merged[cid] = adjusted

        base_weight = 1.0 - self._graph_weight
        for sc in base_results:
            cid = sc.chunk.chunk_id
            sc.chunk.metadata.setdefault("dense_score", sc.score)
            if cid in merged:
                existing = merged[cid]
                merged[cid] = ScoredChunk(
                    chunk=existing.chunk,
                    score=existing.score + sc.score * base_weight,
                    retriever_id=existing.retriever_id,
                )
            else:
                merged[cid] = ScoredChunk(
                    chunk=sc.chunk,
                    score=sc.score * base_weight,
                    retriever_id=sc.retriever_id,
                )

        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return ranked[:k]
