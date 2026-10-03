"""A run that names an embedder reads the index that embedder built.

The collection a corpus lives in is named after the embedder that filled it.
A run could name another embedder for its queries, and the retriever went on
reading the collection of the one the gateway started with: vectors from one
model searched with vectors from another, which the catalogue has an entry
for and which this platform was doing to itself.
"""
from __future__ import annotations

from typing import Any

from core.experiment.runner import _reading_the_index_of
from core.interfaces import ReadingAnEmbeddersIndex
from core.retrieval.graph_hybrid import GraphHybridRetriever
from core.retrieval.hybrid import HybridRetriever


class _Embedder:
    def __init__(self, embedder_id: str) -> None:
        self.embedder_id = embedder_id
        self.vector_size = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class _Dense:
    retriever_id = "dense"

    def __init__(self, embedder_id: str = "start-up") -> None:
        self.embedder_id = embedder_id

    def for_embedder(self, embedder: Any) -> Any:
        return self if embedder.embedder_id == self.embedder_id else _Dense(embedder.embedder_id)

    def retrieve(self, query: str, k: int, filters: Any = None, query_vector: Any = None) -> list[Any]:
        return []


class _Sparse:
    retriever_id = "sparse"

    def retrieve(self, query: str, k: int, filters: Any = None, query_vector: Any = None) -> list[Any]:
        return []


def test_the_merging_retrievers_can_be_bound() -> None:
    assert isinstance(HybridRetriever(_Dense(), _Sparse(), _Embedder("start-up")), ReadingAnEmbeddersIndex)
    assert isinstance(GraphHybridRetriever(_Sparse(), _Dense()), ReadingAnEmbeddersIndex)


def test_a_hybrid_rebinds_its_dense_half_and_embeds_with_the_named_embedder() -> None:
    hybrid = HybridRetriever(_Dense(), _Sparse(), _Embedder("start-up"), merge="rrf")
    named = _Embedder("named")
    bound = _reading_the_index_of(hybrid, named)
    assert bound is not hybrid
    assert bound._dense.embedder_id == "named" and bound._embedder is named
    assert bound._sparse is hybrid._sparse, "the lexical half has no vectors and is left alone"
    assert bound._merge == "rrf"


def test_a_graph_hybrid_rebinds_the_half_that_reads_vectors() -> None:
    graph = GraphHybridRetriever(_Sparse(), _Dense())
    assert _reading_the_index_of(graph, _Embedder("named"))._base.embedder_id == "named"


def test_the_same_embedder_changes_nothing() -> None:
    hybrid = HybridRetriever(_Dense(), _Sparse(), _Embedder("start-up"))
    assert _reading_the_index_of(hybrid, hybrid._embedder) is hybrid


def test_a_retriever_that_reads_no_vectors_is_left_as_it_was() -> None:
    sparse = _Sparse()
    assert _reading_the_index_of(sparse, _Embedder("named")) is sparse


def test_the_check_of_what_the_corpus_covers_reads_the_same_index_as_the_run(monkeypatch) -> None:
    """Found by reading, the moment a corpus could be loaded with another
    embedder: the check enumerated the collection of the gateway's own
    embedder, which for such a corpus is empty, so every question would have
    been classed as asking for a source the corpus lacks and dropped from
    every retrieval metric without a word."""
    import asyncio

    from core.models import Chunk
    from core.registry import registry
    from services.api_gateway.routers import experiments as E

    class _Indexed(_Dense):
        def for_embedder(self, embedder: Any) -> Any:
            return self if embedder.embedder_id == self.embedder_id else _Indexed(embedder.embedder_id)

        def scroll(self, offset: Any = None, limit: int = 100) -> tuple[list[Any], Any]:
            if self.embedder_id != "named":
                return [], None
            return [Chunk(chunk_id="c1", doc_id="d", text="t", strategy_id="s", start_char=0, end_char=1,
                          metadata={"source_code": "base", "article_no": "07"})], None

    class _Pipeline:
        _retriever = _Indexed()

    monkeypatch.setattr(registry, "resolve", lambda kind, name: _Pipeline())
    seen_by_the_gateway_s = asyncio.run(E._build_ref_resolver("acme", "c"))
    seen_by_the_run_s = asyncio.run(E._build_ref_resolver("acme", "c", embedder=_Embedder("named")))
    assert "base/07" in seen_by_the_run_s.known_ref_ids
    assert "base/07" not in getattr(seen_by_the_gateway_s, "known_ref_ids", frozenset())
