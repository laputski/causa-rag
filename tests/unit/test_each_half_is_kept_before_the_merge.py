"""Each half of a merged retrieval is kept as it was before the merge.

A retriever that merges two sources used to hand back the merged list alone,
with each fragment carrying the score its half gave it. A fragment one half
found and the merge dropped left no trace anywhere, and a fragment's rank
inside its own half was recorded nowhere, so the question "which stage lost
the fragment the question needs" could be answered from the merge onwards and
not before it.
"""
from __future__ import annotations

from typing import Any

from adapters.generator_stub import GeneratorStub
from adapters.lightrag import GraphNode, LightRagRetrieverStub
from core.models import Chunk, QueryRequest, ScoredChunk
from core.pipeline import NaivePipeline
from core.retrieval.graph_hybrid import GraphHybridRetriever
from core.retrieval.hybrid import HybridRetriever


def _sc(cid: str, score: float, source: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id=cid, text=f"text of {cid}"),
                       score=score, retriever_id=source)


class _Half:
    """A single-source retriever whose signature takes no extra keyword, like
    QdrantRetriever: passing it `halves` would raise."""

    def __init__(self, source: str, ranked: list[tuple[str, float]]):
        self.retriever_id = source
        self._ranked = ranked

    def retrieve(self, query: str, k: int, filters: Any = None,
                 query_vector: list[float] | None = None) -> list[ScoredChunk]:
        return [_sc(cid, score, self.retriever_id) for cid, score in self._ranked[:k]]


class _Embedder:
    embedder_id = "fake"
    version = "0"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


def _hybrid() -> HybridRetriever:
    # "d-only" is ranked last by the dense half and absent from the sparse
    # half, so a merge cut to two drops it.
    dense = _Half("dense", [("both", 0.9), ("d-first", 0.8), ("d-only", 0.1)])
    sparse = _Half("sparse", [("s-first", 7.0), ("both", 5.0)])
    return HybridRetriever(dense, sparse, _Embedder(), merge="rrf")


def test_the_hybrid_hands_each_half_back_in_its_own_order() -> None:
    halves: dict[str, list[ScoredChunk]] = {}
    merged = _hybrid().retrieve("q", k=2, halves=halves)

    assert [sc.chunk.chunk_id for sc in halves["dense"]] == ["both", "d-first", "d-only"]
    assert [sc.chunk.chunk_id for sc in halves["sparse"]] == ["s-first", "both"]
    assert "d-only" not in [sc.chunk.chunk_id for sc in merged]


def test_a_fragment_the_merge_dropped_reaches_the_answer_record() -> None:
    """The phenomenon itself: one half found it, the merge dropped it, and
    the answer now says so where it used to say nothing."""
    pipeline = NaivePipeline(retriever=_hybrid(), embedder=_Embedder(),
                             generator=GeneratorStub(), top_k=2)
    answer = pipeline.run(QueryRequest(text="q", top_k=2))

    final = [ref.chunk_id for ref in answer.source_refs]
    dense = [ref.chunk_id for ref in answer.dense_source_refs]
    assert "d-only" in dense and "d-only" not in final
    assert [ref.chunk_id for ref in answer.sparse_source_refs] == ["s-first", "both"]


def test_a_single_source_retriever_is_not_handed_the_mapping() -> None:
    """Its signature takes no such keyword, so passing it would fail every
    question of every dense run."""
    pipeline = NaivePipeline(retriever=_Half("dense", [("a", 0.9), ("b", 0.5)]),
                             embedder=_Embedder(), generator=GeneratorStub(), top_k=2)
    answer = pipeline.run(QueryRequest(text="q", top_k=2))

    assert [ref.chunk_id for ref in answer.source_refs] == ["a", "b"]
    assert answer.dense_source_refs == [] and answer.sparse_source_refs == []


def test_the_graph_merge_keeps_its_graph_half_and_its_base_half() -> None:
    graph = LightRagRetrieverStub()
    graph.add_node(GraphNode("g1", "graph"), "a graph of rules")
    base = _Half("dense", [("b1", 0.9), ("b2", 0.4)])
    halves: dict[str, list[ScoredChunk]] = {}

    GraphHybridRetriever(graph, base, graph_weight=0.4).retrieve(
        "graph", k=3, halves=halves, query_vector=[0.0, 1.0])

    assert [sc.chunk.chunk_id for sc in halves["dense"]] == ["b1", "b2"]
    assert "graph" in halves


def test_a_graph_merge_over_a_hybrid_lets_the_hybrid_file_its_own_halves() -> None:
    graph = LightRagRetrieverStub()
    graph.add_node(GraphNode("g1", "graph"), "a graph of rules")
    halves: dict[str, list[ScoredChunk]] = {}

    GraphHybridRetriever(graph, _hybrid(), graph_weight=0.4).retrieve("q", k=2, halves=halves)

    assert [sc.chunk.chunk_id for sc in halves["sparse"]] == ["s-first", "both"]
    assert [sc.chunk.chunk_id for sc in halves["dense"]][:1] == ["both"]
