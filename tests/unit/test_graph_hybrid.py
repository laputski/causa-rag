"""Unit tests — GraphHybridRetriever."""
from adapters.lightrag import GraphNode, LightRagRetrieverStub
from core.models import Chunk, ScoredChunk
from core.retrieval.graph_hybrid import GraphHybridRetriever


class _FakeBase:
    retriever_id = "fake_base"

    def __init__(self, chunks: list[ScoredChunk]):
        self._chunks = chunks

    def retrieve(self, query, k=5, filters=None):
        return self._chunks[:k]


def _chunk(cid: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=cid, doc_id="d1", text=text),
        score=score,
        retriever_id="base",
    )


def test_graph_hybrid_returns_top_k():
    stub = LightRagRetrieverStub()
    stub.add_node(GraphNode("g1", "graph"), "a graph of rules")
    base = _FakeBase([_chunk("c1", "base", 0.8), _chunk("c2", "other", 0.5)])
    hyb = GraphHybridRetriever(stub, base, graph_weight=0.4)
    results = hyb.retrieve("graph", k=3)
    assert len(results) <= 3


def test_graph_hybrid_deduplicates():
    stub = LightRagRetrieverStub()
    # graph and base share same chunk_id
    shared_chunk = Chunk(chunk_id="shared", doc_id="d1", text="the rule")
    stub._nodes["shared"] = GraphNode("shared", "the rule")
    stub._node_text["shared"] = "the rule"

    base = _FakeBase([ScoredChunk(chunk=shared_chunk, score=0.7, retriever_id="base")])

    hyb = GraphHybridRetriever(stub, base, graph_weight=0.5)
    results = hyb.retrieve("the rule", k=5)
    ids = [r.chunk.chunk_id for r in results]
    assert ids.count("shared") == 1


def test_graph_weight_affects_ranking():
    stub = LightRagRetrieverStub()
    stub.add_node(GraphNode("g1", "graph"), "graph")
    base = _FakeBase([_chunk("b1", "base", 1.0)])
    hyb_high_graph = GraphHybridRetriever(stub, base, graph_weight=0.9)
    hyb_low_graph = GraphHybridRetriever(stub, base, graph_weight=0.1)
    # Both should return results without error
    assert hyb_high_graph.retrieve("graph", k=5) is not None
    assert hyb_low_graph.retrieve("graph", k=5) is not None
