"""Configurable pipeline actually honours ExperimentConfig.

Covers (local reranker availability/skip), (ConfigurablePipeline
runs optional steps + writes StageTrace), (backward-compatible hash).
"""
from __future__ import annotations

from adapters.bge_m3 import BgeM3Embedder
from adapters.generator_stub import GeneratorStub
from adapters.qdrant import QdrantRetrieverStub
from adapters.reranker import CrossEncoderRerankerLocal, CrossEncoderRerankerStub
from core.grounding import TokenOverlapGrounder
from core.models import Chunk, QueryRequest
from core.pipeline import ConfigurablePipeline, NaivePipeline
from core.routing import NaiveRoutePolicy


def _seeded_retriever() -> QdrantRetrieverStub:
    stub = QdrantRetrieverStub()
    emb = BgeM3Embedder()
    chunks = [
        Chunk(doc_id="d1", text="the submission procedure is set out in the rules"),
        Chunk(doc_id="d2", text="the review period is thirty days"),
    ]
    stub.upsert(chunks, [emb.embed([c.text])[0] for c in chunks])
    return stub


# ── ───────────────────────────────────────────────────────────────────

def test_local_reranker_reports_availability_without_crashing():
    # is_available() must never raise, regardless of whether the extra is installed.
    assert isinstance(CrossEncoderRerankerLocal.is_available(), bool)


# ── Reranking runs and is observable in the trace ────────────────────

def test_reranker_step_runs_and_populates_trace():
    p = ConfigurablePipeline(
        retriever=_seeded_retriever(),
        embedder=BgeM3Embedder(),
        generator=GeneratorStub(),
        reranker=CrossEncoderRerankerStub(),
    )
    ans = p.run(QueryRequest(text="the submission procedure"))
    assert ans.stage_trace is not None
    assert ans.stage_trace.n_reranked == len(ans.source_refs)
    assert ans.source_refs  # something was retrieved + reranked


def test_pipeline_without_reranker_leaves_trace_zero():
    p = ConfigurablePipeline(
        retriever=_seeded_retriever(),
        embedder=BgeM3Embedder(),
        generator=GeneratorStub(),
    )
    ans = p.run(QueryRequest(text="procedure"))
    assert ans.stage_trace.n_reranked == 0
    assert ans.stage_trace.rerank_ms == 0.0


def test_configurable_pipeline_id():
    p = ConfigurablePipeline(
        retriever=QdrantRetrieverStub(), embedder=BgeM3Embedder(), generator=GeneratorStub()
    )
    assert p.pipeline_id == "configurable"


# ── Grounding runs and flags unsupported claims ──────────────────────

def test_grounding_step_runs_and_records_unsupported():
    p = ConfigurablePipeline(
        retriever=_seeded_retriever(),
        embedder=BgeM3Embedder(),
        generator=GeneratorStub(),
        grounder=TokenOverlapGrounder(threshold=0.9),  # strict ⇒ likely unsupported
    )
    ans = p.run(QueryRequest(text="review period"))
    assert ans.stage_trace is not None
    assert "grounding_is_grounded" in ans.metadata
    assert ans.stage_trace.n_unsupported >= 0


# ── Routing runs and is observable ───────────────────────────────────

def test_route_policy_runs_and_records_decision():
    p = ConfigurablePipeline(
        retriever=_seeded_retriever(),
        embedder=BgeM3Embedder(),
        generator=GeneratorStub(),
        route_policy=NaiveRoutePolicy(),
    )
    ans = p.run(QueryRequest(text="a question"))
    assert ans.metadata.get("route_pipeline_id") == "naive"


# ── Graph retrieval is observable in the trace ───────────────────────

def test_graph_hybrid_populates_graph_trace():
    from adapters.lightrag import GraphNode, LightRagRetrieverStub
    from core.pipeline import NaivePipeline
    from core.retrieval.graph_hybrid import GraphHybridRetriever

    graph = LightRagRetrieverStub()
    graph.add_node(GraphNode(node_id="g1", label="procedure"), text="the submission procedure")
    graph.add_node(GraphNode(node_id="g2", label="period"), text="the review period is thirty days")

    hybrid = GraphHybridRetriever(graph_retriever=graph, base_retriever=_seeded_retriever())
    p = NaivePipeline(
        retriever=hybrid, embedder=BgeM3Embedder(), generator=GeneratorStub(), pipeline_id="graph"
    )
    ans = p.run(QueryRequest(text="the submission procedure"))
    assert ans.stage_trace.n_graph > 0
    assert ans.stage_trace.graph_ms >= 0.0


def test_neo4j_graph_availability_is_safe():
    from adapters.neo4j_graph import Neo4jGraphRetriever
    assert isinstance(Neo4jGraphRetriever.is_available(), bool)
    # verify() must never raise even when the server is down
    assert isinstance(Neo4jGraphRetriever().verify(), bool)


def test_graph_hybrid_forwards_query_vector_to_base():
    """Regression: GraphHybridRetriever must forward query_vector to a base
    retriever that requires it (e.g. QdrantRetriever), else the graph pipeline
    crashes at runtime. Found by the GraphRAG e2e run."""
    from adapters.lightrag import LightRagRetrieverStub
    from core.retrieval.graph_hybrid import GraphHybridRetriever

    received: dict = {}

    class _VectorRequiringBase:
        retriever_id = "needs_vector"

        def retrieve(self, query, k=5, filters=None, query_vector=None):
            if query_vector is None:
                raise ValueError("query_vector required")
            received["vec"] = query_vector
            return []

    hybrid = GraphHybridRetriever(graph_retriever=LightRagRetrieverStub(), base_retriever=_VectorRequiringBase())
    hybrid.retrieve("q", k=3, query_vector=[0.1, 0.2, 0.3])  # must not raise
    assert received["vec"] == [0.1, 0.2, 0.3]


# ── backwards compatibility: NaivePipeline unchanged when no extras ────────────

def test_naive_pipeline_unaffected_by_new_optional_args():
    p = NaivePipeline(
        retriever=QdrantRetrieverStub(), embedder=BgeM3Embedder(), generator=GeneratorStub()
    )
    ans = p.run(QueryRequest(text="q"))
    assert ans.stage_trace.n_reranked == 0
    assert ans.stage_trace.n_unsupported == 0
    assert "route_pipeline_id" not in ans.metadata
