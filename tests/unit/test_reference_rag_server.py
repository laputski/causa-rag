"""services/reference_rag_server dog-foods the platform's own
external-RAG HTTP contract: the in-process NaivePipeline, reachable over
exactly the same POST / and POST /retrieve shape any third-party RAG must
implement to be tested by this platform.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.reference_rag_server.main import app


def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_query_matches_external_rag_contract():
    client = TestClient(app)
    resp = client.post("/", json={"query": "a test question", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)


def test_retrieve_only_returns_sources_without_answer():
    client = TestClient(app)
    resp = client.post("/retrieve", json={"query": "a test question", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "sources" in body
    assert "answer" not in body


def test_pipelines_endpoint_lists_builtin_strategies():
    """The built-in hybrid/graph strategies must be
    selectable through the same HTTP contract, not only the dense-only
    default; /pipelines lets a caller discover the valid pipeline_id values."""
    client = TestClient(app)
    resp = client.get("/pipelines")
    assert resp.status_code == 200
    ids = resp.json()["pipeline_ids"]
    assert {"naive", "hybrid_rrf", "hybrid_weighted", "graph"} <= set(ids)


def test_query_with_explicit_pipeline_id_does_not_error():
    client = TestClient(app)
    for pipeline_id in ("naive", "hybrid_rrf", "hybrid_weighted", "graph"):
        resp = client.post("/", json={"query": "a test question", "top_k": 3, "pipeline_id": pipeline_id})
        assert resp.status_code == 200, pipeline_id
        assert "answer" in resp.json()


def _qdrant_answering() -> bool:
    import os
    import socket
    try:
        socket.create_connection(
            (os.getenv("QDRANT_HOST", "localhost"), int(os.getenv("QDRANT_PORT", "6333"))),
            timeout=0.5,
        ).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _qdrant_answering(),
    reason="needs a running Qdrant: without one the server holds the in-memory "
           "stub, and the rebind no-ops on stubs by design, so there is "
           "nothing here to test",
)
def test_corpus_id_rebinds_to_a_distinct_retriever():
    """A non-default corpus_id must produce a retriever
    bound to that corpus, not silently fall back to whatever corpus this
    server started up against (the root cause of a near-zero-recall run when
    the dataset assumed a different corpus_id than the server's startup
    default)."""
    from services.reference_rag_server.main import _resolve_pipeline

    default_pipeline = _resolve_pipeline("naive", "default")
    other_pipeline = _resolve_pipeline("naive", "handbook")

    assert default_pipeline is not other_pipeline
    assert getattr(other_pipeline._retriever, "_corpus_id", None) == "handbook"
    # Same (pipeline_id, corpus_id) combo is cached, not rebuilt per call.
    assert _resolve_pipeline("naive", "handbook") is other_pipeline


def test_query_with_explicit_corpus_id_does_not_error():
    client = TestClient(app)
    resp = client.post("/", json={"query": "a test question", "top_k": 3, "corpus_id": "handbook"})
    assert resp.status_code == 200
    assert "answer" in resp.json()


def test_rerankers_endpoint_lists_at_least_the_stub():
    """The reranker is optional and degrades honestly — only
    cross_encoder_stub/cross_encoder are guaranteed; cross_encoder_local
    depends on the optional [reranker] extra being installed."""
    client = TestClient(app)
    resp = client.get("/rerankers")
    assert resp.status_code == 200
    ids = resp.json()["reranker_ids"]
    assert {"cross_encoder_stub", "cross_encoder"} <= set(ids)


def test_reranker_id_rebinds_to_a_pipeline_with_that_reranker():
    """A reranker_id must produce a pipeline whose
    _reranker is the matching component, not silently ignored the way it
    was before this server ever passed reranker= to NaivePipeline."""
    from services.reference_rag_server.main import _resolve_pipeline

    bare = _resolve_pipeline("naive", "default")
    reranked = _resolve_pipeline("naive", "default", "cross_encoder_stub")

    assert bare._reranker is None
    assert reranked._reranker is not None
    assert reranked._reranker.reranker_id == "cross_encoder_stub"
    # Same (pipeline_id, corpus_id, reranker_id) combo is cached.
    assert _resolve_pipeline("naive", "default", "cross_encoder_stub") is reranked


def test_unknown_reranker_id_degrades_to_no_reranker_not_an_error():
    from services.reference_rag_server.main import _resolve_pipeline

    pipeline = _resolve_pipeline("naive", "default", "no_such_reranker")
    assert pipeline._reranker is None


def test_query_with_explicit_reranker_id_does_not_error():
    client = TestClient(app)
    resp = client.post(
        "/", json={"query": "a test question", "top_k": 3, "reranker_id": "cross_encoder_stub"},
    )
    assert resp.status_code == 200
    assert "answer" in resp.json()


def test_reranked_response_returns_pre_rerank_source_refs():
    """When a reranker ran, the contract response carries
    the pre-rerank list under trace, so the platform's funnel.py can attribute
    a rerank-failure vs a retrieval-failure for this external RAG."""
    client = TestClient(app)
    resp = client.post(
        "/", json={"query": "a test question", "top_k": 3, "reranker_id": "cross_encoder_stub"},
    )
    assert resp.status_code == 200
    trace = resp.json().get("trace", {})
    assert "pre_rerank_source_refs" in trace
    assert isinstance(trace["pre_rerank_source_refs"], list)


def test_no_reranker_omits_pre_rerank_source_refs():
    client = TestClient(app)
    resp = client.post("/", json={"query": "a test question", "top_k": 3})
    assert resp.status_code == 200
    trace = resp.json().get("trace", {})
    assert "pre_rerank_source_refs" not in trace
