"""causa_rag_client.RagPlatformClient.

Run with: python3 -m pytest clients/python/tests -q
(not under the platform's own tests/unit/ — this package has its own
dependency-light test surface, deliberately separate from testpaths=["tests"]
in the root pyproject.toml so it can be exercised without the platform's
heavy deps installed.)
"""
from __future__ import annotations

import json

import httpx
import pytest
from causa_rag_client import (
    CONTRACT_VERSION,
    ContractVersionMismatch,
    RagPlatformClient,
    RagPlatformError,
)


def _transport(handler):
    return httpx.MockTransport(handler)


def _spec_response(version: str = CONTRACT_VERSION) -> httpx.Response:
    return httpx.Response(200, json={"version": version, "schemas": {}})


def _client(handler, **kwargs) -> RagPlatformClient:
    return RagPlatformClient("https://platform.example", transport=_transport(handler), **kwargs)


def test_version_check_passes_on_match():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/external-rag-spec"
        return _spec_response()

    client = _client(handler)
    assert client.check_contract_version() == CONTRACT_VERSION


def test_version_check_raises_on_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return _spec_response(version="2.0.0")

    with pytest.raises(ContractVersionMismatch, match="2.0.0"):
        _client(handler)


def test_version_check_warns_but_does_not_raise_on_minor_difference():
    """An additive, backward-compatible platform bump (e.g. the
    optional trace.embedders field) must not hard-crash an already-deployed
    RAG using this client on its next restart."""
    def handler(request: httpx.Request) -> httpx.Response:
        return _spec_response(version="1.1.0")

    with pytest.warns(UserWarning, match="1.1.0"):
        client = _client(handler)

    assert client is not None


def test_version_check_warns_but_does_not_raise_on_patch_difference():
    def handler(request: httpx.Request) -> httpx.Response:
        return _spec_response(version="1.0.1")

    with pytest.warns(UserWarning, match="1.0.1"):
        client = _client(handler)

    assert client is not None


def test_version_check_still_raises_on_major_difference():
    """Unlike minor/patch, a major bump is the platform's own signal that
    request/response shapes actually changed — must still hard-fail."""
    def handler(request: httpx.Request) -> httpx.Response:
        return _spec_response(version="2.0.0")

    with pytest.raises(ContractVersionMismatch, match="2.0.0"):
        _client(handler)


def test_version_check_raises_when_platform_version_is_unparseable():
    """A malformed/unexpected version string on the platform side can't be
    compared for "same major" at all — falls back to the old strict
    exact-match behavior rather than silently assuming compatibility."""
    def handler(request: httpx.Request) -> httpx.Response:
        return _spec_response(version="unknown")

    with pytest.raises(ContractVersionMismatch, match="unknown"):
        _client(handler)


def test_version_check_skipped_when_disabled():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the network when check_version=False")

    client = RagPlatformClient(
        "https://platform.example", transport=_transport(handler), check_version=False,
    )
    assert client is not None


def test_register_rag_posts_to_external_rags(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        calls.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(201, json={"id": "abc123", "name": "My RAG", "url": "https://rag.example/query"})

    client = _client(handler)
    result = client.register_rag("My RAG", "https://rag.example/query", description="test")

    assert result["id"] == "abc123"
    method, path, body = calls[0]
    assert method == "POST"
    assert path == "/external-rags"
    assert body["name"] == "My RAG"
    assert body["url"] == "https://rag.example/query"
    assert body["description"] == "test"
    assert body["realm_id"] is None


def test_register_rag_passes_realm_id():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        calls.append(json.loads(request.content))
        return httpx.Response(201, json={"id": "abc123", "name": "My RAG", "url": "https://rag.example/query"})

    client = _client(handler)
    client.register_rag("My RAG", "https://rag.example/query", realm_id="demo")

    assert calls[0]["realm_id"] == "demo"


def test_list_rags_passes_realm_id_as_query_param():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        assert request.url.path == "/external-rags"
        assert dict(request.url.params) == {"realm_id": "demo"}
        return httpx.Response(200, json=[{"id": "abc123", "url": "https://rag.example/query"}])

    client = _client(handler)
    result = client.list_rags(realm_id="demo")

    assert result == [{"id": "abc123", "url": "https://rag.example/query"}]


def test_list_rags_omits_realm_id_param_when_not_given():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        assert dict(request.url.params) == {}
        return httpx.Response(200, json=[])

    client = _client(handler)
    client.list_rags()


def test_upload_dataset_posts_to_unified_datasets_collection():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(201, json={"id": "ds1", "source_rag_id": "abc123"})

    client = _client(handler)
    questions = [{"id": "q1", "question": "Что?", "answerability": "answerable"}]
    result = client.upload_dataset("abc123", questions, filename="toy.jsonl", realm_id="demo")

    assert result["id"] == "ds1"
    path, body = calls[0]
    assert path == "/datasets"
    assert body["filename"] == "toy.jsonl"
    assert body["realm_id"] == "demo"
    assert body["source_rag_id"] == "abc123"
    assert body["questions"] == questions


def test_get_corpus_embedders_returns_matching_entrys_embedder_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        assert request.url.path == "/corpus/collections"
        assert dict(request.url.params) == {"realm_id": "demo"}
        return httpx.Response(200, json=[
            {
                "realm_id": "demo", "corpus_id": "handbook",
                "backends": {
                    "qdrant": {"collection": "demo__handbook__structure_aware__bge_m3", "embedder_id": "bge_m3"},
                    "opensearch": {"index": "rag__demo__handbook__structure_aware"},
                },
            },
            {"realm_id": "demo", "corpus_id": "other", "backends": {}},
        ])

    client = _client(handler)
    result = client.get_corpus_embedders("demo", "handbook")

    assert result == ["bge_m3"]


def test_get_corpus_embedders_collects_all_non_empty_backends():
    """A corpus can legitimately have more than one embedder — e.g. a
    GraphRAG-type corpus has a separate embedder for Neo4j community
    embeddings, distinct from the Qdrant chunk embedder."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        return httpx.Response(200, json=[
            {
                "realm_id": "acme", "corpus_id": "graphrag_docs",
                "backends": {
                    "qdrant": {"embedder_id": "bge_m3"},
                    "neo4j": {"uri": "bolt://localhost:7476", "embedder_id": "nomic-embed-text"},
                },
            },
        ])

    client = _client(handler)
    result = client.get_corpus_embedders("acme", "graphrag_docs")

    assert set(result) == {"bge_m3", "nomic-embed-text"}


def test_get_corpus_embedders_empty_when_corpus_not_registered():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        return httpx.Response(200, json=[])

    client = _client(handler)
    result = client.get_corpus_embedders("demo", "unknown_corpus")

    assert result == []


def test_run_experiment_requires_exactly_one_target():
    def handler(request: httpx.Request) -> httpx.Response:
        return _spec_response()

    client = _client(handler)
    with pytest.raises(ValueError):
        client.run_experiment(name="x", dataset_name="d")
    with pytest.raises(ValueError):
        client.run_experiment(
            name="x", dataset_name="d", rag_id="r1", http_endpoint="https://rag.example/q",
        )


def test_run_experiment_posts_http_pipeline_source_and_external_rag_id():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"run_id": "run1", "config_hash": "deadbeef", "status": "running"})

    client = _client(handler)
    result = client.run_experiment(
        name="toy_run", dataset_name="toy.jsonl", rag_id="abc123",
        corpus_id="handbook", pipeline_id="graph", reranker_id="cross_encoder_stub", top_k=10,
    )

    assert result == {"run_id": "run1", "config_hash": "deadbeef", "status": "running"}
    body = calls[0]
    cfg = body["config"]
    assert body["dataset_name"] == "toy.jsonl"
    assert cfg["pipeline_source"] == "http"
    assert cfg["external_rag_id"] == "abc123"
    assert cfg["corpus_id"] == "handbook"
    assert cfg["pipeline_id"] == "graph"
    assert cfg["top_k"] == 10
    assert cfg["reranker"] == {"kind": "reranker", "component_id": "cross_encoder_stub"}
    assert "http_endpoint" not in cfg


def test_run_experiment_with_inline_http_endpoint_omits_external_rag_id():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"run_id": "run2", "config_hash": "x", "status": "running"})

    client = _client(handler)
    client.run_experiment(name="toy_run", dataset_name="toy.jsonl", http_endpoint="https://rag.example/query")

    cfg = calls[0]["config"]
    assert cfg["http_endpoint"] == "https://rag.example/query"
    assert "external_rag_id" not in cfg


def test_get_results_running_then_done():
    responses = [
        httpx.Response(200, json={"run_id": "run1", "status": "running", "progress": {"processed": 1, "total": 3}}),
        httpx.Response(200, json={"run_id": "run1", "status": "done", "aggregate_metrics": {"correct_refusal": 1.0}}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        return responses.pop(0)

    client = _client(handler)
    first = client.get_results("run1")
    assert first["status"] == "running"
    second = client.get_results("run1")
    assert second["status"] == "done"


def test_wait_for_completion_polls_until_done(monkeypatch):
    statuses = iter(["running", "running", "done"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        return httpx.Response(200, json={"run_id": "run1", "status": next(statuses), "aggregate_metrics": {}})

    client = _client(handler)
    monkeypatch.setattr("time.sleep", lambda _: None)
    result = client.wait_for_completion("run1", poll_interval=0)

    assert result["status"] == "done"


def test_wait_for_completion_times_out():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        return httpx.Response(200, json={"run_id": "run1", "status": "running"})

    client = _client(handler)
    with pytest.raises(TimeoutError):
        client.wait_for_completion("run1", poll_interval=0, timeout=0)


def test_error_response_raises_rag_platform_error_with_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/external-rag-spec":
            return _spec_response()
        return httpx.Response(404, json={"detail": "Run 'nope' not found"})

    client = _client(handler)
    with pytest.raises(RagPlatformError, match="Run 'nope' not found"):
        client.get_results("nope")
