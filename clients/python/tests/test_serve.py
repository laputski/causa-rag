"""causa_rag_client.serve().

Requires the `serve` extra (fastapi/uvicorn) — already present in this
monorepo's own dev environment (services/api_gateway depends on fastapi),
so these tests run unmodified here; a standalone install of
causa-rag-client needs `pip install causa-rag-client[serve]` first.
"""
from __future__ import annotations

from causa_rag_client import serve
from fastapi.testclient import TestClient


def _toy_retrieve(query: str, top_k: int) -> list[dict]:
    return [
        {"doc_id": f"doc-{i}", "chunk_text": f"{query} chunk {i}"}
        for i in range(top_k)
    ]


def _toy_generate(query: str, sources: list[dict]) -> str:
    return f"answer for {query!r} using {len(sources)} sources"


def test_query_matches_external_rag_contract():
    app = serve(_toy_retrieve, _toy_generate)
    client = TestClient(app)
    resp = client.post("/", json={"query": "a test question", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer for 'a test question' using 3 sources"
    assert len(body["sources"]) == 3
    assert body["sources"][0]["doc_id"] == "doc-0"


def test_retrieve_only_returns_sources_without_answer():
    app = serve(_toy_retrieve, _toy_generate)
    client = TestClient(app)
    resp = client.post("/retrieve", json={"query": "a question", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert "sources" in body
    assert "answer" not in body
    assert len(body["sources"]) == 2


def test_no_generate_fn_gives_empty_answer():
    app = serve(_toy_retrieve)
    client = TestClient(app)
    resp = client.post("/", json={"query": "a question", "top_k": 1})
    assert resp.status_code == 200
    assert resp.json()["answer"] == ""


def test_rerank_fn_preserves_pre_rerank_source_refs():
    """Attribution — when rerank_fn is given, the
    pre-rerank list must survive under trace so funnel.py can tell a
    rerank-failure from a retrieval-failure for this RAG."""
    def rerank(query: str, sources: list[dict], top_k: int) -> list[dict]:
        return list(reversed(sources))[:1]

    app = serve(_toy_retrieve, _toy_generate, rerank_fn=rerank)
    client = TestClient(app)
    resp = client.post("/", json={"query": "a question", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sources"]) == 1
    pre_rerank = body["trace"]["pre_rerank_source_refs"]
    assert len(pre_rerank) == 3
    assert body["sources"][0]["doc_id"] == pre_rerank[-1]["doc_id"]


def test_no_rerank_fn_omits_pre_rerank_trace():
    app = serve(_toy_retrieve, _toy_generate)
    client = TestClient(app)
    resp = client.post("/", json={"query": "a question", "top_k": 1})
    assert resp.status_code == 200
    assert "trace" not in resp.json()


def test_tolerates_reference_rag_server_specific_fields():
    """A request built against services/reference_rag_server's optional
    pipeline_id/corpus_id/reranker_id fields must not 400 here — this RAG is
    exactly one strategy, those fields are simply irrelevant, not invalid."""
    app = serve(_toy_retrieve, _toy_generate)
    client = TestClient(app)
    resp = client.post(
        "/", json={"query": "a question", "top_k": 1, "pipeline_id": "graph", "corpus_id": "handbook"},
    )
    assert resp.status_code == 200


def test_health_endpoint():
    app = serve(_toy_retrieve)
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_capabilities_published_when_given():
    caps = {"supports_trace": True, "supports_retrieval_only": True}
    app = serve(_toy_retrieve, _toy_generate, capabilities=caps)
    client = TestClient(app)
    resp = client.get("/capabilities")
    assert resp.status_code == 200
    assert resp.json() == caps


def test_capabilities_endpoint_absent_when_not_given():
    app = serve(_toy_retrieve, _toy_generate)
    client = TestClient(app)
    assert client.get("/capabilities").status_code == 404


# ── corpus_id reaches retrieve_fn when it asks for it ────────────

def test_two_arg_retrieve_fn_never_receives_corpus_id():
    """Backward compat: a RAG serving exactly one corpus keeps its existing
    2-arg signature working unmodified, corpus_id or not in the request."""
    app = serve(_toy_retrieve, _toy_generate)
    client = TestClient(app)
    resp = client.post("/", json={"query": "q", "top_k": 1, "corpus_id": "handbook"})
    assert resp.status_code == 200


def test_three_arg_retrieve_fn_receives_corpus_id():
    seen: list[str | None] = []

    def retrieve_with_corpus(query: str, top_k: int, corpus_id: str | None) -> list[dict]:
        seen.append(corpus_id)
        return [{"doc_id": "d1"}]

    app = serve(retrieve_with_corpus)
    client = TestClient(app)
    resp = client.post("/", json={"query": "q", "top_k": 1, "corpus_id": "handbook"})
    assert resp.status_code == 200
    assert seen == ["handbook"]


def test_three_arg_retrieve_fn_receives_none_when_request_omits_corpus_id():
    seen: list[str | None] = []

    def retrieve_with_corpus(query: str, top_k: int, corpus_id: str | None) -> list[dict]:
        seen.append(corpus_id)
        return []

    app = serve(retrieve_with_corpus)
    client = TestClient(app)
    resp = client.post("/", json={"query": "q", "top_k": 1})
    assert resp.status_code == 200
    assert seen == [None]


def test_three_arg_retrieve_fn_also_used_by_retrieve_only_endpoint():
    seen: list[str | None] = []

    def retrieve_with_corpus(query: str, top_k: int, corpus_id: str | None) -> list[dict]:
        seen.append(corpus_id)
        return []

    app = serve(retrieve_with_corpus)
    client = TestClient(app)
    resp = client.post("/retrieve", json={"query": "q", "top_k": 1, "corpus_id": "docs"})
    assert resp.status_code == 200
    assert seen == ["docs"]


def test_var_positional_retrieve_fn_also_receives_corpus_id():
    """A RAG author using *args instead of a named 3rd param should still
    opt in — arity alone (2 vs 3 fixed params) isn't the only valid shape."""
    seen: list[tuple] = []

    def retrieve_varargs(*args):
        seen.append(args)
        return []

    app = serve(retrieve_varargs)
    client = TestClient(app)
    client.post("/", json={"query": "q", "top_k": 1, "corpus_id": "handbook"})
    assert seen == [("q", 1, "handbook")]
