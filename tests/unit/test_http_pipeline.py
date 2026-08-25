"""adapters/http_pipeline.py."""
from __future__ import annotations

import httpx
import pytest

from adapters.http_pipeline import EgressNotAllowedError, HttpPipeline
from core.models import QueryRequest


def _transport(handler):
    return httpx.MockTransport(handler)


def test_outcome_only_when_no_trace_in_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "an answer with no trace"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question", top_k=5))

    assert answer.text == "an answer with no trace"
    assert answer.stage_trace is None
    assert answer.source_refs == []
    assert answer.metadata["external_trace_available"] is False


# Found live: the run configuration panel had no way to show which model an
# external RAG actually used — a real RAG's response body carries this nested
# under "metadata.model" (confirmed against a real response — a first attempt
# at this fix wrongly assumed a top-level "model" field, which a live curl
# against an actual /platform/query showed was always absent), previously
# never read here at all.
def test_generator_model_read_from_response_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "an answer", "metadata": {"model": "qwen3:32b"}})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question", top_k=5))

    assert answer.metadata["generator_model"] == "qwen3:32b"


def test_generator_model_none_when_response_omits_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "an answer with no model"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question", top_k=5))

    assert answer.metadata["generator_model"] is None


# Found live: "metadata" being present but not a dict (a malformed/typo'd
# external contract) made `.get("model")` raise AttributeError, turning a
# perfectly good answer into a per-question failure instead of degrading
# honestly like every other optional field here.
def test_generator_model_none_when_response_metadata_is_not_a_dict():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "an answer", "metadata": "ok"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question", top_k=5))

    assert answer.metadata["generator_model"] is None
    assert answer.text == "an answer"


def test_full_trace_gives_per_stage_diagnostics():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answer": "an answer with a trace",
            "trace": {
                "stage_trace": {"dense_retrieve_ms": 12.3, "generate_ms": 200.0, "total_ms": 220.0, "n_dense": 3},
                "sources": [{"doc_id": "d1", "chunk_id": "c1", "structural_path": "p", "score": 0.9, "chunk_text": "some text"}],
                "rendered_prompt": "the full prompt",
            },
        })

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question", top_k=5))

    assert answer.text == "an answer with a trace"
    assert answer.stage_trace is not None
    assert answer.stage_trace.n_dense == 3
    assert len(answer.source_refs) == 1
    assert answer.source_refs[0].chunk_text == "some text"
    assert answer.rendered_prompt_preview == "the full prompt"
    assert answer.metadata["external_trace_available"] is True


def test_pre_rerank_source_refs_carried_from_trace():
    """The external RAG's pre-rerank list must reach
    Answer.pre_rerank_source_refs so funnel.py can tell a rerank-failure from a
    retrieval-failure for an external RAG, exactly as for in-process."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answer": "an answer",
            "sources": [{"doc_id": "post", "chunk_text": "after rerank"}],
            "trace": {
                "stage_trace": {"n_dense": 5, "n_reranked": 1, "total_ms": 10.0},
                "pre_rerank_source_refs": [
                    {"doc_id": "pre1", "chunk_text": "before rerank 1"},
                    {"doc_id": "pre2", "chunk_text": "before rerank 2"},
                ],
            },
        })

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question", top_k=5))

    assert [sr.doc_id for sr in answer.source_refs] == ["post"]
    assert [sr.doc_id for sr in answer.pre_rerank_source_refs] == ["pre1", "pre2"]


def test_pre_rerank_absent_degrades_to_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "an answer", "sources": [{"doc_id": "d1"}]})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question"))

    assert answer.pre_rerank_source_refs == []


def test_reported_embedders_carried_into_answer_metadata():
    """Trace.embedders must reach Answer.metadata so
    test_external_rag() (and eventually real runs) can compare it against
    the corpus's own registered embedder(s)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answer": "an answer",
            "sources": [{"doc_id": "d1"}],
            "trace": {"embedders": ["bge_m3"]},
        })

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question"))

    assert answer.metadata["reported_embedders"] == ["bge_m3"]


def test_reported_embedders_absent_when_not_returned():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "an answer", "sources": [{"doc_id": "d1"}]})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question"))

    assert answer.metadata["reported_embedders"] is None


def test_retrieve_endpoint_also_carries_reported_embedders():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "sources": [{"doc_id": "d1"}],
            "trace": {"embedders": ["nomic-embed-text"]},
        })

    pipeline = HttpPipeline(
        url="https://external.example/query", retrieve_endpoint="https://external.example/retrieve",
        transport=_transport(handler),
    )
    answer = pipeline.retrieve(QueryRequest(text="a question"))

    assert answer.metadata["reported_embedders"] == ["nomic-embed-text"]


def test_trace_id_forwarded_as_header():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    req = QueryRequest(text="a question", trace_id="trace-123")
    pipeline.run(req)

    assert seen_headers.get("x-trace-id") == "trace-123"


def test_allowlist_rejects_url_before_any_network_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the network when not on allowlist")

    with pytest.raises(EgressNotAllowedError):
        HttpPipeline(
            url="https://not-allowed.example/query",
            allowlist=["https://external.example"],
            transport=_transport(handler),
        )


def test_allowlist_permits_matching_prefix():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(
        url="https://external.example/query",
        allowlist=["https://external.example"],
        transport=_transport(handler),
    )
    answer = pipeline.run(QueryRequest(text="a question"))
    assert answer.text == "ok"


def test_top_level_sources_take_precedence_over_trace_sources():
    """Sources is canonical at the response top
    level; trace.sources (shape) is only a fallback. A response
    that supplies both must use the top-level one."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answer": "an answer",
            "sources": [{"doc_id": "top-level"}],
            "trace": {"sources": [{"doc_id": "nested-fallback"}]},
        })

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question"))

    assert len(answer.source_refs) == 1
    assert answer.source_refs[0].doc_id == "top-level"


def test_falls_back_to_nested_trace_sources_when_no_top_level_sources():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answer": "an answer",
            "trace": {"sources": [{"doc_id": "nested-fallback"}]},
        })

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    answer = pipeline.run(QueryRequest(text="a question"))

    assert len(answer.source_refs) == 1
    assert answer.source_refs[0].doc_id == "nested-fallback"


def test_native_payload_includes_corpus_id_when_set():
    """corpus_id must travel through the native contract
    so a multi-corpus external RAG (e.g. reference_rag_server) knows which
    corpus this run targets, instead of always answering from whichever it
    started up against."""
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(
        url="https://external.example/query", transport=_transport(handler), corpus_id="handbook",
    )
    pipeline.run(QueryRequest(text="a question"))

    assert seen_body["corpus_id"] == "handbook"


def test_native_payload_omits_corpus_id_when_not_set():
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    pipeline.run(QueryRequest(text="a question"))

    assert "corpus_id" not in seen_body


def test_native_payload_includes_realm_id_when_set():
    """realm_id must travel through too, mirroring corpus_id.
    Without it, an external RAG has no safe way to discover per-Realm facts
    about a corpus_id (e.g. which embedder indexed it via
    GET /corpus/collections?realm_id=&corpus_id=) — corpus_id alone isn't
    guaranteed unique across Realms."""
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(
        url="https://external.example/query", transport=_transport(handler),
        corpus_id="handbook", realm_id="demo",
    )
    pipeline.run(QueryRequest(text="a question"))

    assert seen_body["realm_id"] == "demo"


def test_native_payload_omits_realm_id_when_not_set():
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    pipeline.run(QueryRequest(text="a question"))

    assert "realm_id" not in seen_body


def test_tier2_template_injects_realm_id_as_fallback():
    """realm_id reaches the external RAG even when request_template doesn't
    include a {{realm_id}} placeholder, same fallback-injection as corpus_id."""
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(
        url="https://external.example/query", transport=_transport(handler),
        request_template={"q": "{{query}}"},
        corpus_id="handbook", realm_id="demo",
    )
    pipeline.run(QueryRequest(text="a question"))

    assert seen_body["realm_id"] == "demo"


def test_native_payload_includes_pipeline_id_and_reranker_id_when_set():
    """pipeline_id/reranker_id must travel through the
    native contract too, mirroring corpus_id, so presets like "GraphRAG +
    Rerank" actually drive a dog-fooding external RAG (e.g.
    reference_rag_server), not just top_k."""
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(
        url="https://external.example/query", transport=_transport(handler),
        external_pipeline_id="graph", reranker_id="cross_encoder_local",
    )
    pipeline.run(QueryRequest(text="a question"))

    assert seen_body["pipeline_id"] == "graph"
    assert seen_body["reranker_id"] == "cross_encoder_local"


def test_native_payload_omits_pipeline_id_and_reranker_id_when_not_set():
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    pipeline.run(QueryRequest(text="a question"))

    assert "pipeline_id" not in seen_body
    assert "reranker_id" not in seen_body


def test_native_payload_includes_params_when_set():
    """Extensible, capabilities-gated knobs travel
    through the native contract under "params", generalizing the
    pipeline_id/corpus_id/reranker_id pattern instead of growing a new
    top-level field per knob."""
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(
        url="https://external.example/query", transport=_transport(handler),
        params={"fetch_k": 50, "temperature": 0.2},
    )
    pipeline.run(QueryRequest(text="a question"))

    assert seen_body["params"] == {"fetch_k": 50, "temperature": 0.2}


def test_native_payload_omits_params_when_not_set():
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    pipeline.run(QueryRequest(text="a question"))

    assert "params" not in seen_body


def test_tier2_request_template_builds_native_payload():
    """When request_template is set, run() builds the
    POST body from the RAG's own native shape instead of {query, top_k,
    filters, trace_id}."""
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(
        url="https://external.example/query",
        transport=_transport(handler),
        request_template={"q": "{{query}}", "k": "{{top_k}}"},
    )
    pipeline.run(QueryRequest(text="a question", top_k=7))

    assert seen_body == {"q": "a question", "k": 7}


def test_tier2_response_mapping_parses_native_response():
    """When response_mapping is set, run() translates
    the RAG's own native response shape into the platform's canonical
    {"answer", "sources"} via JSONPath before building the Answer."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "result": {"text": "an answer in a foreign shape", "docs": [{"id": "d1", "content": "some text"}]},
        })

    pipeline = HttpPipeline(
        url="https://external.example/query",
        transport=_transport(handler),
        response_mapping={
            "answer": "$.result.text",
            "sources": "$.result.docs",
            "source_doc_id": "$.id",
            "source_text": "$.content",
        },
    )
    answer = pipeline.run(QueryRequest(text="a question"))

    assert answer.text == "an answer in a foreign shape"
    assert len(answer.source_refs) == 1
    assert answer.source_refs[0].doc_id == "d1"
    assert answer.source_refs[0].chunk_text == "some text"


# Found live: a tier-2-mapped external RAG's model name was permanently
# unrecoverable — response_mapping had no way to extract it, since
# apply_response_mapping() discarded everything but answer/sources.
def test_tier2_response_mapping_extracts_generator_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "result": {"text": "an answer", "model_used": "qwen3:32b", "docs": []},
        })

    pipeline = HttpPipeline(
        url="https://external.example/query",
        transport=_transport(handler),
        response_mapping={
            "answer": "$.result.text",
            "sources": "$.result.docs",
            "generator_model": "$.result.model_used",
        },
    )
    answer = pipeline.run(QueryRequest(text="a question"))

    assert answer.metadata["generator_model"] == "qwen3:32b"


def test_tier2_response_mapping_generator_model_absent_when_not_configured():
    """No generator_model key in response_mapping ⇒ "" (honest degradation),
    not an error — mirrors every other unconfigured mapping key here."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"text": "an answer", "docs": []}})

    pipeline = HttpPipeline(
        url="https://external.example/query",
        transport=_transport(handler),
        response_mapping={"answer": "$.result.text", "sources": "$.result.docs"},
    )
    answer = pipeline.run(QueryRequest(text="a question"))

    assert answer.metadata["generator_model"] == ""


def test_tier2_response_mapping_applies_to_retrieve_endpoint_too():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"docs": [{"id": "d1", "content": "some text"}]})

    pipeline = HttpPipeline(
        url="https://external.example/query",
        retrieve_endpoint="https://external.example/retrieve",
        transport=_transport(handler),
        response_mapping={"sources": "$.docs", "source_doc_id": "$.id", "source_text": "$.content"},
    )
    answer = pipeline.retrieve(QueryRequest(text="a question"))

    assert answer.text == ""
    assert len(answer.source_refs) == 1
    assert answer.source_refs[0].doc_id == "d1"


def test_tier2_template_injects_corpus_id_as_fallback():
    """corpus_id must reach the external RAG even when request_template doesn't
    include a {{corpus_id}} placeholder — the pipeline injects it as a fallback
    top-level field so corpus selection never silently falls through to default."""
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(
        url="https://external.example/query",
        transport=_transport(handler),
        request_template={"query": "{{query}}", "top_k": "{{top_k}}", "pipeline_id": "hybrid_rrf"},
        corpus_id="handbook",
    )
    pipeline.run(QueryRequest(text="a question", top_k=10))

    assert seen_body["corpus_id"] == "handbook"
    # pipeline_id from template takes precedence over external_pipeline_id
    assert seen_body["pipeline_id"] == "hybrid_rrf"


def test_tier2_template_corpus_id_placeholder_in_custom_key():
    """When the template uses {{corpus_id}} under a custom key, that value is
    rendered correctly. The fallback also injects 'corpus_id' at top level —
    a RAG that ignores the unknown field still gets correct routing via its
    own custom key (e.g. 'namespace')."""
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(
        url="https://external.example/query",
        transport=_transport(handler),
        request_template={"query": "{{query}}", "namespace": "{{corpus_id}}", "k": "{{top_k}}"},
        corpus_id="handbook",
    )
    pipeline.run(QueryRequest(text="a question", top_k=5))

    assert seen_body["namespace"] == "handbook"
    # corpus_id also appears as standard fallback field
    assert seen_body["corpus_id"] == "handbook"


def test_satisfies_pipeline_protocol():
    from core.interfaces import Pipeline

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "ok"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    assert isinstance(pipeline, Pipeline)


def test_run_4xx_surfaces_response_body_detail_not_just_status_line():
    """Found live against a real external RAG: a 422 was stored/surfaced as
    just "Client error '422 Unprocessable Entity' for url '...'" — plain
    `raise_for_status()` only puts the status code and URL in the message,
    discarding the response body even when (as here, a FastAPI-style
    service) it names exactly what's wrong: 'Unknown corpus_id=\"handbook_01\":
    no matching collection'. That's the one piece of information that
    actually explains the failure."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "Unknown corpus_id='handbook_01': no matching collection"})

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    with pytest.raises(httpx.HTTPStatusError, match="Unknown corpus_id='handbook_01': no matching collection"):
        pipeline.run(QueryRequest(text="a question", top_k=5))


def test_run_4xx_falls_back_to_raw_text_when_body_has_no_detail_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error, not JSON")

    pipeline = HttpPipeline(url="https://external.example/query", transport=_transport(handler))
    with pytest.raises(httpx.HTTPStatusError, match="internal server error, not JSON"):
        pipeline.run(QueryRequest(text="a question", top_k=5))


def test_retrieve_4xx_also_surfaces_response_body_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "top_k must be positive"})

    pipeline = HttpPipeline(
        url="https://external.example/query", retrieve_endpoint="https://external.example/retrieve",
        transport=_transport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError, match="top_k must be positive"):
        pipeline.retrieve(QueryRequest(text="a question", top_k=5))
