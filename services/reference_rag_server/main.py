"""Reference RAG server — a thin FastAPI
wrapper around the platform's own in-process pipelines, exposing exactly
the external-RAG HTTP contract (POST /, optional POST /retrieve).

Why this exists: the platform's HTTP-model integration path
(adapters/http_pipeline.py) is otherwise only ever exercised by third-party
systems. Running the platform's OWN reference pipeline through that same
HTTP contract means the contract is dog-fooded by its primary author — drift
between "what the spec says" and "what real responses look like" cannot
happen unnoticed, because this server would itself start failing.

Reuses the exact same degrade-to-stub factories the main gateway uses
(_build_generator/_build_retriever/_build_hybrid_retriever in
services.api_gateway.main, and the same Neo4j-or-stub fallback the gateway's
_register_graph_pipeline uses) so this server behaves identically to the
gateway's own pipeline variants — not a second, divergent implementation.

`pipeline_id` (request field, default "naive") selects which built-in
strategy to dog-food: naive (dense-only), hybrid_rrf/hybrid_weighted (dense+
sparse), or graph (GraphHybridRetriever). This is what lets the built-in
hybrid/graph implementations be registered and tested as an external RAG
through the same HTTP contract, not just the dense-only default.

`corpus_id` (request field, default "default") selects which corpus
namespace to query, mirroring ExperimentConfig.corpus_id for in_process
pipelines (core/experiment/runner.py:_rebind_corpus_id) — without this, an
ExperimentConfig's corpus_id had no field to travel through on the native
HTTP contract at all, so this server always answered from whatever corpus it
started up against (the "default" 2-code corpus) regardless of which corpus
the question set/dataset actually assumed. A run against a golden set built for
some other corpus_id would then near-silently collapse to near-zero recall: the
wrong corpus, reported as a pipeline-quality regression.

`reranker_id` (request field, default None) selects an optional reranker
step, mirroring ExperimentConfig.reranker for in_process pipelines
(core/experiment/runner.py:_build_pipeline's `_maybe("reranker", ...)`).
NaivePipeline already accepts a `reranker=` kwarg and skips the step
entirely when it's None (core/pipeline.py) — this server previously never
passed one, so no in_process run with a reranker configured had an
equivalent on the external-RAG side to dog-food against. An unknown or
unavailable reranker_id (e.g. "cross_encoder_local" without the optional
[reranker] extra installed) degrades to no reranker, never an error —
honest degradation, the same pattern as _maybe().
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from adapters.bge_m3 import BgeM3Embedder
from core.diagnostics import TraceLog, contract_for_pipeline
from core.models import QueryRequest
from core.pipeline import NaivePipeline
from services.api_gateway.main import _build_generator, _build_hybrid_retriever, _build_retriever

app = FastAPI(title="Reference RAG Server (contract dogfood)", version="1.0.0")


def _build_rerankers() -> dict[str, Any]:
    from adapters.reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerLocal,
        CrossEncoderRerankerStub,
    )

    rerankers: dict[str, Any] = {
        "cross_encoder_stub": CrossEncoderRerankerStub(),
        "cross_encoder": CrossEncoderReranker(),
    }
    if CrossEncoderRerankerLocal.is_available():
        rerankers["cross_encoder_local"] = CrossEncoderRerankerLocal()
    return rerankers


_RERANKERS = _build_rerankers()


def _build_graph_pipeline(dense_retriever: Any, embedder: Any, generator: Any) -> NaivePipeline:
    """Mirrors services.api_gateway.main._register_graph_pipeline's retriever
    construction (Neo4j when reachable, else the in-memory stub), without its
    registry side effect — this server has no shared ComponentRegistry.
    """
    from adapters.lightrag import LightRagRetrieverStub
    from core.retrieval.graph_hybrid import GraphHybridRetriever

    graph: Any
    try:
        from adapters.neo4j_graph import Neo4jGraphRetriever
        candidate = Neo4jGraphRetriever()
        graph = candidate if candidate.is_available() and candidate.verify() else LightRagRetrieverStub()
    except Exception:
        graph = LightRagRetrieverStub()

    graph_hybrid = GraphHybridRetriever(graph_retriever=graph, base_retriever=dense_retriever)
    return NaivePipeline(retriever=graph_hybrid, embedder=embedder, generator=generator, pipeline_id="graph")


_embedder = BgeM3Embedder()
_dense_retriever = _build_retriever(_embedder)
_generator = _build_generator()

_PIPELINES: dict[str, NaivePipeline] = {
    "naive": NaivePipeline(
        retriever=_dense_retriever, embedder=_embedder, generator=_generator, pipeline_id="naive",
    ),
    "hybrid_rrf": NaivePipeline(
        retriever=_build_hybrid_retriever(_dense_retriever, _embedder, "rrf"),
        embedder=_embedder, generator=_generator, pipeline_id="hybrid_rrf",
    ),
    "hybrid_weighted": NaivePipeline(
        retriever=_build_hybrid_retriever(_dense_retriever, _embedder, "weighted"),
        embedder=_embedder, generator=_generator, pipeline_id="hybrid_weighted",
    ),
    "graph": _build_graph_pipeline(_dense_retriever, _embedder, _generator),
}

# Lazily-built, rebound copies of _PIPELINES, keyed by (pipeline_id,
# corpus_id, reranker_id) — built once per combination, not per request,
# since rebinding constructs a new Qdrant/OpenSearch client.
_bound_cache: dict[tuple[str, str, str], NaivePipeline] = {}


def _resolve_pipeline(pipeline_id: str, corpus_id: str, reranker_id: str | None = None) -> NaivePipeline:
    base = _PIPELINES.get(pipeline_id, _PIPELINES["naive"])
    reranker = _RERANKERS.get(reranker_id) if reranker_id else None
    if corpus_id == "default" and reranker is None:
        return base
    key = (pipeline_id, corpus_id, reranker_id or "")
    cached = _bound_cache.get(key)
    if cached is not None:
        return cached
    from core.experiment.runner import _rebind_corpus_id
    retriever = base._retriever if corpus_id == "default" else _rebind_corpus_id(base._retriever, corpus_id)
    rebound = type(base)(
        retriever=retriever, embedder=base._embedder, generator=base._generator,
        pipeline_id=base.pipeline_id, reranker=reranker,
    )
    _bound_cache[key] = rebound
    return rebound


class ExternalRagRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, Any] = {}
    trace_id: str | None = None
    pipeline_id: str = "naive"
    corpus_id: str = "default"
    reranker_id: str | None = None
    # Tolerated, not interpreted: this reference server
    # already exposes its knobs as dedicated fields above (pipeline_id/
    # corpus_id/reranker_id), so it declares no supported_params (see
    # GET /capabilities) and simply accepts-and-ignores an unrecognized
    # params dict rather than 422ing a caller that sends one anyway.
    params: dict[str, Any] = {}


# The template's own diagnostic contract, and the trace log that
# makes phase 7 possible. Both switch on from the environment rather than
# being always-on: a trace holds a user's question and fragments of retrieved
# documents, so recording it is a deployment's decision, never a default.
_trace_log = TraceLog(
    capacity=int(os.getenv("RAG_TRACE_CAPACITY", "500")),
    enabled=os.getenv("RAG_TRACE_EXPORT", "").lower() == "true",
)


@app.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    """What this system can report about itself.

    The comment on `ExternalRagRequest.params` has pointed at this endpoint
    for a long time while it did not exist — the platform asked every external
    RAG to declare its capabilities and the reference implementation, whose
    internals are entirely known, declared nothing.

    Derived from a real pipeline object rather than written out by hand, so
    the declaration cannot drift from what the server actually does. The
    default `naive` pipeline is the one described, since the fields that vary
    by `pipeline_id`/`reranker_id` are per-request choices a caller makes.
    """
    contract = contract_for_pipeline(
        _resolve_pipeline("naive", "default"),
        retrieve_endpoint="/retrieve",
        max_top_k=None,
        # Empty on purpose: every knob this server reads has a dedicated
        # field, so nothing arrives through `params`. Declaring a key here
        # that is ignored is precisely how a config change looks applied
        # while doing nothing.
        supported_params=(),
        supports_trace_export=_trace_log.enabled,
    )
    return {
        **contract.to_dict(),
        # `supports_pre_rerank` above describes the default pipeline, which
        # has no reranker. A caller passing reranker_id gets the pre-rerank
        # list for that request; stating it plainly beats letting a reader
        # conclude the snapshot is unavailable outright.
        "pre_rerank_requires": "reranker_id in the request",
        "trace_export_endpoint": "/traces" if _trace_log.enabled else None,
    }


@app.get("/traces")
async def traces(limit: int | None = None) -> dict[str, Any]:
    """Recent query traces, newest first.

    Empty (never an error) when export is off, so a collector polling a
    deployment that has not switched recording on gets a clear "nothing
    here" rather than a failure it has to special-case.
    """
    return {
        "enabled": _trace_log.enabled,
        "capacity": _trace_log.capacity,
        "count": len(_trace_log),
        "traces": _trace_log.export(limit=limit),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/pipelines")
async def pipelines() -> dict[str, list[str]]:
    """Lists the built-in strategies this server can dog-food —
    lets a caller (e.g. ExternalRagsPage.tsx) discover valid `pipeline_id`
    values before registering this server as a tier-1 external RAG.
    """
    return {"pipeline_ids": list(_PIPELINES.keys())}


@app.get("/rerankers")
async def rerankers() -> dict[str, list[str]]:
    """Lists the optional rerankers this server can dog-food — mirrors
    /pipelines for discovering valid `reranker_id` values. "cross_encoder_local"
    is absent when the optional [reranker] extra (sentence-transformers) isn't
    installed — honest about what's actually available, not a
    static list that could claim a reranker this process can't run.
    """
    return {"reranker_ids": list(_RERANKERS.keys())}


@app.post("/")
async def query(body: ExternalRagRequest) -> dict[str, Any]:
    pipeline = _resolve_pipeline(body.pipeline_id, body.corpus_id, body.reranker_id)
    req = QueryRequest(
        text=body.query, top_k=body.top_k, filters=body.filters,
        trace_id=body.trace_id or str(uuid.uuid4()),
    )
    answer = pipeline.run(req)
    # Recorded after the answer exists, so a failure to write the
    # trace can never cost the caller their answer. A no-op while export is
    # off, which is the default.
    _trace_log.record(
        trace_id=req.trace_id, query=body.query, answer=answer,
        created_at=datetime.now(UTC).isoformat(), top_k=body.top_k,
    )

    response: dict[str, Any] = {
        "answer": answer.text,
        "sources": [sr.model_dump() for sr in answer.source_refs],
    }
    if answer.stage_trace is not None:
        trace: dict[str, Any] = {
            "stage_trace": answer.stage_trace.model_dump(),
            "rendered_prompt": answer.rendered_prompt_preview,
        }
        # Expose the pre-rerank list so the platform can attribute a
        # rerank-failure vs a retrieval-failure (funnel.py). Presence of the
        # key is the signal that a reranker ran, so it is emitted whenever one
        # did, even empty. A truthiness check here dropped the empty list,
        # which made "reranker ran, retrieval returned nothing" read exactly
        # like "no reranker was configured", and that is the one distinction
        # the field exists to carry.
        if pipeline._reranker is not None:
            trace["pre_rerank_source_refs"] = [sr.model_dump() for sr in answer.pre_rerank_source_refs]
        response["trace"] = trace
    return response


@app.post("/retrieve")
async def retrieve(body: ExternalRagRequest) -> dict[str, Any]:
    """retrieval-only path: runs the same pipeline's
    retrieval step without paying for generation, matching what
    HttpPipeline.retrieve() expects from any retrieve_endpoint.
    """
    pipeline = _resolve_pipeline(body.pipeline_id, body.corpus_id)
    k = body.top_k or pipeline._top_k
    query_vector = pipeline._embedder.embed([body.query])[0]
    chunks = pipeline._retriever.retrieve(body.query, k=k, filters=body.filters, query_vector=query_vector)
    from core.pipeline import _to_source_refs

    return {"sources": [sr.model_dump() for sr in _to_source_refs(chunks)]}
