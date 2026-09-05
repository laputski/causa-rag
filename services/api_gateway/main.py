"""API Gateway.

Routes:
  POST /query                       → Answer
  GET  /health                      → component registry status
  GET  /registry                    → available components for UI form
  GET  /panels                      → URLs of embedded third-party UIs
  GET  /experiments                 → experiment list
  GET  /experiments/{id}            → experiment detail + trace
  POST /experiments                 → start new experiment run
  POST /experiments/compare         → compare two runs
  WS   /experiments/{id}/progress   → live progress stream
  GET  /datasets                    → golden dataset list
  GET  /datasets/{filename}         → dataset content

Every request gets a trace_id via middleware.
"""
from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adapters.bge_m3 import BgeM3Embedder
from adapters.generator_stub import GeneratorStub
from adapters.qdrant import QdrantRetriever, QdrantRetrieverStub
from core.chunking.fixed import FixedChunkingStrategy
from core.chunking.paragraph import ParagraphChunkingStrategy
from core.chunking.sentence import SentenceChunkingStrategy
from core.chunking.structure_aware import StructureAwareChunkingStrategy
from core.models import Answer, QueryRequest
from core.pipeline import NaivePipeline
from core.registry import registry
from core.retrieval.hybrid import HybridRetriever
from services.api_gateway.routers import atlas as atlas_router
from services.api_gateway.routers import corpus as corpus_router
from services.api_gateway.routers import (
    datasets,
    experiments,
    external_rags,
    feedback,
    generation,
    prompts,
)
from services.api_gateway.routers import domain_packs as domain_packs_router
from services.api_gateway.routers import judgments as judgments_router
from services.api_gateway.routers import production as production_router
from services.api_gateway.routers import realms as realms_router
from services.api_gateway.routers import settings as settings_router

log = structlog.get_logger()

_PANELS = {
    "langfuse": os.getenv("LANGFUSE_EXTERNAL_HOST", "http://localhost:3001"),
    "qdrant": os.getenv("QDRANT_DASHBOARD_URL", "http://localhost:6333/dashboard"),
    "opensearch": os.getenv("OPENSEARCH_DASHBOARD_URL", "http://localhost:5601"),
    "deepeval": os.getenv("DEEPEVAL_URL", "http://localhost:8081/report/deepeval"),
    # `/browser/`, and not the root: at the root Neo4j serves its REST API's
    # JSON document, which carries no anti-framing headers, so the probe judged
    # the panel embeddable and the frame then opened the wrong page.
    # `/browser/` itself sends `X-Frame-Options: DENY` and
    # `frame-ancestors 'none'`: it cannot be embedded, and now that is visible
    # before anybody tries.
    "neo4j": os.getenv("NEO4J_BROWSER_URL", "http://localhost:7474/browser/"),
}

_USE_OLLAMA = os.getenv("USE_OLLAMA_GENERATOR", "true").lower() == "true"
_USE_REAL_QDRANT = os.getenv("USE_REAL_QDRANT", "true").lower() == "true"


def _build_generator():
    if _USE_OLLAMA:
        try:
            import httpx
            httpx.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags", timeout=2).raise_for_status()
            from adapters.ollama_generator import OllamaGenerator
            gen = OllamaGenerator()
            log.info("gateway.generator", kind="ollama", model=gen._model)
            return gen
        except Exception as e:
            log.warning("gateway.generator.ollama_unavailable", error=str(e), fallback="stub")
    gen = GeneratorStub()
    log.info("gateway.generator", kind="stub")
    return gen


def _build_retriever(embedder: BgeM3Embedder):
    if _USE_REAL_QDRANT:
        try:
            from qdrant_client import QdrantClient
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            QdrantClient(host=host, port=port, timeout=3, check_compatibility=False).get_collections()
            retr = QdrantRetriever(
                host=host,
                port=port,
                strategy_id=os.getenv("QDRANT_STRATEGY", "structure_aware"),
                embedder_id=embedder.embedder_id,
            )
            log.info("gateway.retriever", kind="qdrant", host=host, port=port)
            return retr
        except Exception as e:
            log.warning("gateway.retriever.qdrant_unavailable", error=str(e), fallback="stub")
    retr = QdrantRetrieverStub()
    log.info("gateway.retriever", kind="stub")
    return retr


def _build_hybrid_retriever(dense_retriever: Any, embedder: BgeM3Embedder, merge: str) -> Any:
    try:
        from adapters.opensearch import OpenSearchRetriever
        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        strategy_id = os.getenv("QDRANT_STRATEGY", "structure_aware")
        sparse = OpenSearchRetriever(host=host, port=port, strategy_id=strategy_id)
        retr = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse,
            embedder=embedder,
            merge=merge,
        )
        log.info("gateway.retriever.hybrid", merge=merge)
        return retr
    except Exception as e:
        log.warning("gateway.hybrid_retriever.unavailable", merge=merge, error=str(e))
        return dense_retriever


def _register_optional_components() -> None:
    """Register the reranker / grounder / route_policy components.

    Heavy/optional rerankers (sentence-transformers) are registered only when
    their dependency is installed — the platform always starts.
    """
    from adapters.reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerLocal,
        CrossEncoderRerankerStub,
    )
    from core.grounding import TokenOverlapGrounder
    from core.routing import NaiveRoutePolicy

    registry.register("reranker", "cross_encoder_stub", CrossEncoderRerankerStub())
    registry.register("reranker", "cross_encoder", CrossEncoderReranker())
    if CrossEncoderRerankerLocal.is_available():
        registry.register("reranker", "cross_encoder_local", CrossEncoderRerankerLocal())
    else:
        log.info("gateway.reranker.local_unavailable", hint="pip install '.[reranker]'")

    registry.register("grounder", "token_overlap", TokenOverlapGrounder())
    registry.register("route_policy", "naive", NaiveRoutePolicy())


def _register_graph_pipeline(dense_retriever: Any, embedder: Any, generator: Any) -> None:
    """Register a functional GraphRAG retriever + pipeline.

    Uses Neo4j when the driver is installed and the server is reachable; otherwise
    falls back to the in-memory graph stub so the option still works in dev/CI.
    """
    from adapters.lightrag import LightRagRetrieverStub
    from core.pipeline import NaivePipeline
    from core.retrieval.graph_hybrid import GraphHybridRetriever

    graph: Any
    try:
        from adapters.neo4j_graph import Neo4jGraphRetriever
        candidate = Neo4jGraphRetriever()
        if candidate.is_available() and candidate.verify():
            graph = candidate
            log.info("gateway.graph.neo4j", uri=candidate._uri)
        else:
            graph = LightRagRetrieverStub()
            log.info("gateway.graph.stub_fallback", reason="neo4j_unavailable")
    except Exception as exc:
        graph = LightRagRetrieverStub()
        log.info("gateway.graph.stub_fallback", reason=str(exc))

    graph_hybrid = GraphHybridRetriever(graph_retriever=graph, base_retriever=dense_retriever)
    registry.register("retriever", "graph_hybrid", graph_hybrid)
    registry.register(
        "pipeline",
        "graph",
        NaivePipeline(retriever=graph_hybrid, embedder=embedder, generator=generator, pipeline_id="graph"),
    )


async def _collect_active_packs_across_realms() -> list[str]:
    """Which domain pack ids to import/register at gateway startup.

    Found live: this used to be just `(await _get_settings_doc()).get(
    "active_packs", [])` — the "global" (no-Realm) settings doc only. But
    ui/src/context/RealmContext.tsx never has a "no Realm" state once at
    least one Realm exists (auto-selects one), so nothing in normal UI use
    ever writes to the "global" doc at all — activating a pack for a real
    Realm via the Domain Packs page had NO effect on startup, ever.

    Confirmed against the running instance: the Realm's own settings doc had
    active_packs=[] while "global" (an orphaned pre-Realm doc) had
    one pack — `/health` showed it in domain_hooks as if active, but
    that Realm's real chat never actually applied any of the pack's hooks,
    since core.pipeline's hooks are resolved from the Realm's OWN settings doc at
    query time (`_build_chat_pipeline`), not "global"'s. The per-Realm gate
    at query time was always correct — only the startup load was too
    narrow. Fixed: union of every Realm's active_packs, plus "global" for
    the pre-Realm/back-compat case, so a pack activated for ANY Realm
    actually gets imported/registered.
    """
    from services.api_gateway.routers.settings import _get_settings_doc

    settings_doc = await _get_settings_doc()
    all_active_packs: set[str] = set(settings_doc.get("active_packs", []))
    try:
        for r in await realms_router.list_realms():
            realm_settings = await _get_settings_doc(r["id"])
            all_active_packs.update(realm_settings.get("active_packs", []))
    except Exception as exc:
        log.warning("gateway.domain_packs.realm_enum_failed", error=str(exc))
    return sorted(all_active_packs)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    embedder = BgeM3Embedder(use_real_model=os.getenv("USE_REAL_BGE_M3", "").lower() == "true")
    dense_retriever = _build_retriever(embedder)
    generator = _build_generator()

    # naive pipeline (dense only)
    naive_pipeline = NaivePipeline(retriever=dense_retriever, embedder=embedder, generator=generator)

    # hybrid pipelines (dense + sparse)
    hybrid_rrf = NaivePipeline(
        retriever=_build_hybrid_retriever(dense_retriever, embedder, "rrf"),
        embedder=embedder,
        generator=generator,
        pipeline_id="hybrid_rrf",
    )
    hybrid_weighted = NaivePipeline(
        retriever=_build_hybrid_retriever(dense_retriever, embedder, "weighted"),
        embedder=embedder,
        generator=generator,
        pipeline_id="hybrid_weighted",
    )

    registry.register("embedder", embedder.embedder_id, embedder)
    registry.register("retriever", dense_retriever.retriever_id, dense_retriever)
    registry.register("generator", generator.generator_id, generator)
    registry.register("pipeline", naive_pipeline.pipeline_id, naive_pipeline)
    registry.register("pipeline", "hybrid_rrf", hybrid_rrf)
    registry.register("pipeline", "hybrid_weighted", hybrid_weighted)
    registry.register("chunker", FixedChunkingStrategy.strategy_id, FixedChunkingStrategy())
    registry.register("chunker", StructureAwareChunkingStrategy.strategy_id, StructureAwareChunkingStrategy())
    registry.register("chunker", SentenceChunkingStrategy.strategy_id, SentenceChunkingStrategy())
    registry.register("chunker", ParagraphChunkingStrategy.strategy_id, ParagraphChunkingStrategy())

    # Configurable pipeline steps (resolved by ExperimentRunner from config).
    _register_optional_components()
    _register_graph_pipeline(dense_retriever, embedder, generator)

    # Domain pack plugins (directory-scan discovery).
    # Activation takes effect on next restart: packs are loaded here, once,
    # at startup — there is no live-reload mechanism in this stage.
    from core.domain.loader import load_active_packs
    all_active_packs = await _collect_active_packs_across_realms()
    loaded_packs = load_active_packs(registry, {"active_packs": all_active_packs})
    log.info("gateway.domain_packs", loaded=loaded_packs)

    # Seed connector_types catalog on first start
    await realms_router.seed_connector_types()

    # Prompts live in Mongo while the pipeline reads them from files,
    # synchronously and with no database access. Creating a prompt writes to
    # both places; everything else (realm import, migration, an edit outside the
    # gateway) writes to Mongo alone. The files fall behind silently, and a
    # realm with no file of its own picks up another realm's active prompt
    # through the last-resort fallback. Bring the files to the database at
    # startup.
    try:
        import adapters.mongodb as mdb
        from core.prompt_store import prompt_store
        synced = prompt_store.sync_from(await mdb.find_many("prompts"))
        if synced:
            log.info("gateway.prompts_synced", count=len(synced), ids=synced)
    except Exception as exc:
        log.warning("gateway.prompts_sync_failed", error=str(exc))

    log.info("gateway.startup", components=registry.list_all())
    yield
    log.info("gateway.shutdown")


app = FastAPI(title="RAG Platform API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(experiments.router)
app.include_router(datasets.router)
app.include_router(feedback.router)
app.include_router(generation.router)
app.include_router(prompts.router)
app.include_router(settings_router.router)
app.include_router(corpus_router.router)
app.include_router(external_rags.router)
app.include_router(external_rags.spec_router)
app.include_router(realms_router.router)
app.include_router(realms_router.connector_types_router)
app.include_router(domain_packs_router.router)
app.include_router(judgments_router.router)
app.include_router(production_router.router)
app.include_router(atlas_router.router)


@app.middleware("http")
async def inject_trace_id(request: Request, call_next: Any) -> Response:
    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    structlog.contextvars.unbind_contextvars("trace_id")
    return response


_trace_cache: dict[str, dict[str, Any]] = {}
_TRACE_CACHE_MAX = 50


class QueryBody(BaseModel):
    text: str
    top_k: int = 5
    filters: dict[str, Any] = {}
    realm_id: str | None = None
    # Explicit choice of which RAG implementation within the
    # Realm should answer. None ⇒ old auto-pick behavior in
    # _resolve_realm_pipeline (exactly one ExternalRag registered).
    external_rag_id: str | None = None
    # Chat's in-process fallback used to always query the startup-time
    # "default" collection regardless of which Realm/corpus the user meant —
    # same decorative-field trap ExperimentConfig.corpus_id had before Stage
    # 12 (see the design notes). Explicit now, not implicit.
    corpus_id: str = "default"
    # Chat used to be permanently dense-only (NaivePipeline over a
    # single Qdrant retriever, no hybrid merge, no graph) with no way to pick
    # anything else, unlike a real experiment run's `retrievalType`. Same
    # ids as core/experiment/config.py#ExperimentConfig.pipeline_id.
    pipeline_id: str = "naive"


async def _build_realm_scoped_retriever(
    realm_id: str | None, corpus_id: str, embedder: Any, pipeline_id: str = "naive",
) -> Any:
    """In-process chat fallback used to always resolve the registry's
    startup-time retriever — built once from env-var Qdrant, on
    corpus_id="default", identically for every Realm. A Realm with its own
    `resources[]` (or its own corpus_id in the shared instance) was
    unreachable from chat entirely without registering an external RAG.

    Reuses corpus.py's Realm-resource lookup (the same one Content/Health/Graph
    already use — [[data-backends]]) rather than duplicating
    it: `_get_realm_resource` returns None when the Realm has no dedicated
    `qdrant`/`opensearch`/`neo4j` resource, and `_resolve_qdrant`/
    `_resolve_neo4j` already degrade to the platform's env-var instance in
    that case — so this covers both "Realm has its own infra" and "Realm
    shares the platform's" with one code path.

    `pipeline_id` mirrors the same ids core/experiment/runner.py's
    registry pipelines use, but builds the dense/sparse/graph retrievers
    fresh and Realm-scoped each time rather than reusing the registry's
    frozen startup-time instances — those are always corpus_id="default" on
    the env-var instance, exactly the leak this function exists to close.
    """
    qdrant_cfg = await corpus_router._get_realm_resource(realm_id, "qdrant") if realm_id else None
    dense = corpus_router._resolve_qdrant(corpus_id, "structure_aware", embedder.embedder_id, qdrant_cfg, realm_id)

    if pipeline_id in ("hybrid_rrf", "hybrid_weighted"):
        try:
            opensearch_cfg = await corpus_router._get_realm_resource(realm_id, "opensearch") if realm_id else None
            from adapters.opensearch import OpenSearchRetriever
            host = opensearch_cfg.get("host", "localhost") if opensearch_cfg else os.getenv("OPENSEARCH_HOST", "localhost")
            port = int(opensearch_cfg.get("port", 9200)) if opensearch_cfg else int(os.getenv("OPENSEARCH_PORT", "9200"))
            sparse = OpenSearchRetriever(
                host=host, port=port, strategy_id="structure_aware", corpus_id=corpus_id, realm_id=realm_id,
            )
            merge = "rrf" if pipeline_id == "hybrid_rrf" else "weighted"
            return HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, embedder=embedder, merge=merge)
        except Exception as e:
            log.warning("chat.hybrid_retriever.unavailable", pipeline_id=pipeline_id, error=str(e))
            return dense

    if pipeline_id == "graph":
        try:
            neo4j_cfg = await corpus_router._get_realm_resource(realm_id, "neo4j") if realm_id else None
            candidate = corpus_router._resolve_neo4j(neo4j_cfg)
            if not (candidate.is_available() and candidate.verify()):
                raise RuntimeError("neo4j unavailable")
            from core.retrieval.graph_hybrid import GraphHybridRetriever
            return GraphHybridRetriever(graph_retriever=candidate, base_retriever=dense)
        except Exception as e:
            log.warning("chat.graph_retriever.unavailable", error=str(e))
            return dense

    return dense


async def _build_realm_scoped_generator(realm_id: str | None) -> Any:
    """Same reasoning as the retriever above, for Ollama. Falls back to the
    shared process-wide generator (registry-resolved once at startup) when
    the Realm has no dedicated `ollama` resource — cheap (no new client),
    and matches the existing behavior for every Realm that doesn't register
    its own model server.
    """
    if realm_id:
        cfg = await corpus_router._get_realm_resource(realm_id, "ollama")
        if cfg:
            from adapters.ollama_generator import OllamaGenerator
            host = cfg.get("host", "localhost")
            port = cfg.get("port", 11434)
            return OllamaGenerator(base_url=f"http://{host}:{port}", model=cfg.get("model"))
    return registry.resolve("generator", "ollama")


async def _build_chat_pipeline(
    active_packs: list[str], realm_id: str | None = None, corpus_id: str = "default",
    pipeline_id: str = "naive",
) -> Any:
    """Wraps the base pipeline with whatever the active domain
    pack(s) provide (route_policy/mask_engine/scorer/refusal_policy), looked
    up generically via the "domain_hooks" kind (no pack-specific id known
    here — domain-neutral by construction). Falls back to the bare
    pipeline if a pack is listed active but wasn't actually loaded at startup
    (e.g. activated in settings, gateway not yet restarted) — never a 500.

    The base retriever/generator are now resolved per-Realm/corpus_id (see
    _build_realm_scoped_retriever/_generator) instead of always reusing the
    registry's single startup-time instances — this is what makes chat able
    to test a Realm other than whichever one the gateway happened to boot
    against. `pipeline_id` picks which retrieval shape
    (naive/hybrid_rrf/hybrid_weighted/graph) that Realm-scoped retriever is.
    """
    embedder = registry.resolve("embedder", "bge_m3")
    retriever = await _build_realm_scoped_retriever(realm_id, corpus_id, embedder, pipeline_id)
    generator = await _build_realm_scoped_generator(realm_id)
    # Chat deliberately applies NO retrieval pins.
    #
    # This used to fetch the active pins from MongoDB on every chat message.
    # That made a served system's answer depend on the platform's database
    # being reachable at query time, which the "stand outside the hot path"
    # invariant forbids: when the lookup
    # failed it returned an empty list and the answer silently changed, with
    # nothing anywhere reporting it. Chat is also the closest thing the
    # platform has to a real request path, so whatever it does is what a
    # system grown from this template inherits.
    #
    # Pins now apply only to an experiment run that explicitly opts in via
    # ExperimentConfig.retrieval_pins_enabled — a what-if on the stand, never
    # a live request. Guarded by
    # tests/unit/test_p1_guardian.py#test_chat_path_applies_no_retrieval_pins.
    base = NaivePipeline(
        retriever=retriever, embedder=embedder, generator=generator, realm_id=realm_id,
    )
    hooks: dict[str, Any] = {}
    for pack_id in active_packs:
        try:
            hooks.update(registry.resolve("domain_hooks", pack_id))
        except KeyError:
            log.warning("gateway.domain_pack_not_loaded", pack_id=pack_id)
    if not hooks:
        return base
    from core.pipeline import ConfigurablePipeline
    # A refusal_policy is only ever consulted when grounding actually ran
    # (core/pipeline.py: `if self._refusal_policy is not None and gr is not
    # None`) — without wiring a grounder here too, a pack's refusal_policy
    # hook silently never fires in chat even though it looks connected.
    # Generic, domain-neutral grounder (registered once at startup,
    # _register_optional_components) — only resolved when actually needed.
    grounder = None
    if hooks.get("refusal_policy") is not None:
        try:
            grounder = registry.resolve("grounder", "token_overlap")
        except KeyError:
            log.warning("gateway.grounder.unavailable", hint="token_overlap grounder not registered")
    return ConfigurablePipeline(
        retriever=base._retriever,
        embedder=base._embedder,
        generator=base._generator,
        route_policy=hooks.get("route_policy"),
        mask_engine=hooks.get("mask_engine"),
        scorer=hooks.get("scorer"),
        grounder=grounder,
        refusal_policy=hooks.get("refusal_policy"),
        realm_id=realm_id,
    )


async def _resolve_realm_pipeline(
    realm_id: str | None, external_rag_id: str | None = None, corpus_id: str = "default",
) -> Any | None:
    """Routes /query to a Realm's ExternalRag via HttpPipeline
    instead of the in-process registry.

    If `external_rag_id` is given (explicit choice from the ChatPage
    implementation selector), routes to that specific RAG — 404 if it
    doesn't exist or isn't registered under `realm_id`. Otherwise falls back
    to the auto-pick: if the realm has exactly one ExternalRag,
    use it; zero or 2+ ⇒ None (in-process fallback).

    `corpus_id` travels through to `HttpPipeline` the same way it already
    does for experiment runs — chat's own selector used
    to have nothing to pass it through to at all.
    """
    if not realm_id:
        return None
    import adapters.mongodb as mdb
    if external_rag_id:
        doc = await mdb.find_one("external_rags", {"id": external_rag_id, "realm_id": realm_id})
        if not doc:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail=f"external_rag_id {external_rag_id!r} not found in Realm {realm_id!r}",
            )
        from adapters.http_pipeline import HttpPipeline
        return HttpPipeline(url=doc["url"], corpus_id=corpus_id, realm_id=realm_id, timeout=doc.get("timeout_s") or 30.0)

    docs = await mdb.find_many("external_rags", query={"realm_id": realm_id})
    docs = [{k: v for k, v in d.items() if k != "_id"} for d in docs]
    if len(docs) != 1:
        return None
    from adapters.http_pipeline import HttpPipeline
    rag = docs[0]
    return HttpPipeline(url=rag["url"], corpus_id=corpus_id, realm_id=realm_id, timeout=rag.get("timeout_s") or 30.0)


@app.post("/query", response_model=Answer)
async def query(body: QueryBody, request: Request) -> Answer:
    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
    req = QueryRequest(text=body.text, top_k=body.top_k, filters=body.filters, trace_id=trace_id)
    log.info("gateway.query", trace_id=trace_id, text=body.text[:100], realm_id=body.realm_id, external_rag_id=body.external_rag_id)
    realm_pipeline = await _resolve_realm_pipeline(body.realm_id, body.external_rag_id, body.corpus_id)
    if realm_pipeline is not None:
        log.info("gateway.query.realm_http", realm_id=body.realm_id)
        import asyncio
        answer = await asyncio.get_event_loop().run_in_executor(None, realm_pipeline.run, req)
    else:
        from services.api_gateway.routers.settings import _get_settings_doc
        settings_doc = await _get_settings_doc(body.realm_id)
        pipeline = await _build_chat_pipeline(
            settings_doc.get("active_packs", []), body.realm_id, body.corpus_id, body.pipeline_id,
        )
        answer = pipeline.run(req)
    log.info("gateway.answer", trace_id=trace_id, answer_len=len(answer.text))
    # cache trace for /trace endpoint
    _trace_cache[trace_id] = {
        "trace_id": trace_id,
        "query": body.text,
        "stage_trace": answer.stage_trace.model_dump() if answer.stage_trace else None,
        "source_refs": [sr.model_dump() for sr in answer.source_refs],
        "rendered_prompt_preview": answer.rendered_prompt_preview,
        "answer_preview": answer.text[:300],
    }
    if len(_trace_cache) > _TRACE_CACHE_MAX:
        oldest = next(iter(_trace_cache))
        del _trace_cache[oldest]
    return answer


@app.get("/trace/latest")
async def get_latest_trace() -> dict[str, Any]:
    if not _trace_cache:
        return {}
    return next(reversed(_trace_cache.values()))


@app.get("/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    from fastapi import HTTPException
    t = _trace_cache.get(trace_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id!r} not found (cache holds last {_TRACE_CACHE_MAX})")
    return t


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "components": registry.list_all()}


@app.get("/registry")
async def get_registry(realm_id: str | None = None) -> dict[str, list[str]]:
    """Available components by kind, narrowed to a realm.

    The registry is one per process — every active pack of every realm loads
    into it, because the gateway serves them all. Activation, though, is each
    realm's own. Without `realm_id` the two facts contradicted each other on
    screen: the new-run form on one Realm offered three component ids that were
    the internals of a pack a different Realm had enabled, and offered
    nothing else — those three kinds had exactly one option each.

    With `realm_id`, another realm's pack components are removed. Everything
    the platform itself provides stays: it belongs to no pack.
    """
    if not realm_id:
        return registry.list_all()
    from core.domain.loader import components_outside_packs
    from services.api_gateway.routers.settings import _get_settings_doc
    settings_doc = await _get_settings_doc(realm_id)
    return components_outside_packs(registry, settings_doc.get("active_packs", []))


@app.get("/panels")
async def get_panels() -> dict[str, str]:
    """Return URLs for third-party UI panels."""
    return {k: v for k, v in _PANELS.items() if v}


@app.get("/panels/status")
async def get_panels_status() -> list[dict[str, Any]]:
    """Whether each tool answers, and whether it permits being framed.

    Both questions are asked here rather than in the browser because neither
    can be answered there. A cross-origin response is opaque to JavaScript, so
    the page cannot read a status code, and it cannot read `X-Frame-Options`
    either — it only finds out by embedding and watching a blank rectangle
    appear.

    Found live, and the reason this endpoint exists: of six panels, Langfuse
    sends `frame-ancestors 'none'` and Qdrant sends `X-Frame-Options: DENY`.
    Neither can ever be embedded, by their own deliberate choice. The page used
    to embed all six anyway, so half of it was blank with nothing saying why.
    """
    import httpx

    async def probe(name: str, url: str) -> dict[str, Any]:
        row: dict[str, Any] = {"id": name, "url": url, "reachable": False,
                               "embeddable": False, "status": None, "blocked_by": None}
        try:
            async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
                response = await client.get(url)
        except Exception:
            return row
        row["reachable"] = True
        row["status"] = response.status_code
        xfo = (response.headers.get("x-frame-options") or "").lower()
        csp = (response.headers.get("content-security-policy") or "").lower()
        if "deny" in xfo or "sameorigin" in xfo:
            row["blocked_by"] = f"X-Frame-Options: {xfo}"
        elif "frame-ancestors 'none'" in csp.replace("  ", " "):
            row["blocked_by"] = "Content-Security-Policy: frame-ancestors 'none'"
        row["embeddable"] = row["blocked_by"] is None
        return row

    return [await probe(k, v) for k, v in _PANELS.items() if v]


@app.get("/report/deepeval")
async def deepeval_report():
    """Render latest DeepEval results as an HTML report."""
    import json
    from pathlib import Path

    from fastapi.responses import HTMLResponse

    results_dir = Path("eval/results")
    files = sorted(results_dir.glob("deepeval_*.json"))
    run_files = sorted((Path("eval/results/runs")).glob("*.json")) if (Path("eval/results/runs")).exists() else []

    def _card(title: str, data: dict) -> str:
        import datetime as _dt

        metrics = data.get("metrics", {})
        per_q = data.get("per_question", [])
        ts = data.get("timestamp")
        # Found live: this used to print the raw epoch float next to the
        # title (e.g. "1784543486.95") — unreadable, and with multiple
        # historical runs sitting side by side there was no way to tell
        # which run was which without opening the JSON file directly.
        ts_label = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        meta_parts = [f"🕐 {ts_label}"]
        if data.get("judge_model"):
            meta_parts.append(f"judge: <code>{data['judge_model']}</code>")
        if data.get("generator_model"):
            meta_parts.append(f"generator: <code>{data['generator_model']}</code>")
        n_q = data.get("n_questions") or len(per_q)
        if n_q:
            meta_parts.append(f"{n_q} questions")
        if data.get("run_id"):
            meta_parts.append(f"run_id: <code>{data['run_id']}</code>")
        meta_html = " · ".join(meta_parts)
        rows = "".join(
            f"<tr><td>{k}</td><td class='num {'good' if v>=0.5 else 'bad'}'>{v:.3f}</td></tr>"
            for k, v in metrics.items()
        )
        q_rows = "".join(
            f"<tr><td>{i+1}</td><td>{q.get('question','')[:80]}</td>"
            f"<td>{q.get('actual_output','')[:120]}</td>"
            f"<td>{q.get('expected_output','')[:80]}</td>"
            f"<td class='{'good' if '[STUB]' not in q.get('actual_output','') else 'bad'}'>"
            f"{'✓' if '[STUB]' not in q.get('actual_output','') else '✗ STUB'}</td></tr>"
            for i, q in enumerate(per_q)
        )
        return f"""
        <div class="card">
          <h2>{title}</h2>
          <div class="meta">{meta_html}</div>
          <table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>
          {'<details><summary>Questions (' + str(len(per_q)) + ')</summary><table class="qtable"><thead><tr><th>#</th><th>Question</th><th>System answer</th><th>Reference</th><th>OK?</th></tr></thead><tbody>' + q_rows + '</tbody></table></details>' if per_q else ''}
        </div>"""

    cards = ""
    for f in reversed(files[-5:]):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            cards += _card(f"DeepEval · {d.get('dataset', f.stem)}", d)
        except Exception:
            pass

    exp_cards = ""
    for f in reversed(run_files[-5:]):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("aggregate_metrics"):
                metrics = d["aggregate_metrics"]
                rows = "".join(
                    f"<tr><td>{k}</td><td class='num {'good' if v>=0.3 else 'bad'}'>{v:.3f}</td></tr>"
                    for k, v in metrics.items()
                )
                exp_cards += f"<div class='card'><h2>Run · {d.get('config_name','?')} <small>{d.get('run_id','')}</small></h2><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table></div>"
        except Exception:
            pass

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>DeepEval Report</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;margin:0;padding:24px}}
  h1{{color:#fff;margin-bottom:24px}} h2{{color:#a78bfa;margin:0 0 6px}} small{{color:#64748b;font-size:12px;margin-left:8px}}
  .meta{{color:#94a3b8;font-size:12px;margin-bottom:14px}} .meta code{{color:#7dd3fc;font-family:monospace}}
  .card{{background:#1e2333;border:1px solid #2d3748;border-radius:8px;padding:20px;margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:8px}}
  th{{background:#2d3748;padding:6px 10px;text-align:left;color:#94a3b8}}
  td{{padding:5px 10px;border-bottom:1px solid #2d3748}}
  .num{{font-weight:600;font-family:monospace}} .good{{color:#4ade80}} .bad{{color:#f87171}}
  .qtable td{{font-size:11px;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  details summary{{cursor:pointer;color:#7dd3fc;margin-top:8px;font-size:13px}}
  .section-title{{color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:24px 0 8px}}
  .empty{{color:#64748b;padding:20px;text-align:center}}
</style>
</head>
<body>
<h1>📊 Evaluation report</h1>
<div class="section-title">DeepEval runs</div>
{cards or '<div class="empty">No data yet. Run: make test-eval</div>'}
<div class="section-title">Experiments (token-overlap metrics)</div>
{exp_cards or '<div class="empty">No data yet. Start a run from the New run page</div>'}
</body></html>"""
    return HTMLResponse(content=html)
