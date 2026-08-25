"""serve(): expose the platform's tier-1 native contract
(POST /, optional POST /retrieve, GET /health) over a few user-supplied
functions, with zero hand-written FastAPI.

Mirrors services/reference_rag_server/main.py's routes/shapes, but driven by
plain Python callables instead of an in-process pipeline — this module never
imports core/adapters (no torch, no Qdrant/Mongo); the HTTP contract is the
only thing shared with the platform.

fastapi/pydantic are optional (the `serve` extra) — importing this module
must not require them, only calling serve()/run_server() does. They're
imported here at module scope (not nested inside serve()) because FastAPI
needs to resolve ContractRequest's field annotations against real module
globals; a pydantic model class defined inside a function body breaks that
resolution (`PydanticUserError: ... is not fully defined`). The try/except
below is what keeps the import optional despite that constraint.
"""
from collections.abc import Callable
from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    class ContractRequest(BaseModel):
        query: str
        top_k: int = 5
        filters: dict = {}
        trace_id: str | None = None
        # Tolerated, not interpreted — a caller built against
        # services/reference_rag_server's optional fields (pipeline_id/
        # corpus_id/reranker_id select between MULTIPLE built-in strategies
        # there; this RAG is exactly one strategy) shouldn't 400 here.
        pipeline_id: str | None = None
        corpus_id: str | None = None
        reranker_id: str | None = None

    _SERVE_AVAILABLE = True
except ImportError:
    _SERVE_AVAILABLE = False

# corpus_id is Optional[str] and LAST — a RAG serving exactly one corpus can
# keep the 2-arg (query, top_k) signature unchanged (see _accepts_corpus_id
# below); a RAG that serves several corpora accepts it as the 3rd param and
# does its own explicit corpus_id -> real storage mapping (a dict/config it
# owns), rather than assuming corpus_id IS a literal collection/index name —
# that assumption is exactly what broke a real integration (a platform
# corpus_id like "handbook" doesn't equal any given RAG's own naming
# scheme; see the design notes "corpus_id must travel through the
# native contract too").
RetrieveFn = Callable[..., list]
GenerateFn = Callable[[str, list], str]
RerankFn = Callable[[str, list, int], list]


def _accepts_corpus_id(fn: Callable[..., Any]) -> bool:
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values())
    return len(positional) >= 3 or has_var_positional


def serve(
    retrieve_fn: RetrieveFn,
    generate_fn: GenerateFn | None = None,
    *,
    rerank_fn: RerankFn | None = None,
    capabilities: dict | None = None,
) -> Any:
    """Builds and returns a FastAPI app implementing the platform's tier-1
    native contract — pass the result to run_server() or any ASGI server.

    `retrieve_fn(query, top_k)` or `retrieve_fn(query, top_k, corpus_id)` ->
    list of source dicts; each needs at least `doc_id` to be usable for
    retrieval metrics on the platform side (see the design notes Tier
    2's "limit of this tier" — the same constraint applies here: no doc_id,
    no recall/precision). Whether `corpus_id` (Optional[str], may be None)
    is passed is decided once, at serve()-build time, by inspecting
    `retrieve_fn`'s own signature — a 2-arg function keeps working exactly
    as before (honest degradation: a RAG that never asked for
    corpus_id was never silently handed a routing responsibility it didn't
    opt into), a 3-arg function receives whatever corpus_id the platform
    request carried (including None when the caller didn't set one).

    `generate_fn(query, sources)` -> answer text. Omit for a retrieval-only
    RAG — POST / then returns an empty answer (pair with the platform's
    ExperimentConfig.retrieval_only, or just call /retrieve directly).

    `rerank_fn(query, sources, top_k)` -> reranked sources. When given, the
    pre-rerank list is preserved under `trace.pre_rerank_source_refs` so
    core/eval/funnel.py can attribute a rerank-failure separately from a
    retrieval-failure for this RAG (the exact gap that
    field exists to close; see the design notes "pre_rerank_source_refs
    — layer attribution for an external RAG").

    `capabilities`, when given, is published verbatim at GET /capabilities so
    a registrant doesn't have to guess what this RAG supports.
    """
    if not _SERVE_AVAILABLE:
        raise ImportError(
            "serve() requires the 'serve' extra: pip install causa-rag-client[serve]"
        )

    app = FastAPI(title="causa-rag-client serve()", version="1.0.0")
    _retrieve_wants_corpus_id = _accepts_corpus_id(retrieve_fn)

    def _do_retrieve(query: str, top_k: int, corpus_id: str | None) -> list:
        if _retrieve_wants_corpus_id:
            return retrieve_fn(query, top_k, corpus_id)
        return retrieve_fn(query, top_k)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    if capabilities is not None:
        @app.get("/capabilities")
        async def get_capabilities() -> dict:
            return capabilities

    @app.post("/retrieve")
    async def retrieve(body: ContractRequest) -> dict:
        return {"sources": _do_retrieve(body.query, body.top_k, body.corpus_id)}

    @app.post("/")
    async def query(body: ContractRequest) -> dict:
        sources = _do_retrieve(body.query, body.top_k, body.corpus_id)
        pre_rerank = None
        if rerank_fn is not None:
            pre_rerank = sources
            sources = rerank_fn(body.query, sources, body.top_k)

        answer = generate_fn(body.query, sources) if generate_fn is not None else ""
        response: dict = {"answer": answer, "sources": sources}
        if pre_rerank is not None:
            response["trace"] = {"pre_rerank_source_refs": pre_rerank}
        return response

    return app


def run_server(app: Any, *, host: str = "0.0.0.0", port: int = 8800) -> None:
    """Starts `app` (the return value of serve()) with uvicorn — the only
    other call a quickstart needs. Requires the `serve` extra (`pip install
    causa-rag-client[serve]`); kept as a separate function (not bundled
    into serve()) so building the app and choosing how to run it stay
    decoupled — e.g. a caller using their own ASGI server never needs this.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)
