"""External RAG registry (model C) — saved list of HTTP endpoints
so a user picks a known external RAG from a dropdown instead of retyping its
URL on every new run. Pure MongoDB CRUD, no file fallback (unlike prompts —
nothing in the pipeline needs synchronous access to this list).
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import adapters.mongodb as mdb

router = APIRouter(prefix="/external-rags", tags=["external-rags"])
# Published at the bare /external-rag-spec path (not
# nested under /external-rags), since it documents the contract for the
# whole registry, not a single record. Separate router, same module.
spec_router = APIRouter(tags=["external-rags"])

_COLLECTION = "external_rags"


# @lat: [[external-rag#External RAG registry]]
class ExternalRagCreateRequest(BaseModel):
    name: str
    url: str
    description: str = ""
    headers: dict[str, str] = {}
    # Optional retrieval-only endpoint, declared at
    # registration time (not guessed by probing). Drives
    # capabilities.supports_retrieval_only once test() confirms it answers.
    retrieve_endpoint: str | None = None
    # Found live registering an agentic RAG: HttpPipeline's flat 30s default
    # (see adapters/http_pipeline.py#HttpPipeline) is too short for a
    # multi-step agentic RAG (classify → retrieve → NLI-validate can
    # legitimately take 60-130s+) — every request, including the
    # registration-time test() probe, timed out even though the RAG was
    # working correctly. None/falsy ⇒ the 30s default, unchanged for every
    # RAG that doesn't set this.
    timeout_s: float | None = None
    # Tier 2, declarative mapping (config-only, no code
    # on either side). Both None ⇒ tier 1 (native contract), unchanged.
    # request_template: the RAG's own native request body shape, with
    # {{query}}/{{top_k}}/{{filters}}/{{trace_id}} placeholders.
    # response_mapping: JSONPath expressions extracting this platform's
    # canonical {"answer", "sources"} out of the RAG's own native response
    # shape (see adapters/jsonpath_mapping.py for the recognized keys).
    request_template: dict[str, Any] | None = None
    response_mapping: dict[str, str] | None = None
    # Which keys of the extensible `params` block
    # (adapters/http_pipeline.py's HttpPipeline `params=`) this RAG actually
    # reads and applies. Unlike the other capabilities fields (derived from
    # a live test() probe — see below), this one cannot be observed from a
    # single probe response: there's no way to tell from the outside
    # whether a knob silently had no effect. Declared by the registrant
    # instead — honest about the source of truth, not a guess dressed up as
    # a measurement. The UI only lets a run vary knobs listed here, so an
    # un-declared knob is never silently sent and silently ignored (the
    # same decorative-field trap chunking_strategy/embedder/generator fell
    # into for in_process runs — see the design notes).
    supported_params: list[str] = []
    # Where this system sits in the space of architectures, as
    # {dimension code: value}: `{"C3": "rrf", "D1": "cross_encoder"}` for a
    # standard hybrid. Declared for the same reason `supported_params` above
    # is: it cannot be observed from a probe, and a guess dressed as a
    # measurement is worse than an honest declaration.
    #
    # It decides which catalogue entries can occur in this system at all,
    # which is the whole of what "applicable" means. Left empty, the failure
    # atlas can say nothing about the system beyond what it says about every
    # system, and says so instead of assuming a shape.
    coordinates: dict[str, str] = {}
    # Realm membership. When set, this RAG endpoint belongs to the
    # named Realm and inherits its shared infrastructure (Qdrant/Neo4j/etc.).
    # Omit for backward compat (env-var-based connections still work).
    realm_id: str | None = None
    # per-(Realm, ExternalRag) defaults for the "New run" form.
    # Found live: the form hardcoded corpus_id to one corpus's literal and
    # never reset it when a different RAG was selected, so a real experiment
    # silently queried an unrelated RAG with the wrong corpus_id. These let the form
    # auto-fill the right values for THIS RAG instead of relying on whoever
    # is running an experiment to remember them by hand every time. Purely
    # advisory — NewExperimentPage.tsx copies them into the form on RAG
    # selection; nothing server-side enforces them.
    default_corpus_id: str | None = None
    default_pipeline_id: str | None = None
    default_reranker_id: str | None = None
    default_params: dict[str, Any] = {}
    # "Case A" inspection: declares that this RAG's own storage IS
    # one of its Realm's already-registered resources (the common local-dev
    # shape — see the design notes, found live with an external RAG
    # sharing the platform's own Qdrant). Purely a UI signal: when true, RealmResourcesPage
    # offers a shortcut into Content/Health/Graph for this RAG's
    # default_corpus_id. The read path itself (corpus.py's browse_chunks/
    # corpus_health/graph_communities) needs no new code or credentials to
    # support this — it already resolves any corpus_id against the Realm's
    # registered resources regardless of who queries it over HTTP. Deliberately
    # NOT a path to register a genuinely separate, third-party credential set
    # for a RAG's own private infrastructure — that's a different trust model,
    # out of scope here (see the design notes for the boundary).
    uses_realm_resources: bool = False


class ExternalRagUpdateRequest(BaseModel):
    """All fields optional — PATCH semantics, only provided keys are changed.
    Registration (POST) has no edit path today; this is the first one, added
    specifically so default_* can be set/corrected after the fact without
    deleting and recreating the record (which would also orphan its
    registered datasets — see services/api_gateway/routers/datasets.py)."""
    name: str | None = None
    url: str | None = None
    description: str | None = None
    headers: dict[str, str] | None = None
    retrieve_endpoint: str | None = None
    timeout_s: float | None = None
    supported_params: list[str] | None = None
    default_corpus_id: str | None = None
    default_pipeline_id: str | None = None
    default_reranker_id: str | None = None
    default_params: dict[str, Any] | None = None
    uses_realm_resources: bool | None = None


@router.get("")
async def list_external_rags(realm_id: str | None = None) -> list[dict[str, Any]]:
    query: dict[str, Any] = {}
    if realm_id:
        query["realm_id"] = realm_id
    docs = await mdb.find_many(_COLLECTION, query=query, sort=[("created_at", -1)])
    return [{k: v for k, v in d.items() if k != "_id"} for d in docs]


@spec_router.get("/external-rag-spec")
async def get_external_rag_spec() -> dict[str, Any]:
    """machine-readable contract any external RAG must
    implement to be tested by this platform. Generated from the same
    Pydantic models (`core/models.py`) that actually parse responses, so the
    published spec cannot drift from the code that enforces it.
    """
    from core.models import ExternalTrace, SourceRef, StageTrace

    return {
        "version": "1.1.0",
        # Additive, backward-compatible: `trace.embedders` (list
        # of embedder ids this RAG used to answer THIS request) is new and
        # entirely optional. causa_rag_client's check_contract_version()
        # only hard-fails on a MAJOR version difference now — a minor bump
        # like this one degrades to a warning for already-deployed SDK
        # clients, not a forced crash on next restart.
        "changelog": {
            "1.1.0": "Added optional response field trace.embedders (list[str]) — "
                     "lets the platform compare against the corpus's registered "
                     "embedder(s) and warn on mismatch instead of only finding out "
                     "as a hard vector-dimension error at query time.",
        },
        "request": {
            "method": "POST",
            "headers": {"X-Trace-Id": "uuid, required"},
            "body": {
                "query": "string, required",
                "top_k": "int, default 5",
                "filters": "object, optional",
                "trace_id": "string, optional (mirrors the header)",
            },
        },
        "response": {
            "answer": "string, required",
            "sources": "list[SourceRef], optional, at the top level (canonical)",
            "trace": "ExternalTrace, optional; sources inside it are read as a fallback",
            "metadata": "object, optional",
        },
        "retrieval_only": {
            "method": "POST {retrieve_endpoint}",
            "body": "the same request",
            "response": {"sources": "list[SourceRef]"},
        },
        # Extensible, capabilities-gated request knobs
        # (fetch_k, fusion alpha, generation temperature/prompt_version,
        # etc.) that don't warrant a dedicated top-level field each. A RAG
        # declares which of these it actually reads via
        # capabilities.supported_params at registration time; the platform
        # never assumes a key not on that list does anything.
        "params": {
            "method": "POST / (and /retrieve)",
            "body": "optional top-level 'params' object, opaque to the platform",
        },
        "schemas": {
            "SourceRef": SourceRef.model_json_schema(),
            "StageTrace": StageTrace.model_json_schema(),
            "ExternalTrace": ExternalTrace.model_json_schema(),
        },
        "capabilities_declaration": {
            "supports_trace": "bool",
            "supports_retrieval_only": "bool",
            "retrieve_endpoint": "string | null",
            "max_top_k": "int | null",
            "source_ref_granularity": "'chunk' | 'document'",
            "supported_params": "list[string] — declared at registration, not probed",
        },
    }


@router.post("", status_code=201)
def _checked_coordinates(declared: dict[str, str]) -> dict[str, str]:
    """The declaration, or a refusal naming what could not be resolved.

    Checked against the published schema the platform keeps a verified copy
    of, so a typo becomes an error at registration and never a system the
    atlas quietly says nothing about. The alternative was storing whatever
    arrived, and an unresolvable coordinate stored is a coordinate no entry
    will ever match: the system would read as one no failure can happen in.
    """
    from core.eval import rag_space

    problems: list[str] = []
    for code, value in sorted(declared.items()):
        dimension = rag_space.get(code)
        if dimension is None:
            problems.append(f"{code} is not a dimension of the schema")
        elif value not in dimension.values:
            problems.append(
                f"{code}={value} is not one of {', '.join(dimension.values)}")
    if problems:
        raise HTTPException(
            status_code=400,
            detail=("Coordinates that do not resolve in the schema "
                    f"{rag_space.SCHEMA_SOURCE}: {'; '.join(problems)}"),
        )
    return dict(declared)


async def create_external_rag(body: ExternalRagCreateRequest) -> dict[str, Any]:
    doc = {
        "id": str(uuid.uuid4())[:8],
        "name": body.name,
        "url": body.url,
        "description": body.description,
        "headers": body.headers,
        "retrieve_endpoint": body.retrieve_endpoint,
        "timeout_s": body.timeout_s,
        "request_template": body.request_template,
        "response_mapping": body.response_mapping,
        "supported_params": body.supported_params,
        "coordinates": _checked_coordinates(body.coordinates),
        # Populated by test(), not at registration time;
        # null until the first successful probe so the UI can show
        # "unchecked" rather than a fabricated default.
        "capabilities": None,
        # Optional Realm membership; None = backward compat (env vars)
        "realm_id": body.realm_id,
        # per-(Realm, ExternalRag) defaults, see
        # ExternalRagCreateRequest.default_corpus_id docstring.
        "default_corpus_id": body.default_corpus_id,
        "default_pipeline_id": body.default_pipeline_id,
        "default_reranker_id": body.default_reranker_id,
        "default_params": body.default_params,
        "uses_realm_resources": body.uses_realm_resources,
        "created_at": datetime.now(UTC).isoformat(),
    }
    await mdb.insert_one(_COLLECTION, dict(doc))
    return doc


@router.patch("/{rag_id}")
async def update_external_rag(rag_id: str, body: ExternalRagUpdateRequest) -> dict[str, Any]:
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    doc = await mdb.find_one(_COLLECTION, {"id": rag_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"External RAG {rag_id!r} not found")
    await mdb.update_one(_COLLECTION, {"id": rag_id}, {"$set": updates})
    merged = {**doc, **updates}
    return {k: v for k, v in merged.items() if k != "_id"}


@router.delete("/{rag_id}", status_code=204)
async def delete_external_rag(rag_id: str) -> None:
    deleted = await mdb.delete_one(_COLLECTION, {"id": rag_id})
    if not deleted:
        raise HTTPException(status_code=404, detail=f"External RAG {rag_id!r} not found")
    # This RAG's own registered datasets live in the unified
    # `datasets` collection now (services/api_gateway/routers/datasets.py),
    # tagged by source_rag_id rather than a separate per-RAG collection.
    # Same "one query" cascade the old dedicated collection gave for free.
    await mdb.delete_many("datasets", {"source_rag_id": rag_id})


async def _known_embedders_for_corpus(realm_id: str | None, corpus_id: str | None) -> set[str]:
    """Every non-empty embedder_id across the corpus's registered backends
    (services/api_gateway/routers/corpus.py's `corpora` registry) —
    a corpus can legitimately have more than one (e.g. a GraphRAG-type corpus
    has a separate embedder for Neo4j community embeddings, distinct from
    the Qdrant chunk embedder). Empty when the corpus isn't registered —
    there's nothing to compare a RAG's self-report against in that case."""
    if not realm_id or not corpus_id:
        return set()
    from services.api_gateway.routers.corpus import _list_corpora
    corpora = await _list_corpora(realm_id)
    entry = next((c for c in corpora if c.get("corpus_id") == corpus_id), None)
    if not entry:
        return set()
    backends = entry.get("backends") or {}
    return {
        b["embedder_id"] for b in backends.values()
        if isinstance(b, dict) and b.get("embedder_id")
    }


@router.post("/{rag_id}/test")
async def test_external_rag(rag_id: str) -> dict[str, Any]:
    """Send one real probe query so the user can verify the endpoint before
    spending a full experiment run on it. Reports whether it answered at
    all, and whether it returned the optional trace contract.
    """
    doc = await mdb.find_one(_COLLECTION, {"id": rag_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"External RAG {rag_id!r} not found")

    from adapters.http_pipeline import EgressNotAllowedError, HttpPipeline
    from core.models import QueryRequest

    try:
        pipeline = HttpPipeline(
            url=doc["url"],
            headers=doc.get("headers") or {},
            request_template=doc.get("request_template"),
            response_mapping=doc.get("response_mapping"),
            timeout=doc.get("timeout_s") or 30.0,
            # The probe used to go out with no corpus/realm context
            # at all, so a reported embedder_id (see below) couldn't be
            # compared against anything meaningful. Mirrors what a real
            # experiment run already passes (core/experiment/runner.py).
            corpus_id=doc.get("default_corpus_id"),
            realm_id=doc.get("realm_id"),
        )
        answer = pipeline.run(QueryRequest(text="ping", top_k=1))

        # Capabilities are DERIVED from this probe's real
        # response, not declared by the user (granularity especially —
        # whether sources carry a non-empty chunk_id — is observable fact,
        # not something the registrant could be trusted to self-report
        # accurately).
        sources = answer.source_refs
        if sources:
            granularity = "chunk" if any(s.chunk_id for s in sources) else "document"
        else:
            granularity = None

        # Compare what this RAG says it used against what the
        # corpus was actually indexed with (Phase A's per-backend
        # embedder_id). Only attempted when the corpus is registered at all
        # (known_embedders non-empty) — the platform never guesses at, nor
        # tries to resolve, an embedder_id it doesn't already know from its
        # own registry; an unregistered corpus just means nothing to compare.
        reported_embedders = answer.metadata.get("reported_embedders")
        known_embedders = await _known_embedders_for_corpus(doc.get("realm_id"), doc.get("default_corpus_id"))
        embedder_mismatch_warning = None
        embedder_hint = None
        if known_embedders:
            if reported_embedders:
                if not (set(reported_embedders) & known_embedders):
                    embedder_mismatch_warning = (
                        f"The RAG reports embedder(s) {sorted(reported_embedders)} while corpus "
                        f"{doc.get('default_corpus_id')!r} is registered with "
                        f"{sorted(known_embedders)}; the embedding spaces may not match."
                    )
            else:
                embedder_hint = (
                    "This RAG does not report which embedder it used (trace.embedders). "
                    "If it does dense retrieval against this corpus, adding the field is "
                    "worth it; see docs/external-rag-contract.md."
                )

        capabilities = {
            "supports_trace": answer.stage_trace is not None,
            "supports_retrieval_only": bool(doc.get("retrieve_endpoint")),
            "retrieve_endpoint": doc.get("retrieve_endpoint"),
            "max_top_k": None,
            "source_ref_granularity": granularity,
            # Declared at registration (see
            # ExternalRagCreateRequest.supported_params docstring for why
            # this one is declared, not probed), merged in here so a single
            # `capabilities` dict is still the one place the UI/client reads
            # everything this RAG supports.
            "supported_params": doc.get("supported_params") or [],
            "reported_embedders": reported_embedders,
            "embedder_mismatch_warning": embedder_mismatch_warning,
            "embedder_hint": embedder_hint,
        }
        await mdb.update_one(_COLLECTION, {"id": rag_id}, {"$set": {"capabilities": capabilities}})

        return {
            "ok": True,
            "answer_preview": answer.text[:200],
            "has_trace": answer.stage_trace is not None,
            "n_sources": len(answer.source_refs),
            "capabilities": capabilities,
            # Risk mitigation: when tier 2 (mapping) is
            # configured, show what the platform actually parsed out of the
            # RAG's raw response, so a bad JSONPath is visibly a mapping bug
            # (empty/wrong sources HERE) and not silently mistaken for poor
            # retrieval quality once real metric runs start.
            "connection_tier": "mapped" if doc.get("response_mapping") else "native",
            "parsed_sources_preview": [s.model_dump() for s in answer.source_refs[:3]],
        }
    except EgressNotAllowedError as exc:
        return {"ok": False, "error": f"Blocked by the allowlist: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# This RAG's own dataset registration/listing/deletion moved to
# the unified services/api_gateway/routers/datasets.py (POST/GET /datasets,
# DELETE /datasets/by-id/{id}, all with an optional source_rag_id=<this
# rag's id> instead of a dedicated per-RAG sub-resource) — see that module's
# docstring for why.
