from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResourceConfig(BaseModel):
    """Typed connection config for a backend resource in a Realm.

    Uses extra='allow' so each connector type can carry its own params
    (host/port for Qdrant, uri/user/password for Neo4j, etc.) without
    a union of subtypes. Validated against connector_types catalog at write time.
    """

    model_config = {"extra": "allow"}

    type: str


class DocumentNode(BaseModel):
    """Generic structural node (section, article, paragraph, …)."""

    node_id: str
    node_type: str
    title: str | None = None
    content: str = ""
    level: int = 0
    children: list[DocumentNode] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


DocumentNode.model_rebuild()


class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    content: str
    content_hash: str = ""
    structure: DocumentNode | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _chunk_id(source: str, start: int, end: int, strategy: str) -> str:
    """Deterministic chunk ID — same content always gets the same ID."""
    key = f"{source}:{strategy}:{start}:{end}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    structural_path: str = ""
    start_char: int = 0
    end_char: int = 0
    strategy_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever_id: str = ""


class StageTrace(BaseModel):
    embed_ms: float = 0.0
    dense_retrieve_ms: float = 0.0
    sparse_retrieve_ms: float = 0.0
    merge_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0
    n_dense: int = 0
    n_sparse: int = 0
    n_merged: int = 0
    n_deduped: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_chars: int = 0
    # Configurable pipeline steps (0 when the step is disabled)
    rerank_ms: float = 0.0
    grounding_ms: float = 0.0
    graph_ms: float = 0.0
    n_reranked: int = 0
    n_unsupported: int = 0
    n_graph: int = 0


class SourceRef(BaseModel):
    doc_id: str
    # Optional: an external RAG may identify a source only
    # at document granularity. doc_id is the one identifier retrieval metrics
    # actually need (matched via source_code/article_no); chunk_id defaults to
    # "" rather than forcing the external system to invent a stub value.
    chunk_id: str = ""
    structural_path: str = ""
    score: float = 0.0
    chunk_text: str = ""
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_rank: int = 0
    # Set when this chunk was injected by a matching
    # retrieval_pins entry (core/pins/overlay.py) rather than found by
    # ordinary retrieval, so the run's own trace stays auditable: a
    # reviewer can tell which sources in the answer came from a pin, not
    # just that the answer changed. False for every chunk that came from
    # ordinary retrieval — the overlay is the only thing that ever sets it.
    pinned: bool = False
    # Eval Measurement Trustworthiness, Phase 0 — corpus-relative identity
    # ("{source_code}/{article_no}", e.g. "SRC001/5"), distinct from
    # doc_id (an internal UUID). Lets retrieval_metrics.py match against
    # golden article_refs without article-number collisions across corpora
    # that share numbers (e.g. two source documents both having an article 5).
    # None on chunks ingested before this field existed (graceful default).
    source_code: str | None = None
    article_no: str | None = None


class GroundingResult(BaseModel):
    is_grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class ExternalTrace(BaseModel):
    """Optional trace contract an external RAG (model C) may return.

    Every field is optional — an external system that returns none of them
    still gets outcome-only diagnostics (eval + regression-guard on
    text/reference comparison); one that returns all of them gets the same
    per-stage pipeline-trace as an in-process pipeline.
    """
    stage_trace: StageTrace | None = None
    sources: list[SourceRef] | None = None
    rendered_prompt: str | None = None
    # The ranked list BEFORE the external RAG's reranker cut
    # it to top-k. Without it, core/eval/funnel.py cannot tell "retrieval never
    # found the chunk" (retrieval/fusion failure) from "the reranker dropped it"
    # (rerank failure) for an external RAG — both read as zero recall. Optional:
    # a RAG with no reranker (or that doesn't expose it) omits it, and funnel
    # degrades to coarse attribution rather than claiming a layer it can't see.
    pre_rerank_source_refs: list[SourceRef] | None = None
    # Which embedder(s) this RAG used to answer THIS request, so
    # the platform can compare against the corpus's own registered
    # embedder(s) (services/api_gateway/routers/corpus.py's `corpora`
    # registry) and warn on mismatch instead of only discovering it as a hard
    # Qdrant "Vector dimension error" once retrieval actually reaches a
    # foreign-embedder collection. A list, not a single string — a RAG can
    # legitimately use more than one embedder for one corpus (e.g. dense
    # chunk retrieval + a separate graph-community embedder). Absent ⇒ no
    # comparison is attempted, same honest-degradation posture as every
    # other optional trace field.
    embedders: list[str] | None = None


class Answer(BaseModel):
    answer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_refs: list[SourceRef] = Field(default_factory=list)
    grounding: GroundingResult | None = None
    refused: bool = False
    refusal_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    stage_trace: StageTrace | None = None
    rendered_prompt_preview: str = ""
    # Funnel diagnosis, Phase 1 — the ranked list BEFORE the reranker cut it
    # to top-k, so retrieval_recall_at_k can be computed both before and
    # after reranking. Empty when no reranker ran (nothing to distinguish —
    # source_refs IS the pre-rerank list in that case). Without this,
    # "the reranker dropped the right chunk" and "retrieval never found it"
    # are indistinguishable from stored data alone.
    pre_rerank_source_refs: list[SourceRef] = Field(default_factory=list)
    # The candidate window before the final cut, present only
    # when fetch_k widened it past top_k. Distinct from the field above:
    # that one means "what the reranker was given" and core/eval/funnel.py
    # reads it as evidence about the reranker, while this one exists to
    # answer "what would a different context size have shown" without
    # re-running anything (core/eval/counterfactual.py).
    candidate_source_refs: list[SourceRef] = Field(default_factory=list)
    # Computed from source_refs metadata (core/citation.py), not generated
    # by the model — the model's own in-text citation numbers are
    # unreliable (transcription confusion, hallucinated numbers absent from
    # any retrieved chunk). Empty when there's nothing to cite (refusal, no
    # retrieved chunks).
    computed_citations: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    mode: str = ""
    question_type: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditEventType(StrEnum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    QUERY = "query"
    ANSWER = "answer"
    ERROR = "error"


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)
