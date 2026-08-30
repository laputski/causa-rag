"""ExperimentConfig — full serialisable description of a RAG pipeline run.

A stable config_hash uniquely identifies a configuration, enabling:
- reproducible reruns
- cache boundary decisions
- run deduplication
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ComponentRef(BaseModel):
    """Reference to a registered component by kind + id."""
    kind: str
    component_id: str
    params: dict[str, Any] = Field(default_factory=dict)


# Fields introduced after the initial config schema. They are excluded from the
# hash when left at their default, so configs created before they existed keep their
# original config_hash and stay comparable with historical runs.
_BACKCOMPAT_DEFAULTS: dict[str, Any] = {
    "grounding": None,
    "route_policy": None,
    "config_schema_version": "1.0",
    "corpus_id": "default",
    "pipeline_source": "in_process",
    "http_endpoint": None,
    "external_rag_id": None,
    "external_rag_name": None,  # display-only backfill, see field comment above
    "scorer": None,
    "mask_engine": None,
    "refusal_policy": None,
    "retrieval_only": False,
    "params": {},
    "retrieval_pins_enabled": False,
    "fetch_k": None,
}


class ExperimentConfig(BaseModel):
    """Complete, reproducible description of one RAG pipeline experiment.

    All fields that affect retrieval/generation/eval quality are captured here.
    Fields are sorted before hashing to ensure stability across Python versions.
    """
    name: str
    version: str = "1.0.0"
    seed: int = 42

    # Pipeline components (by registry reference)
    chunking_strategy: ComponentRef
    embedder: ComponentRef
    retrievers: list[ComponentRef] = Field(default_factory=list)
    reranker: ComponentRef | None = None
    generator: ComponentRef
    pipeline_id: str = "naive"

    # Configurable pipeline steps (optional; absent ⇒ the plain pipeline)
    grounding: ComponentRef | None = None
    route_policy: ComponentRef | None = None

    # domain-pack-provided steps (optional; absent ⇒ no domain pack)
    scorer: ComponentRef | None = None
    mask_engine: ComponentRef | None = None
    refusal_policy: ComponentRef | None = None

    # Retrieval params
    top_k: int = 5
    merge_strategy: str = "rrf"
    merge_alpha: float = 0.5

    # Eval
    dataset_name: str = ""
    dataset_version: str = ""

    # Which corpus namespace this run targets (multi-corpus)
    corpus_id: str = "default"

    # Where the pipeline actually runs. "in_process" keeps
    # resolving pipeline_id from the registry as before; "http" routes to
    # http_endpoint via adapters.http_pipeline.HttpPipeline instead.
    pipeline_source: str = "in_process"
    http_endpoint: str | None = None

    # Which registered external RAG this run targets.
    # When set, runner resolves url/headers from the ExternalRag record instead
    # of requiring http_endpoint inline. external_rag_id is a comparison
    # dimension (which external system was tested), NOT a security/tenant
    # boundary. Excluded from config_hash when None (back-compat).
    external_rag_id: str | None = None

    # Found live: a run made via external_rag_id left http_endpoint null (the
    # URL is resolved fresh from the ExternalRag record at run time, see
    # services/api_gateway/routers/experiments.py#create_experiment — never
    # inline in the submitted config), so the run detail page could only ever
    # show "External (—)", with no way to tell which service a "Connection
    # refused" failure was even against without manually cross-referencing
    # external_rag_id against GET /external-rags. create_experiment now
    # backfills both this and http_endpoint with the resolved name/URL AFTER
    # config_hash is computed (the @model_validator above runs once, at
    # construction) — purely informational, so a RAG's URL rotating over
    # time doesn't change the hash identity of "the same logical config
    # against the same registered RAG".
    external_rag_name: str | None = None

    # When true, the runner calls the retrieval-only path: an external RAG's
    # optional retrieve_endpoint, or NaivePipeline.retrieve() for an
    # in_process run. Either way it scores retrieval metrics without paying
    # for LLM generation on every measurement. The switch is a hasattr in
    # core/experiment/runner.py, so a pipeline type that grows no such
    # method keeps generating and says nothing about it.
    retrieval_only: bool = False

    # Opt-in for the
    # retrieval-pin overlay, OFF by default. A pin is a per-question override
    # of retrieval; the objective review of the mechanism found it fails all
    # four questions this project asks of anything that
    # affects an answer at query time, so it is no longer applied unless a
    # run explicitly asks for it as a what-if on the stand.
    #
    # Off means genuinely off, not "loaded and then unused": the services
    # layer skips the pin lookup entirely, so no store round-trip and no O(N)
    # vector scan is paid (measured 72 ms/query at 1000 pins). Excluded from
    # config_hash while False, so every historical run keeps its hash and
    # stays comparable — see _BACKCOMPAT_DEFAULTS above.
    retrieval_pins_enabled: bool = False

    # How many candidates retrieval fetches, as opposed to
    # how many reach the answer. None means "as many as reach the answer",
    # which is what every run did before and keeps historical config hashes
    # unchanged (see _BACKCOMPAT_DEFAULTS above).
    #
    # Setting it wider is what makes a counterfactual about context size
    # answerable at all: without a window past top_k there is no record of
    # where the expected source actually sat, so "what if top_k were larger"
    # can only be answered by running the whole experiment again.
    fetch_k: int | None = None

    # Extensible per-request knobs for an external RAG
    # (e.g. fetch_k, fusion alpha, generation temperature/prompt_version)
    # that don't warrant a dedicated ExperimentConfig field each. A RAG
    # declares which of these it actually applies via its registration's
    # `capabilities.supported_params` (services/api_gateway/routers/
    # external_rags.py); the UI only lets a user vary knobs the RAG
    # declared, rather than silently sending a knob no RAG-side code reads
    # (the exact decorative-field trap chunking_strategy/embedder/generator
    # already fell into — see the design notes "Decorative").
    # No-op for in_process pipelines (nothing reads it there).
    params: dict[str, Any] = Field(default_factory=dict)

    # Schema version — bumped when the config shape changes (not hashed when default)
    config_schema_version: str = "1.0"

    # Computed — set after model validation
    config_hash: str = ""

    @model_validator(mode="after")
    def _compute_hash(self) -> ExperimentConfig:
        self.config_hash = self._compute()
        return self

    def _compute(self) -> str:
        """Canonical JSON → SHA256. Excludes config_hash + unset back-compat fields."""
        payload = self.model_dump(exclude={"config_hash"})
        for key, default in _BACKCOMPAT_DEFAULTS.items():
            if payload.get(key) == default:
                payload.pop(key, None)
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def diff(self, other: ExperimentConfig) -> dict[str, dict[str, Any]]:
        """Return fields that differ between self and other."""
        a = self.model_dump(exclude={"config_hash"})
        b = other.model_dump(exclude={"config_hash"})
        return {
            key: {"before": a[key], "after": b[key]}
            for key in a
            if a[key] != b[key]
        }
