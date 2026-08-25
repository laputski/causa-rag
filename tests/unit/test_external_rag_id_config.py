"""external_rag_id config field + backward-compatible hash.

external_rag_id is a comparison dimension (which external system was tested),
excluded from config_hash when None so historical runs stay comparable; setting
it changes the hash (a different external RAG is a different experiment).
"""
from __future__ import annotations

from core.experiment.config import ComponentRef, ExperimentConfig


def _base() -> dict:
    return dict(
        name="t",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
    )


def test_unset_external_rag_id_keeps_prior_hash():
    """A config that never mentions external_rag_id must hash identically to one
    that sets it explicitly to None — so older runs stay comparable."""
    old = ExperimentConfig(**_base())
    explicit_none = ExperimentConfig(**_base(), external_rag_id=None)
    assert old.config_hash == explicit_none.config_hash


def test_setting_external_rag_id_changes_hash():
    old = ExperimentConfig(**_base())
    external = ExperimentConfig(**_base(), pipeline_source="http", external_rag_id="abc12345")
    assert external.config_hash != old.config_hash


def test_different_external_rag_id_differs():
    a = ExperimentConfig(**_base(), pipeline_source="http", external_rag_id="rag_a")
    b = ExperimentConfig(**_base(), pipeline_source="http", external_rag_id="rag_b")
    assert a.config_hash != b.config_hash


def test_external_rag_name_unset_keeps_prior_hash():
    """external_rag_name is a display-only backfill (see field comment) —
    a config that never sets it must hash identically to one that sets it
    explicitly to None, same back-compat convention as external_rag_id."""
    old = ExperimentConfig(**_base())
    explicit_none = ExperimentConfig(**_base(), external_rag_name=None)
    assert old.config_hash == explicit_none.config_hash


def test_backfilling_external_rag_name_and_http_endpoint_after_construction_does_not_change_hash():
    """Found live: a run made via external_rag_id left http_endpoint null in
    the stored config — no way to tell which external service a failure was
    even against without cross-referencing external_rag_id against
    GET /external-rags by hand. create_experiment now backfills both
    fields from the resolved ExternalRag record, but only AFTER
    config_hash was already computed (the @model_validator runs once, at
    construction) — this is what keeps a RAG's URL rotating over time from
    silently changing the hash identity of "the same logical config against
    the same registered RAG"."""
    cfg = ExperimentConfig(**_base(), pipeline_source="http", external_rag_id="abc123")
    hash_before = cfg.config_hash

    cfg.external_rag_name = "agentic-rag"
    cfg.http_endpoint = "http://localhost:8003/platform/query"

    assert cfg.config_hash == hash_before


def test_runner_resolves_external_rag_id_to_http_pipeline():
    """The runner's _build_pipeline must call the injected resolver for a
    config with external_rag_id and build an HttpPipeline from its url/headers
    — not require an inline http_endpoint."""
    from adapters.http_pipeline import HttpPipeline
    from core.experiment.runner import ExperimentRunner
    from core.registry import ComponentRegistry

    captured = {}

    def resolver(rag_id: str) -> dict:
        captured["rag_id"] = rag_id
        return {"url": "http://allowed.example/rag", "headers": {"X-Key": "secret"}}

    runner = ExperimentRunner(
        registry=ComponentRegistry(), external_rag_resolver=resolver,
    )
    cfg = ExperimentConfig(**_base(), pipeline_source="http", external_rag_id="abc123")
    pipeline = runner._build_pipeline(cfg)

    assert isinstance(pipeline, HttpPipeline)
    assert captured["rag_id"] == "abc123"
    assert pipeline._url == "http://allowed.example/rag"


def test_runner_uses_registered_timeout_s_override():
    """Found live: HttpPipeline's flat 30s default is too short for a
    multi-step agentic RAG. A registered timeout_s must override it
    per-RAG."""
    from adapters.http_pipeline import HttpPipeline
    from core.experiment.runner import ExperimentRunner
    from core.registry import ComponentRegistry

    def resolver(rag_id: str) -> dict:
        return {"url": "http://slow.example/rag", "timeout_s": 180.0}

    runner = ExperimentRunner(
        registry=ComponentRegistry(), external_rag_resolver=resolver,
    )
    cfg = ExperimentConfig(**_base(), pipeline_source="http", external_rag_id="abc123")
    pipeline = runner._build_pipeline(cfg)

    assert isinstance(pipeline, HttpPipeline)
    assert pipeline._timeout == 180.0


def test_runner_defaults_to_30s_when_timeout_s_unset():
    from adapters.http_pipeline import HttpPipeline
    from core.experiment.runner import ExperimentRunner
    from core.registry import ComponentRegistry

    def resolver(rag_id: str) -> dict:
        return {"url": "http://fast.example/rag"}

    runner = ExperimentRunner(
        registry=ComponentRegistry(), external_rag_resolver=resolver,
    )
    cfg = ExperimentConfig(**_base(), pipeline_source="http", external_rag_id="abc123")
    pipeline = runner._build_pipeline(cfg)

    assert isinstance(pipeline, HttpPipeline)
    assert pipeline._timeout == 30.0


def test_runner_http_without_endpoint_or_resolver_raises():
    from core.experiment.runner import ExperimentRunner
    from core.registry import ComponentRegistry

    runner = ExperimentRunner(registry=ComponentRegistry())
    cfg = ExperimentConfig(**_base(), pipeline_source="http")
    import pytest
    with pytest.raises(ValueError, match="http_endpoint"):
        runner._build_pipeline(cfg)
