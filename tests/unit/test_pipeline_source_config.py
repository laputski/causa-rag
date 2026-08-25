"""ExperimentConfig.pipeline_source/http_endpoint."""
from __future__ import annotations

from core.experiment.config import ComponentRef, ExperimentConfig


def _base() -> dict:
    return dict(
        name="t",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
    )


def test_default_pipeline_source_keeps_prior_hash():
    old = ExperimentConfig(**_base())
    explicit = ExperimentConfig(**_base(), pipeline_source="in_process", http_endpoint=None)
    assert old.config_hash == explicit.config_hash


def test_http_pipeline_source_changes_hash():
    old = ExperimentConfig(**_base())
    http = ExperimentConfig(**_base(), pipeline_source="http", http_endpoint="https://external.example/query")
    assert old.config_hash != http.config_hash


def test_different_http_endpoints_change_hash():
    a = ExperimentConfig(**_base(), pipeline_source="http", http_endpoint="https://a.example/query")
    b = ExperimentConfig(**_base(), pipeline_source="http", http_endpoint="https://b.example/query")
    assert a.config_hash != b.config_hash
