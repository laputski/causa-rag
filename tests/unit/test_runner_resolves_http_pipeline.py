"""ExperimentRunner resolves HttpPipeline when pipeline_source=http."""
from __future__ import annotations

import pytest

from adapters.http_pipeline import HttpPipeline
from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentRunner
from core.registry import ComponentRegistry


def _base() -> dict:
    return dict(
        name="t",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
    )


def test_build_pipeline_returns_http_pipeline_when_configured():
    runner = ExperimentRunner(registry=ComponentRegistry())
    config = ExperimentConfig(
        **_base(), pipeline_source="http", http_endpoint="https://external.example/query",
    )

    pipeline = runner._build_pipeline(config)

    assert isinstance(pipeline, HttpPipeline)
    assert pipeline._url == "https://external.example/query"


def test_build_pipeline_passes_pipeline_id_and_reranker_to_http_pipeline():
    """pipeline_id/reranker travel through to
    HttpPipeline (mirroring corpus_id), so a dog-fooding external RAG like
    reference_rag_server actually receives them instead of only top_k."""
    runner = ExperimentRunner(registry=ComponentRegistry())
    config = ExperimentConfig(
        **_base(), pipeline_source="http", http_endpoint="https://external.example/query",
        pipeline_id="graph", reranker=ComponentRef(kind="reranker", component_id="cross_encoder_local"),
    )

    pipeline = runner._build_pipeline(config)

    assert pipeline._external_pipeline_id == "graph"
    assert pipeline._reranker_id == "cross_encoder_local"


def test_build_pipeline_bypasses_registry_for_http_source():
    # An empty registry would raise KeyError on resolve("pipeline", ...) —
    # proves the http branch never touches the registry at all.
    runner = ExperimentRunner(registry=ComponentRegistry())
    config = ExperimentConfig(
        **_base(), pipeline_source="http", http_endpoint="https://external.example/query",
    )
    runner._build_pipeline(config)  # must not raise


def test_build_pipeline_requires_http_endpoint():
    runner = ExperimentRunner(registry=ComponentRegistry())
    config = ExperimentConfig(**_base(), pipeline_source="http", http_endpoint=None)

    with pytest.raises(ValueError):
        runner._build_pipeline(config)


def test_in_process_default_still_resolves_from_registry():
    """In-process path still goes through the registry, not literally
    returning the registered instance unchanged — _build_pipeline now
    always rebinds corpus_id/top_k (see core/experiment/runner.py
    _rebind_corpus_id), so a fresh same-type pipeline is built, reusing the
    registered retriever as-is when corpus_id already matches "default"
    (a bare object() sentinel can't stand in for that anymore — it has no
    _retriever/_embedder/_generator/pipeline_id for the rebuild to read)."""
    from adapters.qdrant import QdrantRetrieverStub
    from core.pipeline import NaivePipeline

    registry = ComponentRegistry()
    registered_retriever = QdrantRetrieverStub()
    registered = NaivePipeline(
        retriever=registered_retriever, embedder=object(), generator=object(), pipeline_id="naive",
    )
    registry.register("pipeline", "naive", registered)
    runner = ExperimentRunner(registry=registry)
    config = ExperimentConfig(**_base())  # pipeline_source defaults to in_process, corpus_id="default"

    pipeline = runner._build_pipeline(config)

    assert isinstance(pipeline, NaivePipeline)
    assert pipeline.pipeline_id == "naive"
    # corpus_id="default" on both sides — _rebind_corpus_id is a no-op,
    # same retriever instance carried through (QdrantRetrieverStub has no
    # _corpus_id attribute at all, so the no-op path is what's exercised).
    assert pipeline._retriever is registered_retriever
