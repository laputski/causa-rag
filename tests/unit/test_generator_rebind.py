"""core/experiment/runner.py:_rebind_generator, and its wiring into
_build_pipeline via ExperimentConfig.params["model"].

Locks in the fix for a real gap: an in_process run's generator model was
completely decorative (see the design notes "Decorative") —
registry-resolved pipelines are built exactly once at gateway startup with
whatever model was current then, and the only way to change it was
PUT /settings/model (process-wide, affecting every Realm's chat too, not
scoped to one run). An external RAG's own model was already selectable via
this same params["model"] key (read RAG-side) — this closes
the matching gap for in_process runs.
"""
from __future__ import annotations

from unittest.mock import patch

from core.experiment.runner import _rebind_generator


class _FakeOllamaGenerator:
    """Duck-types adapters.ollama_generator.OllamaGenerator's relevant
    attributes without a real Ollama connection — class name matters, not
    identity, since _rebind_generator dispatches on type(...).__name__."""

    def __init__(self, model: str) -> None:
        self._base_url = "http://fake-ollama:11434"
        self._model = model
        self._timeout = 99.0


_FakeOllamaGenerator.__name__ = "OllamaGenerator"


class _FakeStubGenerator:
    def __init__(self) -> None:
        self._model = "whatever"


def test_no_op_when_model_is_falsy() -> None:
    gen = _FakeOllamaGenerator(model="qwen3:8b")
    assert _rebind_generator(gen, None) is gen
    assert _rebind_generator(gen, "") is gen


def test_no_op_when_model_already_matches() -> None:
    gen = _FakeOllamaGenerator(model="qwen3:8b")
    assert _rebind_generator(gen, "qwen3:8b") is gen


def test_no_op_for_non_ollama_generator_type() -> None:
    """A stub/vllm generator has no equivalent constructor shape — pass
    through unchanged rather than raising or silently ignoring the request
    in a way that looks like it worked."""
    gen = _FakeStubGenerator()
    assert _rebind_generator(gen, "qwen3:32b") is gen


def test_rebinds_to_the_requested_model() -> None:
    gen = _FakeOllamaGenerator(model="qwen3:8b")
    with patch("adapters.ollama_generator.OllamaGenerator") as mock_cls:
        sentinel = object()
        mock_cls.return_value = sentinel
        result = _rebind_generator(gen, "qwen3:32b")
    assert result is sentinel
    mock_cls.assert_called_once_with(base_url="http://fake-ollama:11434", model="qwen3:32b", timeout=99.0)


# ── Integration: ExperimentRunner._build_pipeline actually applies the override ──

def _make_config(params: dict[str, object] | None = None):
    from core.experiment.config import ComponentRef, ExperimentConfig
    return ExperimentConfig(
        name="t",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
        pipeline_id="naive",
        params=params or {},
    )


def _make_runner_with_ollama_pipeline():
    from adapters.bge_m3 import BgeM3Embedder
    from adapters.ollama_generator import OllamaGenerator
    from adapters.qdrant import QdrantRetrieverStub
    from core.experiment.runner import ExperimentRunner
    from core.pipeline import NaivePipeline
    from core.registry import ComponentRegistry

    reg = ComponentRegistry()
    embedder = BgeM3Embedder()
    startup_generator = OllamaGenerator(model="qwen3:8b")
    pipeline = NaivePipeline(retriever=QdrantRetrieverStub(), embedder=embedder, generator=startup_generator)
    reg.register("embedder", "bge_m3", embedder)
    reg.register("pipeline", "naive", pipeline)
    runner = ExperimentRunner(registry=reg)
    return runner, startup_generator


def test_build_pipeline_uses_override_model_when_params_model_set() -> None:
    runner, startup_generator = _make_runner_with_ollama_pipeline()
    config = _make_config(params={"model": "qwen3:32b"})

    built = runner._build_pipeline(config)

    assert built._generator._model == "qwen3:32b"
    assert built._generator is not startup_generator


def test_build_pipeline_reuses_shared_generator_when_no_model_override() -> None:
    """Additive only: absent params["model"] must not change existing
    behavior (the shared startup-time generator instance, unchanged)."""
    runner, startup_generator = _make_runner_with_ollama_pipeline()
    config = _make_config(params={})

    built = runner._build_pipeline(config)

    assert built._generator is startup_generator
