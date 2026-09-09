"""core/experiment/runner.py:_on_the_model, and its wiring into
_build_pipeline via ExperimentConfig.params["model"].

Locks in the fix for a real gap: an in_process run's generator model was
completely decorative (see the design notes "Decorative") —
registry-resolved pipelines are built exactly once at gateway startup with
whatever model was current then, and the only way to change it was
PUT /settings/model (process-wide, affecting every Realm's chat too, not
scoped to one run). An external RAG's own model was already selectable via
this same params["model"] key (read RAG-side) — this closes
the matching gap for in_process runs.

Whether a component can change its model is the component's own answer now
(`core.interfaces.ChoosingItsModel`), so this builds the real generator, which
opens no connection until it generates. It used to carry a fake whose class
name had to be rewritten to be recognised, because the choice was a case
analysis over adapter class names written in the builder.
"""
from __future__ import annotations

from adapters.ollama_generator import OllamaGenerator
from core.experiment.runner import _on_the_model


def _generator(model: str = "qwen3:8b") -> OllamaGenerator:
    return OllamaGenerator(base_url="http://fake-ollama:11434", model=model, timeout=99.0)


class _AGeneratorWithNoModelToChange:
    """A stub or a vLLM generator: it answers nothing about choosing a model,
    and comes back as it was instead of raising or looking as though the
    request had worked."""

    _model = "whatever"


def test_no_op_when_model_is_falsy() -> None:
    gen = _generator()
    assert _on_the_model(gen, None) is gen
    assert _on_the_model(gen, "") is gen


def test_no_op_when_model_already_matches() -> None:
    gen = _generator()
    assert _on_the_model(gen, "qwen3:8b") is gen


def test_a_generator_with_no_model_to_change_is_left_alone() -> None:
    gen = _AGeneratorWithNoModelToChange()
    assert _on_the_model(gen, "qwen3:32b") is gen


def test_rebinds_to_the_requested_model() -> None:
    """The copy runs the model asked for and keeps everything else it was
    built with: another address or another timeout would be a different
    server answering the question."""
    result = _on_the_model(_generator(), "qwen3:32b")
    assert result._model == "qwen3:32b"
    assert result._base_url == "http://fake-ollama:11434"
    assert result._timeout == 99.0


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


def test_no_model_override_keeps_the_model_the_gateway_started_with() -> None:
    """Absent params["model"], the model does not move.

    The instance does. A run carries a seed, the seed is applied now, and a
    seeded generator is a copy: there is no configuration without a seed,
    because the field has always defaulted to one, and treating that default
    as "unset" would put the field back where it was, recorded and read by
    nobody.
    """
    runner, startup_generator = _make_runner_with_ollama_pipeline()
    config = _make_config(params={})

    built = runner._build_pipeline(config)

    assert built._generator._model == startup_generator._model
    assert (built._generator._base_url, built._generator._timeout) == (
        startup_generator._base_url, startup_generator._timeout)
    assert built._generator._seed == config.seed
    assert startup_generator._seed is None, (
        "the generator every other caller shares was seeded under them"
    )
