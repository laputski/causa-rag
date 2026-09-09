"""The seed a run carries reaches the only part of it that samples.

`ExperimentConfig.seed` has been on a configuration since the first version of
it, and the run page says of it, in both languages, that the same configuration
and the same seed give identical answers on a repeat run. Nothing read the
field. The sentence was a promise the platform did not keep, which is the class
of defect this whole platform exists to find in other people's systems.

Measured before the fix, against the model this platform runs and at the
temperature it runs it. On one question, ten identical requests gave two
different answers and ten with a seed gave one; on another, eight of eight
agreed with a seed and without. So the divergence is occasional and depends on
the question, which is exactly what the comparison machinery already assumes
when it resamples a flipped question before trusting it. The two answers that
differed differed by a single word, which is small and is enough to move a
similarity score across a threshold.

What the seed can fix is bounded and the bound is worth stating: the generator
is the only part of an in-process run that samples. Retrieval, reranking and
embedding are deterministic, and a system answering over HTTP samples on its
own side where a seed of ours does not reach.
"""
from __future__ import annotations

from typing import Any

import pytest

from adapters.ollama_generator import OllamaGenerator
from core.experiment.runner import _sampling_fixed_by


class _Answered:
    """Ollama's answer, and a note of what it was asked."""

    def __init__(self, sent: dict[str, Any]) -> None:
        self.sent = sent

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {"response": "an answer"}


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """What the last request carried."""
    captured: dict[str, Any] = {}

    def _post(url: str, json: dict[str, Any], timeout: float) -> _Answered:
        captured.clear()
        captured.update(json)
        return _Answered(json)

    import adapters.ollama_generator as adapter
    monkeypatch.setattr(adapter.httpx, "post", _post)
    return captured


def test_a_generator_with_a_seed_asks_for_that_seed(sent: dict[str, Any]) -> None:
    """The half everything else rests on. A seed the platform records and does
    not send is the field it was before, one layer further in."""
    OllamaGenerator(model="m", seed=42).generate("a question")
    assert sent["options"]["seed"] == 42


def test_a_generator_without_one_asks_for_nothing(sent: dict[str, Any]) -> None:
    """Absent, the server samples as it would have, which is where every
    caller that never had a seed stays. The chat path is one of them: a
    conversation is not a measurement and nobody repeats one expecting the
    same words."""
    OllamaGenerator(model="m").generate("a question")
    assert "seed" not in sent["options"]


def test_the_seed_travels_with_a_copy_on_another_model(sent: dict[str, Any]) -> None:
    """The two ways a generator varies compose. Choosing a model used to build
    a fresh generator, and a fresh one had no seed."""
    OllamaGenerator(model="m", seed=42).with_model("another").generate("a question")
    assert sent["model"] == "another"
    assert sent["options"]["seed"] == 42


def test_the_model_travels_with_a_seeded_copy(sent: dict[str, Any]) -> None:
    OllamaGenerator(model="m").with_seed(7).generate("a question")
    assert sent["model"] == "m"
    assert sent["options"]["seed"] == 7


# ── what the build does with it ──────────────────────────────────────────────

class _AGeneratorThatCannotBeSeeded:
    """A stub, or a system answering over HTTP. It samples where a seed of
    ours does not reach, and comes back as it was."""


def test_a_component_that_cannot_be_seeded_is_left_alone() -> None:
    generator = _AGeneratorThatCannotBeSeeded()
    assert _sampling_fixed_by(generator, 42) is generator


def test_no_seed_asked_for_leaves_the_component_as_it_is() -> None:
    generator = OllamaGenerator(model="m")
    assert _sampling_fixed_by(generator, None) is generator


def test_the_build_hands_the_generator_the_seed_the_run_carries() -> None:
    """The rebind existing is not the same as the builder calling it, and the
    two have been a separate change every time."""
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentRunner
    from core.pipeline import NaivePipeline
    from core.registry import ComponentRegistry

    class _Leaf:
        retriever_id = "stub"
        embedder_id = "bge_m3"

        def retrieve(self, **kwargs: Any) -> list:
            return []

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1]] * len(texts)

    generator = OllamaGenerator(model="m")
    registry = ComponentRegistry()
    registry.register("generator", "ollama", generator)
    registry.register("embedder", "bge_m3", _Leaf())
    registry.register("pipeline", "naive", NaivePipeline(
        retriever=_Leaf(), embedder=_Leaf(), generator=generator))

    built = ExperimentRunner(registry)._build_pipeline(ExperimentConfig(
        name="c", seed=1234,
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
    ))
    assert built._generator._seed == 1234
    assert generator._seed is None, "the registered generator was changed under everyone else"
