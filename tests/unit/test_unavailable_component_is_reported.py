"""A component the configuration named and the registry could not produce.

The pipeline builder degrades to running without it, deliberately: an
uninstalled reranker extra should not fail a whole run. What it did not do was
say so. A run whose configuration named a reranker and that reranked nothing
was indistinguishable, on every screen and in every stored document, from one
that reranked, because what gets stored is the configuration that was asked for.

Observed directly: resolving an unregistered reranker returned nothing, and the
configuration the run filed still carried the reference.

The distinction decides whether a proving-ground pair means anything. The pair
staging a reranker that does not cover the corpus language compares two
rerankers, and proves nothing at all on a machine where neither one ran.
"""
from __future__ import annotations

from typing import Any

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult, ExperimentRunner
from core.registry import ComponentRegistry


class _Pipeline:
    pipeline_id = "naive"

    def __init__(self, **kwargs: Any) -> None:
        self._retriever = _Leaf()
        self._embedder = _Leaf()
        self._generator = _Leaf()
        self.__dict__.update(kwargs)


class _Leaf:
    retriever_id = "stub"

    def retrieve(self, **kwargs: Any) -> list:
        return []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1]] * len(texts)


def _registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register("pipeline", "naive", _Pipeline())
    # The embedder the configuration below names. It was left out while the
    # field was decorative and nothing resolved it; now that the build takes
    # the embedder a run asked for, a registry without one is a run that could
    # not produce a component it named, which is what this file is about.
    registry.register("embedder", "bge", _Leaf())
    return registry


def _config(**fields: Any) -> ExperimentConfig:
    return ExperimentConfig(**{
        "name": "c",
        "chunking_strategy": ComponentRef(kind="chunker", component_id="fixed"),
        "embedder": ComponentRef(kind="embedder", component_id="bge"),
        "generator": ComponentRef(kind="generator", component_id="ollama"),
        **fields,
    })


def test_a_component_that_could_not_be_produced_is_named() -> None:
    collected: list[str] = []
    ExperimentRunner(_registry())._build_pipeline(
        _config(reranker=ComponentRef(kind="reranker", component_id="cross_encoder_local")),
        unavailable=collected,
    )
    assert collected == ["reranker:cross_encoder_local"]


def test_a_run_with_everything_available_names_nothing() -> None:
    """The other half. A list that filled up on a healthy run would be read as
    noise within a week and then ignored on the run that mattered."""
    registry = _registry()
    registry.register("reranker", "cross_encoder_local", _Leaf())
    collected: list[str] = []
    ExperimentRunner(registry)._build_pipeline(
        _config(reranker=ComponentRef(kind="reranker", component_id="cross_encoder_local")),
        unavailable=collected,
    )
    assert collected == []


def test_a_configuration_naming_nothing_optional_names_nothing() -> None:
    collected: list[str] = []
    ExperimentRunner(_registry())._build_pipeline(_config(), unavailable=collected)
    assert collected == []


def test_the_run_still_completes_without_the_component() -> None:
    """Degrading is the deliberate part and must not become a failure: on a
    machine without the reranker extra, a whole grid would stop."""
    pipeline = ExperimentRunner(_registry())._build_pipeline(
        _config(reranker=ComponentRef(kind="reranker", component_id="cross_encoder_local")),
    )
    assert pipeline is not None


def test_the_stored_run_carries_the_list() -> None:
    """It has to survive into the document, since that is where anybody reading
    a run later will look."""
    result = ExperimentResult(config=_config(), unavailable_components=["reranker:x"])
    assert result.to_dict()["unavailable_components"] == ["reranker:x"]


def test_a_run_stored_before_the_field_existed_says_nothing_rather_than_everything() -> None:
    """An absent list means the run predates the field, and never that every
    component was available. The same reading `coverage_check` already has."""
    assert ExperimentResult(config=_config()).to_dict()["unavailable_components"] == []


def test_an_embedder_the_registry_has_not_got_is_named_like_any_other() -> None:
    """The field the plan called decorative, now that it is not.

    Both branches of the build used to take the pipeline's own embedder, so a
    run naming another model queried with whatever the gateway had registered
    and said nothing about the difference. It says it now, in the same list
    every other component uses, and the run still happens: an uninstalled
    component should not lose a measurement, it should be named in it.
    """
    registry = ComponentRegistry()
    registry.register("pipeline", "naive", _Pipeline())
    collected: list[str] = []
    pipeline = ExperimentRunner(registry)._build_pipeline(
        _config(embedder=ComponentRef(kind="embedder", component_id="a_model_nobody_installed")),
        unavailable=collected,
    )
    assert collected == ["embedder:a_model_nobody_installed"]
    assert pipeline is not None, "the run was lost instead of being told what it ran without"


def test_the_embedder_a_run_names_is_the_one_it_queries_with() -> None:
    """The half that makes the naming above worth anything: a registered
    embedder is actually used, and not merely resolved and dropped."""
    registry = _registry()
    asked_for = _Leaf()
    registry.register("embedder", "another_model", asked_for)
    pipeline = ExperimentRunner(registry)._build_pipeline(
        _config(embedder=ComponentRef(kind="embedder", component_id="another_model")),
    )
    # The stub keeps whatever it was constructed with, so this reads what the
    # build handed over and not what the stub set for itself.
    assert pipeline.embedder is asked_for
