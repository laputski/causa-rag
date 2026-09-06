"""The graph point becomes something a person can vary.

A graph pipeline mixes what the graph reached with what the base retriever
found. `graph_weight` decides how much of the ranking the graph gets and
`hops` decides how far from a matched unit it may walk, and those two are the
whole of what the point has to be set.

Neither had a field on a configuration at all. The pipeline could be chosen
and could not be varied: every run of it used whatever the gateway had
constructed at start-up, two graph runs could not differ in anything anybody
had set, and no failure could be staged on it by a setting. That is the
absence this closes.
"""
from __future__ import annotations

from typing import Any

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import _rebind_graph
from core.retrieval.graph_hybrid import GraphHybridRetriever


class _Stub:
    def retrieve(self, **kwargs: Any) -> list:
        return []

    def neighbours(self, *args: Any, **kwargs: Any) -> list:
        return []


def _graph(**kwargs: Any) -> GraphHybridRetriever:
    return GraphHybridRetriever(graph_retriever=_Stub(), base_retriever=_Stub(), **kwargs)


def _config(**fields: Any) -> ExperimentConfig:
    return ExperimentConfig(
        name="c",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
        **fields,
    )


def test_a_configuration_can_carry_both_parameters() -> None:
    config = _config(pipeline_id="graph", graph_weight=0.8, hops=3)
    assert config.graph_weight == 0.8
    assert config.hops == 3


def test_neither_field_moves_a_historical_configuration_hash() -> None:
    """The convention rrf_k already follows: a field added with None as its
    default leaves every run recorded before it existed hashing as it did."""
    before = _config(pipeline_id="graph").config_hash
    after = _config(pipeline_id="graph", graph_weight=None, hops=None).config_hash
    assert before == after


def test_setting_the_weight_reaches_the_retriever() -> None:
    assert _rebind_graph(_graph(graph_weight=0.4), 0.9, None)._graph_weight == 0.9


def test_setting_the_hops_reaches_the_retriever() -> None:
    assert _rebind_graph(_graph(hops=1), None, 4)._hops == 4


def test_setting_one_leaves_the_other_where_it_was() -> None:
    """The failure the merge rebind had: rebuilding for one parameter reset
    another to its default, so changing the weight silently moved the hops."""
    rebound = _rebind_graph(_graph(graph_weight=0.25, hops=3), 0.9, None)
    assert rebound._hops == 3
    assert _rebind_graph(_graph(graph_weight=0.25, hops=3), None, 4)._graph_weight == 0.25


def test_setting_neither_returns_the_retriever_untouched() -> None:
    """None means keep what was constructed, and keeping it means the same
    object: a rebuild that changed nothing would still drop anything a later
    wrapper had put on this one."""
    built = _graph(graph_weight=0.25, hops=3)
    assert _rebind_graph(built, None, None) is built


def test_a_retriever_of_another_kind_is_left_alone() -> None:
    """A graph parameter set on a configuration whose pipeline is not the
    graph one is a mistake in the configuration, and rebuilding a hybrid as
    a graph would answer it by changing the architecture."""
    other = _Stub()
    assert _rebind_graph(other, 0.9, 4) is other


def test_building_a_pipeline_applies_both_parameters() -> None:
    """The half a test of the rebind alone cannot cover.

    Removing the call from the build left every test of `_rebind_graph`
    green, which is the shape of a field that is declared, applied by a
    function nobody calls, and silently decorative. Only building the
    pipeline catches it.
    """
    from core.experiment.runner import ExperimentRunner
    from core.pipeline import NaivePipeline
    from core.registry import ComponentRegistry

    class _Component(_Stub):
        embedder_id = retriever_id = generator_id = "fake"

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def generate(self, prompt: str) -> str:
            return "ok"

    component = _Component()
    registry = ComponentRegistry()
    for kind in ("embedder", "retriever", "generator"):
        registry.register(kind, "fake", component)
    registry.register("pipeline", "graph", NaivePipeline(
        retriever=_graph(graph_weight=0.4, hops=1), embedder=component, generator=component,
        pipeline_id="graph",
    ))

    built = ExperimentRunner(registry)._build_pipeline(
        _config(pipeline_id="graph", graph_weight=0.9, hops=4))
    assert built._retriever._graph_weight == 0.9
    assert built._retriever._hops == 4
