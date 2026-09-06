"""The new-run form stops guessing what a pipeline is made of.

`AGENTS.md` says the form builds itself from the registry, and it did for the
list of pipelines and for nothing else. Two fields beside that list were
written by a case analysis over three known names: the retriever came from
`pipeline_id === 'graph' ? 'graph_hybrid' : 'qdrant_dense'` and the merge
from `pipeline_id === 'hybrid_weighted' ? 'weighted' : 'rrf'`.

So a pipeline outside those names was recorded as a dense retriever fusing by
rank whatever it actually was, and the merge is not bookkeeping: the runner
applies it. A record that contradicts the run is the failure this platform
sells the detection of.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.registry import ComponentRegistry


class _Retriever:
    retriever_id = "qdrant_dense"

    def retrieve(self, **kwargs: Any) -> list:
        return []


class _Hybrid(_Retriever):
    retriever_id = "hybrid"

    def __init__(self, merge: str) -> None:
        self._merge = merge


class _Pipeline:
    def __init__(self, retriever: Any, pipeline_id: str) -> None:
        self._retriever = retriever
        self.pipeline_id = pipeline_id


@pytest.fixture
def described(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, str]]:
    import services.api_gateway.main as gateway

    registry = ComponentRegistry()
    registry.register("pipeline", "naive", _Pipeline(_Retriever(), "naive"))
    registry.register("pipeline", "hybrid_weighted", _Pipeline(_Hybrid("weighted"), "hybrid_weighted"))
    registry.register("pipeline", "hybrid_rrf", _Pipeline(_Hybrid("rrf"), "hybrid_rrf"))
    monkeypatch.setattr(gateway, "registry", registry)
    return asyncio.run(gateway.describe_pipelines())


def test_each_pipeline_names_the_retriever_it_holds(described: dict[str, dict[str, str]]) -> None:
    assert described["naive"]["retriever"] == "qdrant_dense"
    assert described["hybrid_rrf"]["retriever"] == "hybrid"


def test_each_pipeline_names_the_merge_it_was_built_with(
    described: dict[str, dict[str, str]],
) -> None:
    assert described["hybrid_weighted"]["merge_strategy"] == "weighted"
    assert described["hybrid_rrf"]["merge_strategy"] == "rrf"


def test_a_retriever_that_merges_nothing_says_nothing(
    described: dict[str, dict[str, str]],
) -> None:
    """Empty, and not "rrf". A dense pipeline fuses no sources, and recording
    a fusion it did not perform is what the case analysis did for every name
    but one."""
    assert described["naive"]["merge_strategy"] == ""


def test_a_pipeline_nobody_here_has_heard_of_describes_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point. An architecture registered tomorrow is described
    without an edit to the form or to this file."""
    import services.api_gateway.main as gateway

    class _Future(_Retriever):
        retriever_id = "something_new"

    registry = ComponentRegistry()
    registry.register("pipeline", "not_yet_invented", _Pipeline(_Future(), "not_yet_invented"))
    monkeypatch.setattr(gateway, "registry", registry)

    described = asyncio.run(gateway.describe_pipelines())
    assert described["not_yet_invented"]["retriever"] == "something_new"
    assert described["not_yet_invented"]["merge_strategy"] == ""
