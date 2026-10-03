"""Answers are measured with the platform's own embedder, whatever the run
searched with.

Found live, on the first run that named an embedder served over HTTP: the
similarity of each answer to its reference was computed with the embedder the
configuration names, which is the system under test. Two runs differing in
their embedder would then be scored by two different instruments, and the
difference between their scores would be partly the instruments'. The judge
of assertions is kept out of the configuration for the same reason.
"""
from __future__ import annotations

from typing import Any

import pytest

from core.experiment.config import ComponentRef, ExperimentConfig
from core.registry import registry
from services.api_gateway.routers import experiments as E


class _Embedder:
    def __init__(self, embedder_id: str) -> None:
        self.embedder_id = embedder_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


def _config(embedder: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="c", chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id=embedder),
        generator=ComponentRef(kind="generator", component_id="ollama"))


@pytest.fixture
def embedders(monkeypatch) -> dict[str, Any]:
    known = {"bge_m3": _Embedder("bge_m3"), "openai_compatible": _Embedder("openai_compatible")}

    def resolve(kind: str, component_id: str) -> Any:
        if kind != "embedder" or component_id not in known:
            raise KeyError(component_id)
        return known[component_id]
    monkeypatch.setattr(registry, "resolve", resolve)
    return known


def test_a_run_searching_with_another_embedder_is_still_measured_with_the_platform_s(embedders) -> None:
    assert E._measuring_embedder(_config("openai_compatible")) is embedders["bge_m3"]


def test_a_run_on_the_platform_s_own_embedder_is_measured_as_before(embedders) -> None:
    assert E._measuring_embedder(_config("bge_m3")) is embedders["bge_m3"]


def test_an_installation_without_the_platform_s_embedder_falls_back_to_the_run_s(embedders) -> None:
    """Never an error: a run that cannot be measured the usual way is still
    measured, by what it has."""
    del embedders["bge_m3"]
    assert E._measuring_embedder(_config("openai_compatible")) is embedders["openai_compatible"]
