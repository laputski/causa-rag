"""A run is answered by the model server of the Realm that launched it.

Established by reading every place a generator is built: the chat and the
feedback triage used the Realm's own model server, and a run, the drafting of
questions and the drafting of prompts used the one the gateway started with.
A Realm that declared its own server was answered by another, and the run
recorded the model's name and not where it had been served.
"""
from __future__ import annotations

from typing import Any

from adapters.ollama_generator import OllamaGenerator
from adapters.openai_compatible import OpenAICompatibleGenerator
from core.experiment.runner import _served_by
from core.interfaces import ServedByARealm


def test_both_generators_can_be_bound_to_a_realm() -> None:
    assert isinstance(OllamaGenerator(), ServedByARealm)
    assert isinstance(OpenAICompatibleGenerator(), ServedByARealm)


def test_the_local_server_of_a_realm_replaces_the_gateway_s() -> None:
    gateway = OllamaGenerator(base_url="http://localhost:11434", model="start-up-model", seed=3)
    bound = _served_by(gateway, {"ollama": {"host": "models.internal", "port": 11500, "model": "realm-model"}})
    assert bound.served_from() == {"provider": "ollama", "address": "http://models.internal:11500",
                                   "model": "realm-model", "remote": False}
    assert bound._seed == 3, "what the run fixed about the sampling is kept"


def test_a_realm_that_declares_no_server_keeps_the_gateway_s() -> None:
    gateway = OllamaGenerator(base_url="http://localhost:11434", model="m")
    assert _served_by(gateway, None) is gateway
    assert _served_by(gateway, {"ollama": None, "openai_compatible": None}) is gateway


def test_a_resource_that_names_no_model_keeps_the_one_already_chosen() -> None:
    gateway = OllamaGenerator(base_url="http://localhost:11434", model="chosen")
    assert _served_by(gateway, {"ollama": {"host": "h", "port": 1}})._model == "chosen"


def test_a_component_that_calls_no_server_is_left_as_it_was() -> None:
    class _Stub:
        def generate(self, prompt: str, **params: Any) -> str:
            return ""
    stub = _Stub()
    assert _served_by(stub, {"ollama": {"host": "h"}}) is stub


def test_the_pipeline_a_run_builds_carries_the_bound_generator() -> None:
    from adapters.bge_m3 import BgeM3Embedder
    from adapters.qdrant import QdrantRetrieverStub
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentRunner
    from core.pipeline import NaivePipeline
    from core.registry import ComponentRegistry

    registry = ComponentRegistry()
    embedder, generator = BgeM3Embedder(), OllamaGenerator(base_url="http://localhost:11434", model="m")
    registry.register("embedder", "bge_m3", embedder)
    registry.register("generator", "ollama", generator)
    registry.register("pipeline", "naive", NaivePipeline(
        retriever=QdrantRetrieverStub(), embedder=embedder, generator=generator))
    config = ExperimentConfig(
        name="c", chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama"))
    pipeline = ExperimentRunner(registry)._build_pipeline(
        config, "acme", model_cfg={"ollama": {"host": "models.internal", "port": 11500}})
    assert pipeline._generator.served_from()["address"] == "http://models.internal:11500"
