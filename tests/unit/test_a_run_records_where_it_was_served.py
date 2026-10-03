"""A run records which services answered for it, and where they are.

Once a Realm can keep its stores and its model somewhere else, where a run's
questions and fragments went is part of what the run is. It was recorded
nowhere: a run named its model and not the server, and its stores not at all.
What is recorded carries no key.
"""
from __future__ import annotations

import json
from typing import Any

from adapters.bge_m3 import BgeM3Embedder
from adapters.connections import OpenSearchTarget, QdrantTarget
from adapters.openai_compatible import OpenAICompatibleGenerator
from core.experiment.runner import ExperimentResult, where_it_was_served
from core.interfaces import SayingWhereItReads
from core.retrieval.hybrid import HybridRetriever


class _Dense:
    retriever_id = "dense"

    def reads_from(self) -> dict[str, dict[str, Any]]:
        target = QdrantTarget(host="q.example.org", port=6333, https=True, api_key="store-key")
        return {"qdrant": {"address": target.base_url, "remote": target.is_remote}}

    def retrieve(self, query: str, k: int, filters: Any = None, query_vector: Any = None) -> list[Any]:
        return []


class _Sparse:
    retriever_id = "sparse"

    def reads_from(self) -> dict[str, dict[str, Any]]:
        target = OpenSearchTarget(host="localhost", port=9200, user="u", password="store-pass")
        return {"opensearch": {"address": target.base_url, "remote": target.is_remote}}

    def retrieve(self, query: str, k: int, filters: Any = None, query_vector: Any = None) -> list[Any]:
        return []


class _Pipeline:
    def __init__(self) -> None:
        self._embedder = BgeM3Embedder()
        self._retriever = HybridRetriever(_Dense(), _Sparse(), self._embedder)
        self._generator = OpenAICompatibleGenerator(
            base_url="https://api.example.org/v1", api_key="model-key", model="answers")


def test_every_service_of_a_run_is_named_with_where_it_is() -> None:
    served = where_it_was_served(_Pipeline())
    assert served["generator"] == {"provider": "openai_compatible", "address": "https://api.example.org/v1",
                                   "model": "answers", "remote": True}
    assert served["embedder"] == {"provider": "in_process", "model": "bge_m3", "remote": False}
    assert served["stores"] == {
        "qdrant": {"address": "https://q.example.org:6333", "remote": True},
        "opensearch": {"address": "http://localhost:9200", "remote": False}}
    assert served["leaves_this_machine"] == ["generator", "qdrant"]


def test_nothing_recorded_can_open_anything() -> None:
    said = json.dumps(where_it_was_served(_Pipeline()))
    assert "model-key" not in said and "store-key" not in said and "store-pass" not in said


def test_a_merging_retriever_says_where_both_its_halves_read() -> None:
    assert isinstance(_Pipeline()._retriever, SayingWhereItReads)


def test_a_pipeline_whose_parts_say_nothing_records_that_nothing_was_said() -> None:
    class _Bare:
        _retriever = object()
        _embedder = object()
        _generator = object()
    served = where_it_was_served(_Bare())
    assert served["stores"] == {} and served["leaves_this_machine"] == []


def test_a_system_reached_over_http_is_named_as_the_one_place_everything_went() -> None:
    class _Http:
        _url = "https://rag.example.org/query"
    served = where_it_was_served(_Http())
    assert served["external_system"] == {"address": "https://rag.example.org", "remote": True}
    assert served["leaves_this_machine"] == ["external_system"]


def test_the_record_travels_with_the_run() -> None:
    from core.experiment.config import ComponentRef, ExperimentConfig
    result = ExperimentResult(config=ExperimentConfig(
        name="c", chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama")))
    result.served_by = {"leaves_this_machine": ["generator"]}
    assert result.to_dict()["served_by"] == {"leaves_this_machine": ["generator"]}
