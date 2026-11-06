"""A gateway started while a service is down serves the real component.

It used to probe each service once at start-up and, finding one down, put a
stub in its place for the life of the process: the stub generator's canned
text, a dense half that returned whatever the stub held, an in-memory graph.
A gateway started a minute before the stack, which is the ordinary order on
a laptop, then answered every run with the stub, and the runs finished green.

The real components connect when first used, so they are built whatever the
probe says. A run against a service that is still down fails on its
questions and says so, and one started after the service came up works,
with no restart. A stub is still served when a flag asks for one, which is a
choice and not an accident.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

import services.api_gateway.main as M
from adapters.generator_stub import GeneratorStub
from adapters.ollama_generator import OllamaGenerator
from adapters.qdrant import QdrantRetriever, QdrantRetrieverStub


class _Embedder:
    embedder_id = "bge_m3"


def _unreachable(*args: Any, **kwargs: Any) -> Any:
    raise httpx.ConnectError("connection refused")


def test_a_store_down_at_start_up_still_gets_the_real_retriever(monkeypatch) -> None:
    import adapters.connections as C

    def client(self: Any, timeout: float = 60) -> Any:
        class _Down:
            def get_collections(self) -> Any:
                raise ConnectionError("connection refused")
        return _Down()
    monkeypatch.setattr(C.QdrantTarget, "client", client)
    monkeypatch.setattr(M, "_USE_REAL_QDRANT", True)
    assert isinstance(M._build_retriever(_Embedder()), QdrantRetriever)


def test_a_stub_retriever_is_served_only_when_asked_for(monkeypatch) -> None:
    monkeypatch.setattr(M, "_USE_REAL_QDRANT", False)
    assert isinstance(M._build_retriever(_Embedder()), QdrantRetrieverStub)


def test_a_model_server_down_at_start_up_still_gets_the_real_generator(monkeypatch) -> None:
    monkeypatch.setattr(M, "_USE_OLLAMA", True)
    monkeypatch.setattr(httpx, "get", _unreachable)
    assert isinstance(M._build_generator(), OllamaGenerator)


def test_a_stub_generator_is_served_only_when_asked_for(monkeypatch) -> None:
    monkeypatch.setattr(M, "_USE_OLLAMA", False)
    assert isinstance(M._build_generator(), GeneratorStub)


@pytest.fixture
def registered(monkeypatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(M.registry, "register", lambda kind, name, obj: seen.__setitem__(f"{kind}:{name}", obj))
    return seen


def test_a_graph_store_down_at_start_up_still_gets_the_real_graph(monkeypatch, registered) -> None:
    from adapters.neo4j_graph import Neo4jGraphRetriever
    monkeypatch.setattr(Neo4jGraphRetriever, "is_available", lambda self: True)
    monkeypatch.setattr(Neo4jGraphRetriever, "verify", lambda self: False)
    M._register_graph_pipeline(object(), _Embedder(), GeneratorStub())
    assert isinstance(registered["retriever:graph_hybrid"]._graph, Neo4jGraphRetriever)


def test_a_gateway_without_the_graph_driver_serves_the_in_memory_graph(monkeypatch, registered) -> None:
    """No driver installed is how the platform is configured, and not a
    service that happens to be down: the in-memory graph keeps the option
    working in development and CI, as before."""
    from adapters.lightrag import LightRagRetrieverStub
    from adapters.neo4j_graph import Neo4jGraphRetriever
    monkeypatch.setattr(Neo4jGraphRetriever, "is_available", lambda self: False)
    M._register_graph_pipeline(object(), _Embedder(), GeneratorStub())
    assert isinstance(registered["retriever:graph_hybrid"]._graph, LightRagRetrieverStub)


def test_a_question_asked_of_a_store_that_is_down_fails_and_says_so() -> None:
    """The refusal of the connection itself, named: an earlier draft of this
    test caught any exception and passed on a wrong call to `retrieve`."""
    from qdrant_client.http.exceptions import ResponseHandlingException

    from adapters.connections import QdrantTarget
    retriever = QdrantRetriever(target=QdrantTarget(host="127.0.0.1", port=1))
    with pytest.raises(ResponseHandlingException, match="refused"):
        retriever.retrieve("how often", 3, query_vector=[0.0] * 1024)
