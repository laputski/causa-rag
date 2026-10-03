"""The models offered for a Realm are the ones its own server has.

The list came from the gateway's local server whatever Realm asked, so a
Realm served by another one was offered models it could not run and none of
those it could.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import adapters.openai_compatible as oc
import services.api_gateway.model_servers as servers
from services.api_gateway.routers import settings as S


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(S.router)
    with TestClient(app) as c:
        yield c


def _realm(monkeypatch, resources: dict[str, Any]) -> None:
    async def the_resources(realm_id):
        return resources if realm_id else {}
    monkeypatch.setattr(servers, "model_resources", the_resources)


def test_a_realm_on_a_hosted_server_is_offered_that_server_s_models(client, monkeypatch) -> None:
    _realm(monkeypatch, {"openai_compatible": {"base_url": "https://api.example.org/v1", "api_key": "k"}})
    monkeypatch.setattr(oc.OpenAICompatibleGenerator, "list_models", lambda self: ["model-a", "model-b"])
    models = client.get("/models?realm_id=acme").json()
    assert [m["name"] for m in models] == ["model-a", "model-b"]
    assert {m["provider"] for m in models} == {"openai_compatible"}


def test_a_hosted_server_s_models_are_kept_when_only_those_that_write_are_wanted(client, monkeypatch) -> None:
    """The API does not say what a model can do, so none can be excluded for
    not saying it."""
    _realm(monkeypatch, {"openai_compatible": {"base_url": "https://api.example.org/v1"}})
    monkeypatch.setattr(oc.OpenAICompatibleGenerator, "list_models", lambda self: ["model-a"])
    assert [m["name"] for m in client.get("/models?realm_id=acme&completion_only=true").json()] == ["model-a"]


def test_a_realm_with_its_own_local_server_is_asked_there(client, monkeypatch) -> None:
    _realm(monkeypatch, {"ollama": {"host": "models.internal", "port": 11500}})
    asked: list[str] = []

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *a: Any) -> bool:
            return False

        async def get(self, url: str) -> Any:
            asked.append(url)

            class _R:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, Any]:
                    return {"models": [{"name": "m", "size": 5e9, "capabilities": ["completion"]}]}
            return _R()

    monkeypatch.setattr(S.httpx, "AsyncClient", _Client)
    models = client.get("/models?realm_id=acme").json()
    assert asked == ["http://models.internal:11500/api/tags"]
    assert models[0]["name"] == "m" and models[0]["provider"] == "ollama"


def test_a_realm_with_both_servers_lists_the_one_a_call_names(client, monkeypatch) -> None:
    """The form of a new run offers a generator and then the models of that
    generator's server. With both declared the unnamed choice is the local
    one, so without the name the hosted server's models could not be listed
    at all."""
    _realm(monkeypatch, {"ollama": {"host": "localhost", "port": 11434},
                         "openai_compatible": {"base_url": "https://api.example.org/v1", "api_key": "k"}})
    monkeypatch.setattr(oc.OpenAICompatibleGenerator, "list_models", lambda self: ["model-a"])
    models = client.get("/models?realm_id=acme&provider=openai_compatible").json()
    assert [(m["name"], m["provider"]) for m in models] == [("model-a", "openai_compatible")]


def test_a_hosted_server_that_cannot_be_asked_offers_no_model(client, monkeypatch) -> None:
    """The fallback for a failed listing is the name of the local default
    model. Offered under a hosted server it would be a model that server
    never had, chosen from a list that looks like its own."""
    _realm(monkeypatch, {"openai_compatible": {"base_url": "https://api.example.org/v1"}})

    def unreachable(self) -> list[str]:
        raise RuntimeError("the model server answered 401")
    monkeypatch.setattr(oc.OpenAICompatibleGenerator, "list_models", unreachable)
    assert client.get("/models?realm_id=acme").json() == []
