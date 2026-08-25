"""GET /models — proxies Ollama's /api/tags. `capabilities`/`completion_only`
added so the question generator (services/api_gateway/routers/generation.py)
can offer only text-generation-capable models, excluding the embedding
models Ollama also serves locally (BGE-M3, nomic-embed-text)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers.settings import router

_TAGS_RESPONSE = {
    "models": [
        {"name": "qwen3:8b", "size": 5_000_000_000, "modified_at": "t", "capabilities": ["completion", "tools"]},
        {"name": "BGE-M3:latest", "size": 500_000_000, "modified_at": "t", "capabilities": ["embedding"]},
    ]
}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _mock_httpx_client():
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = _TAGS_RESPONSE
    instance = MagicMock()
    instance.get = AsyncMock(return_value=resp)
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    return instance


def test_models_includes_capabilities(client) -> None:
    with patch("httpx.AsyncClient", return_value=_mock_httpx_client()):
        resp = client.get("/models")
    assert resp.status_code == 200
    names = {m["name"]: m["capabilities"] for m in resp.json()}
    assert names["qwen3:8b"] == ["completion", "tools"]
    assert names["BGE-M3:latest"] == ["embedding"]


def test_models_completion_only_excludes_embedding_models(client) -> None:
    with patch("httpx.AsyncClient", return_value=_mock_httpx_client()):
        resp = client.get("/models?completion_only=true")
    names = [m["name"] for m in resp.json()]
    assert names == ["qwen3:8b"]
