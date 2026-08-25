"""POST /prompts/generate — AI-drafted prompt (pre-fills the new-prompt
form, never saves anything itself).

Found live: prompts were only ever written by hand, with no way to get a
first draft tailored to a specific corpus's actual content and a specific
model. Mirrors services/api_gateway/routers/generation.py's question
generator's own "sample real chunks, ask the chosen Ollama model, tolerant
JSON parse" shape, but as a single one-shot call (no background job/WS
progress — there's exactly one output to wait for, not a multi-minute
batch of many LLM calls).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers.prompts import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def _body(**overrides):
    return {"realm_id": "acme", "corpus_id": "handbook_01", "model": "qwen3:8b", **overrides}


def test_generate_prompt_draft_happy_path(client):
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = [_chunk("A sample fragment from the corpus.")]
    llm_response = (
        '{"name": "Technical documentation", "description": "Answers from the manuals.", '
        '"template": "Context:\\n{context}\\n\\nQuestion: {query}\\nAnswer:"}'
    )

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate", return_value=llm_response):
        resp = client.post("/prompts/generate", json=_body())

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Technical documentation"
    assert body["description"] == "Answers from the manuals."
    assert "{context}" in body["template"]
    assert "{query}" in body["template"]


def test_generate_prompt_draft_502_when_placeholders_missing(client):
    """The model must include the LITERAL {context}/{query} tokens, not
    fill them in with real content from the sampled excerpts — a template
    without both is unusable regardless of how well-formed the JSON is."""
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = [_chunk("A sample.")]
    llm_response = '{"name": "X", "description": "Y", "template": "Answer the question about A sample."}'

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate", return_value=llm_response):
        resp = client.post("/prompts/generate", json=_body())

    assert resp.status_code == 502


def test_generate_prompt_draft_502_when_response_unparseable(client):
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = [_chunk("A sample.")]

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate", return_value="not JSON at all"):
        resp = client.post("/prompts/generate", json=_body())

    assert resp.status_code == 502


def test_generate_prompt_draft_503_when_corpus_empty(client):
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = []

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant):
        resp = client.post("/prompts/generate", json=_body())

    assert resp.status_code == 503


def test_generate_prompt_draft_503_when_qdrant_unavailable(client):
    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", side_effect=ConnectionError("no route to host")):
        resp = client.post("/prompts/generate", json=_body())

    assert resp.status_code == 503


def test_generate_prompt_draft_502_when_llm_call_fails(client):
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = [_chunk("A sample.")]

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate", side_effect=TimeoutError("[Errno 60]")):
        resp = client.post("/prompts/generate", json=_body())

    assert resp.status_code == 502


def test_generate_prompt_draft_samples_at_most_three_chunks(client):
    """Keeps the meta-prompt bounded regardless of corpus size — the
    excerpts are only for domain/tone context, not a full corpus dump."""
    mock_qdrant = MagicMock()
    mock_qdrant.scroll_all.return_value = [_chunk(f"fragment {i}") for i in range(50)]
    captured: dict = {}

    def _fake_generate(prompt: str, **kwargs):
        captured["prompt"] = prompt
        return '{"name": "X", "description": "Y", "template": "{context} {query}"}'

    with patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("services.api_gateway.routers.corpus._resolve_qdrant", return_value=mock_qdrant), \
         patch("adapters.ollama_generator.OllamaGenerator.generate", side_effect=_fake_generate):
        resp = client.post("/prompts/generate", json=_body())

    assert resp.status_code == 200
    import re
    included = set(re.findall(r"fragment (\d+)\b", captured["prompt"]))
    assert len(included) <= 3
