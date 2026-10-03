"""A corpus is read with the embedder that filled it, by every reader.

A run could name the embedder a corpus was loaded with. The chat, the corpus
browser, its health check and the drafting of questions could not: each built
its retriever for the gateway's own embedder, whose collection for such a
corpus is empty. The registry records which embedder filled a corpus, so the
registry is asked.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from adapters.openai_compatible import OpenAICompatibleEmbedder
from services.api_gateway import model_servers as M
from services.api_gateway.routers import corpus as C
from tests.unit._fake_mongo import FakeMongo

_HOSTED = {"openai_compatible": {"base_url": "https://api.example.org/v1", "embedding_model": "embeds"}}


def _registry(monkeypatch, embedder_id: str | None) -> None:
    docs = [] if embedder_id is None else [{
        "realm_id": "acme", "corpus_id": "c", "backends": {"qdrant": {"embedder_id": embedder_id}}}]
    FakeMongo({"corpora": docs}).install(monkeypatch)


def test_the_registry_says_which_embedder_filled_a_corpus(monkeypatch) -> None:
    _registry(monkeypatch, "openai_compatible__embeds")
    assert asyncio.run(C._corpus_embedder_id("acme", "c")) == "openai_compatible__embeds"


def test_a_corpus_the_registry_does_not_know_is_read_as_asked(monkeypatch) -> None:
    _registry(monkeypatch, None)
    assert asyncio.run(C._corpus_embedder_id("acme", "c")) == "bge_m3"
    assert asyncio.run(C._corpus_embedder_id("acme", "c", "another")) == "another"


def test_a_registry_that_cannot_be_reached_changes_nothing(monkeypatch) -> None:
    async def refuse(collection, query, projection=None):
        raise ConnectionError("down")
    monkeypatch.setattr("adapters.mongodb.find_one", refuse)
    assert asyncio.run(C._corpus_embedder_id("acme", "c")) == "bge_m3"


def test_the_embedder_for_a_corpus_is_built_from_the_identifier_the_registry_gives() -> None:
    local = object()
    assert M.embedder_for("bge_m3", _HOSTED, local) is local
    hosted = M.embedder_for("openai_compatible__embeds", _HOSTED, local)
    assert isinstance(hosted, OpenAICompatibleEmbedder) and hosted.embedder_id == "openai_compatible__embeds"


def test_an_identifier_the_realm_s_server_does_not_answer_to_is_refused() -> None:
    """The collection was filled by one model and the Realm now names
    another. Querying with the new one would search one model's vectors with
    another's, so nothing is built."""
    with pytest.raises(M.EmbedderMismatch):
        M.embedder_for("openai_compatible__some-older-model", _HOSTED, object())


@pytest.mark.parametrize("module,function", [
    ("services.api_gateway.routers.corpus", "browse_chunks"),
    ("services.api_gateway.routers.corpus", "corpus_health"),
    ("services.api_gateway.routers.corpus", "run_chunk_coherence_diagnostics"),
    ("services.api_gateway.routers.generation", "generate_questions"),
    ("services.api_gateway.routers.prompts", "generate_prompt_draft"),
    ("services.api_gateway.main", "_build_realm_scoped_retriever"),
])
def test_every_reader_of_a_corpus_asks_the_registry_which_embedder(module: str, function: str) -> None:
    import importlib
    source = inspect.getsource(getattr(importlib.import_module(module), function))
    assert "_corpus_embedder_id" in source, f"{function} reads a corpus without asking what filled it"
