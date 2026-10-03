"""A corpus is loaded with the embedder that is named for it.

The loader built the local embedder whatever was asked, and the registry
wrote that embedder's name beside every corpus whatever had filled it. A
corpus is searched with the vectors it was loaded with, so the name beside it
is what tells a run which embedder to bring.
"""
from __future__ import annotations

import pytest

from services.api_gateway.routers import corpus as corpus_module
from services.ingestion import cli


def test_the_local_embedder_is_what_a_load_gets_when_nothing_is_named(monkeypatch) -> None:
    monkeypatch.delenv("CAUSA_EMBEDDER", raising=False)
    assert cli._embedder_named(None).embedder_id == "bge_m3"


def test_a_named_remote_embedder_is_built_from_the_loader_s_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.org/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embeds")
    embedder = cli._embedder_named("openai_compatible")
    assert embedder.embedder_id == "openai_compatible__embeds"
    assert embedder.served_from()["address"] == "https://api.example.org/v1"


def test_the_environment_names_the_embedder_when_the_command_does_not(monkeypatch) -> None:
    monkeypatch.setenv("CAUSA_EMBEDDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embeds")
    assert cli._embedder_named(None).embedder_id == "openai_compatible__embeds"


def test_an_embedder_nobody_has_is_refused_before_anything_is_loaded() -> None:
    with pytest.raises(ValueError, match="no-such-embedder"):
        cli._embedder_named("no-such-embedder")


@pytest.mark.asyncio
async def test_the_gateway_hands_the_loader_the_realm_s_embedding_server(monkeypatch, tmp_path):
    seen: dict = {}

    async def fake_exec(*argv, **kwargs):
        seen["argv"] = list(argv)
        seen["env"] = dict(kwargs.get("env") or {})
        raise RuntimeError("stop after capturing the invocation")

    monkeypatch.setattr(corpus_module.asyncio, "create_subprocess_exec", fake_exec)
    corpus_module._progress["job-e"] = []
    await corpus_module._run_ingestion(
        "job-e", tmp_path, "fixed", 512, 64, [], {"n_files": 1, "realm_id": "acme"},
        corpus_id="acme-corpus", embedder="openai_compatible",
        realm_resources={"openai_compatible": {
            "type": "openai_compatible", "base_url": "https://api.example.org/v1",
            "api_key": "embed-key", "model": "answers", "embedding_model": "embeds"}},
    )
    assert seen["argv"][seen["argv"].index("--embedder") + 1] == "openai_compatible"
    assert "embed-key" not in " ".join(seen["argv"])
    env = seen["env"]
    assert (env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"], env["OPENAI_EMBEDDING_MODEL"]) == (
        "https://api.example.org/v1", "embed-key", "embeds")


def test_the_registry_records_the_embedder_that_filled_the_corpus() -> None:
    resources = {"openai_compatible": {"base_url": "https://api.example.org/v1", "embedding_model": "Embed/3"}}
    assert corpus_module._embedder_id_for("bge_m3", resources) == "bge_m3"
    assert corpus_module._embedder_id_for("openai_compatible", resources) == "openai_compatible__embed-3"


@pytest.mark.asyncio
async def test_a_load_reports_the_fragments_it_made_and_not_the_embedder_s_cache(monkeypatch, tmp_path):
    """Found live, loading with an embedder that remembers nothing between
    calls: the job reported no fragments for a corpus of 220. The count was
    read from the size of the embedder's cache, which for the local embedder
    happens to equal the number of distinct fragments."""
    class _Process:
        returncode = 0

        def __init__(self) -> None:
            lines = ["ingest.file_done chunks=7 file=a.md\n",
                     "ingest.done files=40 total_chunks=220 strategy=fixed hits=0 misses=220 "
                     "hit_ratio=0.0 size=0\n"]
            self.stdout = self._read(lines)

        @staticmethod
        async def _read(lines):
            for line in lines:
                yield line.encode()

        async def wait(self) -> int:
            return 0

    async def fake_exec(*argv, **kwargs):
        return _Process()

    async def no_registry(**kwargs):
        return None

    monkeypatch.setattr(corpus_module.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(corpus_module, "_register_corpus", no_registry)
    saved: dict = {}

    async def keep(collection, query, doc):
        saved.update(doc)

    monkeypatch.setattr("adapters.mongodb.upsert_one", keep)
    monkeypatch.setattr("adapters.mongodb.insert_one", keep, raising=False)
    corpus_module._progress["job-n"] = []
    doc = {"n_files": 40, "realm_id": "acme"}
    await corpus_module._run_ingestion("job-n", tmp_path, "fixed", 512, 64, [], doc, corpus_id="c")
    assert doc["n_chunks"] == 220
