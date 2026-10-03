"""The ingestion subprocess gets the graph store's password in its
environment and never on its command line.

Found by reading: the password of a Realm's Neo4j went to the subprocess as
`--neo4j-password <value>`, and a command line is readable by every user on
the machine for as long as the process runs (`ps`, `/proc/<pid>/cmdline`).
The ingestion CLI already falls back to `NEO4J_PASSWORD` when the flag is
absent (`adapters/neo4j_graph.py`), so the environment was always enough.
"""
from __future__ import annotations

import pytest

from services.api_gateway.routers import corpus as corpus_module


@pytest.mark.asyncio
async def test_the_password_travels_in_the_environment(monkeypatch, tmp_path):
    seen: dict = {}

    async def fake_exec(*argv, **kwargs):
        seen["argv"] = list(argv)
        seen["env"] = dict(kwargs.get("env") or {})
        raise RuntimeError("stop after capturing the invocation")

    monkeypatch.setattr(corpus_module.asyncio, "create_subprocess_exec", fake_exec)
    corpus_module._progress["job-1"] = []
    await corpus_module._run_ingestion(
        "job-1", tmp_path, "fixed", 512, 64, [], {"n_files": 1, "realm_id": "acme"},
        corpus_id="acme-corpus",
        realm_resources={"neo4j": {"type": "neo4j", "uri": "bolt://g:7687",
                                    "user": "neo4j", "password": "s3cret-pass"}},
    )

    assert "s3cret-pass" not in " ".join(seen["argv"])
    assert seen["env"]["NEO4J_PASSWORD"] == "s3cret-pass"
    # The address and the user travel the same way now: a store's whole
    # description is read from one place, and the command line names none of it.
    assert seen["env"]["NEO4J_URI"] == "bolt://g:7687" and seen["env"]["NEO4J_USER"] == "neo4j"
    assert not [a for a in seen["argv"] if "neo4j" in a or "qdrant" in a or "opensearch" in a]


@pytest.mark.asyncio
async def test_a_hosted_store_s_key_and_credentials_travel_in_the_environment(monkeypatch, tmp_path):
    """A store behind TLS with a key could not be loaded into at all: the
    loader was given a host and a port on its command line and nothing else."""
    seen: dict = {}

    async def fake_exec(*argv, **kwargs):
        seen["argv"] = list(argv)
        seen["env"] = dict(kwargs.get("env") or {})
        raise RuntimeError("stop after capturing the invocation")

    monkeypatch.setattr(corpus_module.asyncio, "create_subprocess_exec", fake_exec)
    for name in ("QDRANT_URL", "QDRANT_API_KEY", "OPENSEARCH_USER", "OPENSEARCH_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    corpus_module._progress["job-2"] = []
    await corpus_module._run_ingestion(
        "job-2", tmp_path, "fixed", 512, 64, [], {"n_files": 1, "realm_id": "acme"},
        corpus_id="acme-corpus",
        realm_resources={
            "qdrant": {"type": "qdrant", "host": "q.example.org", "port": 6333, "https": True,
                       "api_key": "qdrant-key"},
            "opensearch": {"type": "opensearch", "host": "s.example.org", "port": 443,
                           "use_ssl": True, "user": "u", "password": "search-pass"},
        },
    )
    line = " ".join(seen["argv"])
    assert "qdrant-key" not in line and "search-pass" not in line
    env = seen["env"]
    assert (env["QDRANT_HOST"], env["QDRANT_HTTPS"], env["QDRANT_API_KEY"]) == ("q.example.org", "true", "qdrant-key")
    assert (env["OPENSEARCH_HOST"], env["OPENSEARCH_USE_SSL"], env["OPENSEARCH_USER"],
            env["OPENSEARCH_PASSWORD"]) == ("s.example.org", "true", "u", "search-pass")
