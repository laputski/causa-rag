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
    # The URI and user are not secrets and stay where they were.
    assert "--neo4j-uri" in seen["argv"]
