"""The health check asks a store where the installation says it is.

It asked `http://host:port` whatever the environment said, so a store behind
TLS with a key was reported as down while every run against it worked, a
database addressed by a `mongodb+srv://` name had a port probed that the name
does not carry, and a machine whose stores are all somewhere else was told to
install Docker.
"""
from __future__ import annotations

import importlib
from typing import Any

import pytest


@pytest.fixture
def doctor(monkeypatch):
    """The module as it reads a given environment, reloaded per test."""
    def load(**env: str) -> Any:
        for name in ("QDRANT_HOST", "QDRANT_PORT", "QDRANT_HTTPS", "QDRANT_URL", "QDRANT_API_KEY",
                     "OPENSEARCH_HOST", "OPENSEARCH_PORT", "OPENSEARCH_USE_SSL", "OPENSEARCH_USER",
                     "OPENSEARCH_PASSWORD", "OPENSEARCH_VERIFY_CERTS", "MONGODB_URL", "REDIS_URL",
                     "NEO4J_URI"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        import tools.doctor as module
        return importlib.reload(module)
    return load


def _asked(monkeypatch, module: Any, answer: Any = None) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(url: str, timeout: float = 4.0, **kwargs: Any) -> tuple[bool, Any]:
        calls.append({"url": url, **kwargs})
        return True, answer if answer is not None else {"result": {"collections": []}, "status": "green"}
    monkeypatch.setattr(module, "_http", fake)
    return calls


def test_a_hosted_vector_store_is_asked_over_tls_with_its_key(doctor, monkeypatch) -> None:
    module = doctor(QDRANT_HOST="q.example.org", QDRANT_HTTPS="true", QDRANT_API_KEY="the-key")
    calls = _asked(monkeypatch, module)
    check = module.check_qdrant()
    assert calls[0]["url"] == "https://q.example.org:6333/collections"
    assert calls[0]["headers"] == {"api-key": "the-key"}
    assert check.status == module.OK and "the-key" not in (check.component + check.detail)


def test_a_hosted_lexical_store_is_asked_with_its_credentials(doctor, monkeypatch) -> None:
    module = doctor(OPENSEARCH_HOST="s.example.org", OPENSEARCH_PORT="443", OPENSEARCH_USE_SSL="true",
                    OPENSEARCH_USER="u", OPENSEARCH_PASSWORD="p")
    calls = _asked(monkeypatch, module)
    module.check_opensearch()
    assert calls[0]["url"] == "https://s.example.org:443/_cluster/health"
    assert calls[0]["auth"] == ("u", "p")


def test_a_store_that_is_elsewhere_is_not_fixed_by_starting_containers(doctor, monkeypatch) -> None:
    module = doctor(QDRANT_HOST="q.example.org")
    monkeypatch.setattr(module, "_http", lambda url, timeout=4.0, **kw: (False, "refused"))
    check = module.check_qdrant()
    assert check.status == module.FAIL and "make infra" not in check.fix


def test_a_local_store_keeps_its_old_advice(doctor, monkeypatch) -> None:
    module = doctor()
    monkeypatch.setattr(module, "_http", lambda url, timeout=4.0, **kw: (False, "refused"))
    assert module.check_qdrant().fix == "make infra"


def test_a_database_named_by_a_seed_list_has_no_port_probed(doctor, monkeypatch) -> None:
    """`mongodb+srv://` names a DNS record and carries no port. Probing
    27017 on it reported a hosted database as down."""
    module = doctor(MONGODB_URL="mongodb+srv://user:pw@cluster0.example.mongodb.net/db")
    probed: list[tuple[str, int]] = []
    monkeypatch.setattr(module, "_port_open", lambda host, port, timeout=2.0: probed.append((host, port)) or False)

    class _Client:
        def __init__(self, url: str, **kwargs: Any) -> None:
            self.url = url

        def __getitem__(self, name: str) -> Any:
            return self

        def list_collection_names(self) -> list[str]:
            return ["a", "b"]

    import pymongo
    monkeypatch.setattr(pymongo, "MongoClient", _Client)
    check = module.check_mongodb()
    assert probed == [] and check.status == module.OK
    assert "pw" not in (check.component + check.detail), "the address is printed without its password"


def test_docker_is_not_asked_for_when_every_store_is_elsewhere(doctor, monkeypatch) -> None:
    module = doctor(QDRANT_HOST="q.example.org", OPENSEARCH_HOST="s.example.org",
                    MONGODB_URL="mongodb+srv://u:p@c.example.mongodb.net/db",
                    REDIS_URL="rediss://r.example.org:6380")
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    assert module.check_docker().status == module.OFF


def test_docker_is_still_asked_for_when_a_store_is_local(doctor, monkeypatch) -> None:
    module = doctor(QDRANT_HOST="q.example.org")
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    assert module.check_docker().status == module.FAIL
