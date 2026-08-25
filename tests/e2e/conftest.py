"""Fixtures for the end-to-end walkthrough over the demo material.

These tests drive the real gateway against the real stack: Qdrant, OpenSearch,
MongoDB and Ollama. Nothing is mocked, because the point is to prove the
platform works on a machine that has just cloned the repository, and a mock
proves only that the mock works.

Isolation is by realm id. Every run issues `e2e-<hex8>`, and Qdrant collection
and OpenSearch index names are realm-scoped by construction
(`adapters/qdrant.py#_collection_name`), so one run cannot see another's data
or a developer's own realms. Teardown removes everything carrying that prefix,
following the rule the integration fixtures already learned the hard way: clean
up by prefix rather than by the one exact name the test happened to create,
because derived names are the ones that accumulate.

Run with:
    make test-e2e

Every fixture skips rather than fails when a service is missing, so a partial
stack produces a clear "not run" instead of a misleading red.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from .realm_cleanup import purge_realm, sweep_stale_realms

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")



def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: full-platform walkthrough over the demo realm")


def _reachable(host: str, port: int) -> bool:
    import socket
    sock = socket.socket()
    sock.settimeout(2)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


@pytest.fixture(scope="session")
def services() -> dict[str, bool]:
    """Which backing services answer. Reported once so a skip says which one."""

    import httpx

    try:
        httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3).raise_for_status()
        ollama_up = True
    except Exception:
        ollama_up = False

    return {
        "qdrant": _reachable(QDRANT_HOST, QDRANT_PORT),
        "opensearch": _reachable(OPENSEARCH_HOST, OPENSEARCH_PORT),
        "mongodb": _reachable(os.getenv("MONGODB_HOST", "localhost"), 27017),
        "ollama": ollama_up,
    }


@pytest.fixture(scope="session")
def require_stack(services: dict[str, bool]) -> None:
    missing = [name for name, up in services.items() if not up and name != "ollama"]
    if missing:
        pytest.skip(
            f"not reachable: {', '.join(missing)}. "
            f"Start the stack with: make infra"
        )


@pytest.fixture(scope="session")
def require_ollama(services: dict[str, bool]) -> None:
    if not services["ollama"]:
        pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}. Install it: https://ollama.com/download")


@pytest.fixture(scope="session")
def realm_id() -> str:
    """A realm nobody else is using, so the run cannot collide with real data."""
    return f"e2e-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def state() -> dict[str, Any]:
    """Carries ids between ordered steps.

    Each step records what the next one needs; each step that needs something
    skips when it is absent, so one early failure reports itself once rather
    than as a cascade of unrelated errors.
    """
    return {}


@pytest.fixture(scope="session")
def client(require_stack):  # noqa: ANN001
    """The real gateway app over the real stack.

    USE_REAL_BGE_M3 is forced on: stub vectors would let retrieval "work"
    while measuring nothing, which is the exact failure this platform exists
    to detect and would make every assertion below meaningless.
    """
    os.environ["USE_REAL_BGE_M3"] = "true"
    from fastapi.testclient import TestClient

    from services.api_gateway.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session", autouse=True)
def cleanup(realm_id: str, request: pytest.FixtureRequest):  # noqa: ANN201
    """Remove everything this run created, whether it passed or not."""
    try:
        stale = sweep_stale_realms(keep=realm_id)
        if stale:
            print(f"\ne2e: removed {len(stale)} realm(s) left by an interrupted run: {', '.join(stale)}")
    except Exception as exc:
        print(f"  e2e stale-realm sweep skipped: {exc}")

    yield

    removed: dict[str, int] = {}

    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        names = [c.name for c in qc.get_collections().collections if c.name.startswith(realm_id)]
        for name in names:
            qc.delete_collection(name)
        removed["qdrant_collections"] = len(names)
    except Exception as exc:
        removed["qdrant_collections"] = -1
        print(f"  qdrant cleanup skipped: {exc}")

    try:
        from opensearchpy import OpenSearch
        os_client = OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}], timeout=5)
        indices = [n for n in os_client.indices.get_alias(index="*") if realm_id in n]
        for name in indices:
            os_client.indices.delete(index=name, ignore=[404])
        removed["opensearch_indices"] = len(indices)
    except Exception as exc:
        removed["opensearch_indices"] = -1
        print(f"  opensearch cleanup skipped: {exc}")

    try:
        removed["mongo_documents"] = purge_realm(realm_id)
    except Exception as exc:
        removed["mongo_documents"] = -1
        print(f"  mongo cleanup skipped: {exc}")

    print(f"\ne2e cleanup for realm {realm_id}: " +
          ", ".join(f"{k}={v}" for k, v in removed.items()))
