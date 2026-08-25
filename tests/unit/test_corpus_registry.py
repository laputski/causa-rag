"""`corpora` Mongo registry and GET /corpus/collections.

Replaces reliance on the (frequently empty) `corpus_ingests` upload-history
collection for "what corpus_ids exist for this Realm" — see
services/api_gateway/routers/corpus.py#_register_corpus/_list_corpora and
the design notes "Corpora registry".
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers.corpus import _register_corpus, router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def test_collections_endpoint_exists():
    paths = {r.path for r in router.routes}
    assert any(p.endswith("/corpus/collections") for p in paths)


def test_list_collections_filters_by_realm_id(client) -> None:
    docs = [{"id": "1", "realm_id": "demo", "corpus_id": "handbook"}]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=docs)) as mock_find:
        resp = client.get("/corpus/collections?realm_id=demo")

    assert resp.status_code == 200
    assert resp.json() == docs
    assert mock_find.call_args.kwargs["query"] == {"realm_id": "demo"}


def test_list_collections_empty_realm_returns_empty_not_every_realm(client) -> None:
    """Mirrors list_ingests' existing convention (see corpus.py) — a Realm
    with genuinely no registered corpora gets [], never every other Realm's
    corpora leaking through an unfiltered query."""
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=[])):
        resp = client.get("/corpus/collections?realm_id=acme")
    assert resp.json() == []


@pytest.mark.asyncio
async def test_register_corpus_inserts_when_new():
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)) as mock_insert:
        doc = await _register_corpus(
            realm_id="demo", corpus_id="handbook", storage_type="dense_sparse",
            backends={"qdrant": {"collection": "demo__handbook__structure_aware__bge_m3"}},
        )

    assert doc["realm_id"] == "demo"
    assert doc["corpus_id"] == "handbook"
    assert "id" in doc and "created_at" in doc
    mock_insert.assert_called_once()


@pytest.mark.asyncio
async def test_register_corpus_upserts_when_already_registered():
    """Re-running ingest (or the migration script) for the same (realm_id,
    corpus_id) refreshes the record in place — no duplicate registry rows."""
    existing = {"id": "abc123", "realm_id": "demo", "corpus_id": "handbook", "created_at": "2026-01-01"}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update, \
         patch("adapters.mongodb.insert_one", AsyncMock()) as mock_insert:
        doc = await _register_corpus(
            realm_id="demo", corpus_id="handbook", storage_type="dense_sparse",
            backends={"qdrant": {"collection": "new-name"}},
        )

    mock_update.assert_called_once()
    mock_insert.assert_not_called()
    assert doc["id"] == "abc123"
    assert doc["backends"]["qdrant"]["collection"] == "new-name"


# ── POST /corpus/collections — public registration for corpora ──
# prepared outside the platform's own /corpus/ingest (an external ingestor's
# own pipeline writing directly to Qdrant/OpenSearch/Neo4j). Thin wrapper
# over the same _register_corpus() the tests above already exercise.

def test_register_collection_endpoint_exists():
    paths = {r.path for r in router.routes}
    assert any(p.endswith("/corpus/collections") for p in paths)


def test_register_collection_inserts_new_corpus(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)) as mock_insert:
        resp = client.post(
            "/corpus/collections",
            json={
                "realm_id": "acme",
                "corpus_id": "external_docs",
                "storage_type": "dense_only",
                "backends": {"qdrant": {"collection": "acme__external_docs", "embedder_id": "bge_m3"}},
                "owner": "external-ingestor",
            },
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["realm_id"] == "acme"
    assert body["corpus_id"] == "external_docs"
    assert body["owner"] == "external-ingestor"
    assert body["backends"]["qdrant"]["embedder_id"] == "bge_m3"
    mock_insert.assert_called_once()


def test_register_collection_upserts_when_already_registered(client) -> None:
    """Same idempotent upsert as the internal ingest path — re-registering
    an already-known (realm_id, corpus_id) refreshes it, doesn't duplicate."""
    existing = {"id": "xyz", "realm_id": "acme", "corpus_id": "external_docs", "created_at": "2026-01-01"}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update, \
         patch("adapters.mongodb.insert_one", AsyncMock()) as mock_insert:
        resp = client.post(
            "/corpus/collections",
            json={
                "realm_id": "acme",
                "corpus_id": "external_docs",
                "storage_type": "dense_only",
                "backends": {"qdrant": {"collection": "acme__external_docs", "embedder_id": "bge_m3"}},
            },
        )

    assert resp.status_code == 201
    mock_update.assert_called_once()
    mock_insert.assert_not_called()
    assert resp.json()["id"] == "xyz"


def test_register_collection_defaults_owner_to_external_ingestor(client) -> None:
    """No owner given — defaults to external_ingestor, not "platform" (which
    would misrepresent who actually created this corpus)."""
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        resp = client.post(
            "/corpus/collections",
            json={
                "realm_id": "acme",
                "corpus_id": "external_docs",
                "storage_type": "dense_only",
                "backends": {"qdrant": {"collection": "acme__external_docs"}},
            },
        )

    assert resp.json()["owner"] == "external_ingestor"


@pytest.mark.asyncio
async def test_register_corpus_stores_per_backend_embedder_id():
    """embedder_id lives per-backend, not one flat field on the corpus — a
    corpus can have more than one (e.g. GraphRAG-type: BGE-M3 for Qdrant
    chunks, a separate embedder for Neo4j community embeddings)."""
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)):
        doc = await _register_corpus(
            realm_id="acme", corpus_id="graphrag_docs", storage_type="graphrag",
            backends={
                "qdrant": {"collection": "acme__graphrag_docs", "embedder_id": "bge_m3"},
                "neo4j": {"uri": "bolt://localhost:7476", "embedder_id": "nomic-embed-text"},
            },
        )

    assert doc["backends"]["qdrant"]["embedder_id"] == "bge_m3"
    assert doc["backends"]["neo4j"]["embedder_id"] == "nomic-embed-text"


@pytest.mark.asyncio
async def test_register_corpus_clears_deleted_at_on_reregister():
    """Re-ingesting (or re-registering) a soft-deleted corpus revives it —
    same "create revives a soft-deleted id" convention as realms.py#create_realm."""
    existing = {
        "id": "abc123", "realm_id": "demo", "corpus_id": "handbook",
        "created_at": "2026-01-01", "deleted_at": "2026-02-01T00:00:00",
    }
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)):
        doc = await _register_corpus(
            realm_id="demo", corpus_id="handbook", storage_type="dense_sparse",
            backends={"qdrant": {"collection": "demo__handbook"}},
        )

    assert doc["deleted_at"] is None


# ── GET /corpus/collections — soft-deleted corpora hidden by default ───────────

def test_list_collections_hides_soft_deleted_by_default(client) -> None:
    docs = [
        {"id": "1", "realm_id": "acme", "corpus_id": "active", "deleted_at": None},
        {"id": "2", "realm_id": "acme", "corpus_id": "gone", "deleted_at": "2026-02-01T00:00:00"},
    ]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=docs)):
        resp = client.get("/corpus/collections?realm_id=acme")

    assert [d["corpus_id"] for d in resp.json()] == ["active"]


def test_list_collections_include_deleted_shows_them(client) -> None:
    docs = [
        {"id": "1", "realm_id": "acme", "corpus_id": "active", "deleted_at": None},
        {"id": "2", "realm_id": "acme", "corpus_id": "gone", "deleted_at": "2026-02-01T00:00:00"},
    ]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=docs)):
        resp = client.get("/corpus/collections?realm_id=acme&include_deleted=true")

    assert {d["corpus_id"] for d in resp.json()} == {"active", "gone"}


# ── PUT /corpus/collections/{id} — edit description, corpus_id/realm_id immutable ──

def test_update_collection_edits_description(client) -> None:
    existing = {"id": "abc123", "realm_id": "acme", "corpus_id": "manuals", "description": "old", "deleted_at": None}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        resp = client.put("/corpus/collections/abc123", json={"description": "new description"})

    assert resp.status_code == 200
    assert resp.json()["description"] == "new description"
    assert resp.json()["corpus_id"] == "manuals"
    mock_update.assert_called_once()


def test_update_collection_404_when_not_found(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)):
        resp = client.put("/corpus/collections/nope", json={"description": "x"})
    assert resp.status_code == 404


def test_update_collection_404_when_soft_deleted(client) -> None:
    existing = {"id": "abc123", "deleted_at": "2026-02-01T00:00:00"}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)):
        resp = client.put("/corpus/collections/abc123", json={"description": "x"})
    assert resp.status_code == 404


# ── DELETE /corpus/collections/{id} — soft by default, ?hard=true drops physical data ──

def test_delete_collection_soft_by_default(client) -> None:
    existing = {"id": "abc123", "realm_id": "acme", "corpus_id": "manuals", "deleted_at": None}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update, \
         patch("adapters.mongodb.delete_one", AsyncMock()) as mock_delete:
        resp = client.delete("/corpus/collections/abc123")

    assert resp.status_code == 204
    mock_update.assert_called_once()
    call_args = mock_update.call_args[0]
    assert call_args[0] == "corpora"
    assert call_args[1] == {"id": "abc123"}
    assert call_args[2]["$set"]["deleted_at"] is not None
    mock_delete.assert_not_called()


def test_delete_collection_404_when_already_gone(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)):
        resp = client.delete("/corpus/collections/nope")
    assert resp.status_code == 404


def test_delete_collection_404_when_already_soft_deleted_and_not_hard(client) -> None:
    existing = {"id": "abc123", "deleted_at": "2026-02-01T00:00:00"}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)):
        resp = client.delete("/corpus/collections/abc123")
    assert resp.status_code == 404


def test_delete_collection_hard_drops_qdrant_and_opensearch_then_registry_row(client) -> None:
    existing = {
        "id": "abc123", "realm_id": "acme", "corpus_id": "manuals", "deleted_at": None,
        "backends": {
            "qdrant": {"collection": "acme__manuals__structure_aware__bge_m3"},
            "opensearch": {"index": "acme__manuals__structure_aware"},
        },
    }
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)), \
         patch("adapters.mongodb.delete_one", AsyncMock(return_value=1)) as mock_delete, \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("qdrant_client.QdrantClient") as mock_qdrant_cls, \
         patch("opensearchpy.OpenSearch") as mock_os_cls:
        resp = client.delete("/corpus/collections/abc123?hard=true")

    assert resp.status_code == 204
    mock_qdrant_cls.return_value.delete_collection.assert_called_once_with("acme__manuals__structure_aware__bge_m3")
    mock_os_cls.return_value.indices.delete.assert_called_once_with(index="acme__manuals__structure_aware", ignore=[404])
    mock_delete.assert_called_once_with("corpora", {"id": "abc123"})


def test_delete_collection_hard_still_removes_registry_row_when_backend_drop_fails(client) -> None:
    """A backend that's unreachable or already gone must not block removing
    the registry pointer itself — best-effort, not all-or-nothing."""
    existing = {
        "id": "abc123", "realm_id": "acme", "corpus_id": "manuals", "deleted_at": None,
        "backends": {"qdrant": {"collection": "acme__manuals"}},
    }
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)), \
         patch("adapters.mongodb.delete_one", AsyncMock(return_value=1)) as mock_delete, \
         patch("services.api_gateway.routers.corpus._get_realm_resource", AsyncMock(return_value=None)), \
         patch("qdrant_client.QdrantClient", side_effect=RuntimeError("connection refused")):
        resp = client.delete("/corpus/collections/abc123?hard=true")

    assert resp.status_code == 204
    mock_delete.assert_called_once_with("corpora", {"id": "abc123"})


# ── POST /corpus/collections/{id}/restore ───────────────────────────────────────

def test_restore_collection(client) -> None:
    existing = {"id": "abc123", "realm_id": "acme", "corpus_id": "manuals", "deleted_at": "2026-02-01T00:00:00"}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        resp = client.post("/corpus/collections/abc123/restore")

    assert resp.status_code == 200
    assert resp.json()["deleted_at"] is None
    mock_update.assert_called_once_with("corpora", {"id": "abc123"}, {"$set": {"deleted_at": None}})


def test_restore_collection_404_when_not_soft_deleted(client) -> None:
    existing = {"id": "abc123", "deleted_at": None}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)):
        resp = client.post("/corpus/collections/abc123/restore")
    assert resp.status_code == 404
