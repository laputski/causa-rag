"""create_prompt version numbering — regression guard.

Found live: creating a Realm's first-ever prompt landed on `prompt_v5`
because create_prompt computed next_version from prompt_store.list(), a
shared file store with no realm_id concept at all, so it counted the four
prompts belonging to a different Realm too. Version must be scoped to the requesting
Realm (via Mongo, where realm_id actually lives), while the generated id
still needs to stay globally unique in the shared file store.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def prompt_store_dir(tmp_path, monkeypatch):
    import services.api_gateway.routers.prompts as prompts_module
    from core.prompt_store import PromptStore

    store = PromptStore(prompts_dir=tmp_path)
    monkeypatch.setattr(prompts_module, "prompt_store", store)
    return store


def _make_fake_find_many(docs: list[dict]):
    async def _fake(collection, query=None, sort=None, limit=0):
        query = query or {}
        return [d for d in docs if all(d.get(k) == v for k, v in query.items())]
    return _fake


async def _fake_upsert_noop(collection, query, doc):
    return None


def test_first_prompt_in_new_realm_gets_version_1_even_if_another_realm_has_prompts(prompt_store_dir):
    """A new realm's first prompt starts at version 1, whatever another realm's
    prompt count happens to be."""
    from services.api_gateway.routers.prompts import router

    mongo_docs = [
        {"id": "handbook_ru_v1", "version": 1, "realm_id": "demo"},
        {"id": "handbook_ru_v2", "version": 2, "realm_id": "demo"},
        {"id": "handbook_ru_v3", "version": 3, "realm_id": "demo"},
        {"id": "demo_prompt_v1", "version": 4, "realm_id": "demo"},
    ]
    # The other realm's prompts also exist as files in the shared store: id
    # uniqueness is checked against the file store, and not against Mongo.
    for d in mongo_docs:
        prompt_store_dir.save({**d, "name": "n", "template": "t {context} {query}"})

    app = FastAPI()
    app.include_router(router)
    with patch("adapters.mongodb.find_many", side_effect=_make_fake_find_many(mongo_docs)), \
         patch("adapters.mongodb.upsert_one", side_effect=_fake_upsert_noop):
        with TestClient(app) as client:
            resp = client.post("/prompts", json={
                "name": "Acme prompt", "template": "t {context} {query}", "realm_id": "acme",
            })

    assert resp.status_code == 201
    body = resp.json()
    assert body["version"] == 1, f"this Realm's first prompt should be version 1, got {body['version']}"
    assert body["id"] == "prompt_v1", f"expected prompt_v1, got {body['id']!r}"


def test_second_prompt_in_same_realm_gets_version_2(prompt_store_dir):
    from services.api_gateway.routers.prompts import router

    mongo_docs = [{"id": "prompt_v1", "version": 1, "realm_id": "acme"}]
    prompt_store_dir.save({**mongo_docs[0], "name": "n", "template": "t {context} {query}"})

    app = FastAPI()
    app.include_router(router)
    with patch("adapters.mongodb.find_many", side_effect=_make_fake_find_many(mongo_docs)), \
         patch("adapters.mongodb.upsert_one", side_effect=_fake_upsert_noop):
        with TestClient(app) as client:
            resp = client.post("/prompts", json={
                "name": "Acme prompt 2", "template": "t {context} {query}", "realm_id": "acme",
            })

    assert resp.status_code == 201
    assert resp.json()["version"] == 2


def test_id_collision_with_another_realms_file_gets_disambiguated(prompt_store_dir):
    """Two Realms' first prompt both compute version 1 — the id must not
    collide in the shared file store even though versions legitimately do."""
    from services.api_gateway.routers.prompts import router

    # The other Realm already has a "prompt_v1" file on disk (its own first auto-named
    # prompt), but Mongo has zero prompts for realm_id=realm_b.
    prompt_store_dir.save({
        "id": "prompt_v1", "name": "handbook prompt", "version": 1, "template": "t {context} {query}",
    })
    mongo_docs = [{"id": "prompt_v1", "version": 1, "realm_id": "demo"}]

    app = FastAPI()
    app.include_router(router)
    with patch("adapters.mongodb.find_many", side_effect=_make_fake_find_many(mongo_docs)), \
         patch("adapters.mongodb.upsert_one", side_effect=_fake_upsert_noop):
        with TestClient(app) as client:
            resp = client.post("/prompts", json={
                "name": "Acme prompt", "template": "t {context} {query}", "realm_id": "acme",
            })

    assert resp.status_code == 201
    body = resp.json()
    assert body["version"] == 1
    assert body["id"] != "prompt_v1", "id must not collide with the existing prompt_v1 file"
    assert body["id"] == "prompt_v1-1"


def test_no_realm_id_falls_back_to_global_file_count(prompt_store_dir):
    """Back-compat: an old client that never sends realm_id keeps the
    original global-count behavior (no Realm context to scope by)."""
    from services.api_gateway.routers.prompts import router

    prompt_store_dir.save({"id": "a", "name": "n", "version": 1, "template": "t"})
    prompt_store_dir.save({"id": "b", "name": "n", "version": 2, "template": "t"})

    app = FastAPI()
    app.include_router(router)
    with patch("adapters.mongodb.find_many", side_effect=_make_fake_find_many([])), \
         patch("adapters.mongodb.upsert_one", side_effect=_fake_upsert_noop):
        with TestClient(app) as client:
            resp = client.post("/prompts", json={
                "name": "no realm prompt", "template": "t {context} {query}",
            })

    assert resp.status_code == 201
    assert resp.json()["version"] == 3


def test_mongo_unavailable_falls_back_to_global_file_count(prompt_store_dir):
    """If Mongo can't be read, degrade to the old global-file-count behavior
    rather than crashing the create request."""
    from services.api_gateway.routers.prompts import router

    prompt_store_dir.save({"id": "handbook_ru_v1", "name": "n", "version": 1, "template": "t"})

    async def _boom(collection, query=None, sort=None, limit=0):
        raise ConnectionError("mongo down")

    app = FastAPI()
    app.include_router(router)
    with patch("adapters.mongodb.find_many", side_effect=_boom), \
         patch("adapters.mongodb.upsert_one", side_effect=_fake_upsert_noop):
        with TestClient(app) as client:
            resp = client.post("/prompts", json={
                "name": "fallback prompt", "template": "t {context} {query}", "realm_id": "acme",
            })

    assert resp.status_code == 201
    assert resp.json()["version"] == 2
