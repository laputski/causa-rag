"""Realm isolation — regression guard.

Every realm-scoped collection (external_rags, experiment_runs, settings,
corpus_ingests, datasets, prompts) has been fixed for a Realm-leak bug at
least once, each time discovered live rather than caught by a test (see
the design notes). This file is the test net that should have caught them:
for each collection, a doc with no `realm_id` is invisible under any
Realm filter, a filtered request only returns that Realm's own docs, and an
unfiltered request still returns everything (back-compat).

Mongo itself isn't running for these — `adapters.mongodb.find_many`/
`find_one` are patched with an in-memory fake that replicates Mongo's exact-
match query semantics (a missing field never equals a queried value), so the
same bug (an endpoint silently ignoring `realm_id`) would fail here exactly
as it did live.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _matches(doc: dict[str, Any], query: dict[str, Any] | None) -> bool:
    query = query or {}
    return all(doc.get(k) == v for k, v in query.items())


def _make_fake_find_many(fixtures: dict[str, list[dict[str, Any]]]):
    async def _fake_find_many(collection, query=None, sort=None, limit=0):
        docs = fixtures.get(collection, [])
        return [d for d in docs if _matches(d, query)]
    return _fake_find_many


# ── Shared behavior across the four "clean Mongo-filter" list endpoints ──────
# datasets/prompts/corpus_ingests/external_rags all build
# `{"realm_id": realm_id} if realm_id else {}` (or None) and pass it straight
# to find_many — same shape, same bug class, one parametrized test.

def _datasets_case():
    from services.api_gateway.routers.datasets import router
    fixtures = {
        "datasets": [
            {"filename": "handbook.v1.jsonl", "realm_id": "demo"},
            {"filename": "manuals.v1.jsonl", "realm_id": "acme"},
            {"filename": "legacy.v0.jsonl"},  # pre-scoping, no realm_id at all
        ]
    }
    return dict(
        router=router, fixtures=fixtures, collection="datasets", path="/datasets",
        id_field="filename", realm_a_id="handbook.v1.jsonl", realm_b_id="manuals.v1.jsonl",
        legacy_id="legacy.v0.jsonl",
    )


def _prompts_case():
    from services.api_gateway.routers.prompts import router
    fixtures = {
        "prompts": [
            {"id": "handbook_ru_v1", "name": "n", "version": 1, "template": "t", "realm_id": "demo"},
            {"id": "manuals_v1", "name": "n", "version": 1, "template": "t", "realm_id": "acme"},
            {"id": "legacy_v0", "name": "n", "version": 1, "template": "t"},
        ]
    }
    return dict(
        router=router, fixtures=fixtures, collection="prompts", path="/prompts",
        id_field="id", realm_a_id="handbook_ru_v1", realm_b_id="manuals_v1", legacy_id="legacy_v0",
    )


def _corpus_ingests_case():
    from services.api_gateway.routers.corpus import router
    fixtures = {
        "corpus_ingests": [
            {"job_id": "j1", "corpus_id": "handbook", "realm_id": "demo"},
            {"job_id": "j2", "corpus_id": "acme-corpus", "realm_id": "acme"},
            {"job_id": "j3", "corpus_id": "orphan"},
        ]
    }
    return dict(
        router=router, fixtures=fixtures, collection="corpus_ingests", path="/corpus",
        id_field="job_id", realm_a_id="j1", realm_b_id="j2", legacy_id="j3",
    )


def _external_rags_case():
    from services.api_gateway.routers.external_rags import router
    fixtures = {
        "external_rags": [
            {"id": "r1", "name": "acme-rag", "url": "http://x", "headers": {}, "realm_id": "demo"},
            {"id": "r2", "name": "acme-rag", "url": "http://y", "headers": {}, "realm_id": "acme"},
            {"id": "r3", "name": "orphan-rag", "url": "http://z", "headers": {}},
        ]
    }
    return dict(
        router=router, fixtures=fixtures, collection="external_rags", path="/external-rags",
        id_field="id", realm_a_id="r1", realm_b_id="r2", legacy_id="r3",
    )


_CASES = [_datasets_case, _prompts_case, _corpus_ingests_case, _external_rags_case]


@pytest.mark.parametrize("case_factory", _CASES, ids=[c.__name__ for c in _CASES])
def test_realm_filter_returns_only_that_realms_docs(case_factory):
    case = case_factory()
    app = FastAPI()
    app.include_router(case["router"])
    with patch("adapters.mongodb.find_many", side_effect=_make_fake_find_many(case["fixtures"])):
        with TestClient(app) as client:
            resp = client.get(f"{case['path']}?realm_id=demo")
    assert resp.status_code == 200
    ids = {d[case["id_field"]] for d in resp.json()}
    assert ids == {case["realm_a_id"]}, (
        f"{case['collection']}: expected only {case['realm_a_id']!r} under realm_id=demo, got {ids}"
    )


@pytest.mark.parametrize("case_factory", _CASES, ids=[c.__name__ for c in _CASES])
def test_realm_filter_hides_docs_from_other_realms(case_factory):
    case = case_factory()
    app = FastAPI()
    app.include_router(case["router"])
    with patch("adapters.mongodb.find_many", side_effect=_make_fake_find_many(case["fixtures"])):
        with TestClient(app) as client:
            resp = client.get(f"{case['path']}?realm_id=realm_b")
    ids = {d[case["id_field"]] for d in resp.json()}
    assert case["realm_a_id"] not in ids, f"{case['collection']}: realm A's doc leaked into realm B's list"
    assert case["legacy_id"] not in ids, f"{case['collection']}: unscoped doc leaked into the Realm's list"


@pytest.mark.parametrize("case_factory", _CASES, ids=[c.__name__ for c in _CASES])
def test_realm_filter_hides_unscoped_docs_under_any_realm(case_factory):
    """A doc with no realm_id (pre-scoping) must not silently appear for
    every Realm — same convention as the corpus_ingests/experiment_runs
    backfill precedent (see the design notes): invisible, not global-by-default.
    """
    case = case_factory()
    app = FastAPI()
    app.include_router(case["router"])
    with patch("adapters.mongodb.find_many", side_effect=_make_fake_find_many(case["fixtures"])):
        with TestClient(app) as client:
            resp = client.get(f"{case['path']}?realm_id=demo")
    ids = {d[case["id_field"]] for d in resp.json()}
    assert case["legacy_id"] not in ids


@pytest.mark.parametrize("case_factory", _CASES, ids=[c.__name__ for c in _CASES])
def test_no_realm_filter_returns_everything(case_factory):
    """Back-compat: a caller that doesn't pass realm_id (pre-Realm client,
    or a context with no active Realm) still sees the full unscoped list."""
    case = case_factory()
    app = FastAPI()
    app.include_router(case["router"])
    with patch("adapters.mongodb.find_many", side_effect=_make_fake_find_many(case["fixtures"])):
        with TestClient(app) as client:
            resp = client.get(case["path"])
    ids = {d[case["id_field"]] for d in resp.json()}
    assert ids == {case["realm_a_id"], case["realm_b_id"], case["legacy_id"]}


# ── settings: find_one keyed by _id, not find_many + filter ──────────────────

async def test_settings_doc_is_keyed_by_realm_id_not_shared():
    from services.api_gateway.routers.settings import _get_settings_doc

    docs = {
        "demo": {"_id": "demo", "active_model": "qwen-a"},
        "acme": {"_id": "acme", "active_model": "qwen-b"},
    }

    async def fake_find_one(collection, query):
        return docs.get(query.get("_id"))

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one):
        realm_a_settings = await _get_settings_doc("demo")
        realm_b_settings = await _get_settings_doc("acme")

    assert realm_a_settings["active_model"] == "qwen-a"
    assert realm_b_settings["active_model"] == "qwen-b"


async def test_settings_unknown_realm_gets_defaults_not_someone_elses_doc():
    from services.api_gateway.routers.settings import _DEFAULT_SETTINGS, _get_settings_doc

    async def fake_find_one(collection, query):
        return {"_id": "demo", "active_model": "qwen-a"} if query.get("_id") == "demo" else None

    with patch("adapters.mongodb.find_one", side_effect=fake_find_one):
        result = await _get_settings_doc("brand-new-realm")

    assert result["active_model"] == _DEFAULT_SETTINGS["active_model"]


# ── experiment_runs: in-memory filter over an unconditional find_many ────────

def test_experiment_list_filters_by_realm_id():
    from services.api_gateway.routers import experiments as experiments_module

    docs = [
        {"run_id": "run-realm_a", "config_name": "run-realm_a", "realm_id": "demo"},
        {"run_id": "run-b", "config_name": "run-b", "realm_id": "acme"},
        {"run_id": "run-legacy", "config_name": "run-legacy"},  # pre-Realm run
    ]

    async def fake_find_many(collection, query=None, sort=None, limit=0):
        return list(docs)  # _get_results() applies no server-side filter

    app = FastAPI()
    app.include_router(experiments_module.router)
    with patch("adapters.mongodb.find_many", side_effect=fake_find_many):
        with TestClient(app) as client:
            resp_a = client.get("/experiments?realm_id=demo")
            resp_none = client.get("/experiments")

    ids_a = {r["run_id"] for r in resp_a.json()}
    assert ids_a == {"run-realm_a"}, "another Realm's run, or the unscoped legacy one, leaked into this Realm's list"

    all_ids = {r["run_id"] for r in resp_none.json()}
    assert all_ids == {"run-realm_a", "run-b", "run-legacy"}
