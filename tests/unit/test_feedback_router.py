"""PUT/GET/DELETE feedback on run answers + the cross-run analysis query.

Locks in: feedback lives in its own `answer_feedback` collection (never
rewrites the run document itself), a PUT merges into any existing doc
instead of replacing it wholesale (setting one field must not clear
another), and realm-scoping 404s the same way datasets.py's per-question
endpoints do.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers.feedback import router

_RUN = {
    "run_id": "run1",
    "realm_id": "demo",
    "question_results": [
        {"question_id": "q1", "question": "Is X true?", "generated_answer": "Yes, per article 5."},
        {"question_id": "q2", "question": "Is Y true?", "generated_answer": "No."},
    ],
}


def _find_one_side_effect(run_doc=None, feedback_doc=None):
    async def _side_effect(collection, query):
        if collection == "experiment_runs":
            return run_doc
        if collection == "answer_feedback":
            return feedback_doc
        return None
    return _side_effect


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def test_routes_exist():
    paths = {r.path for r in router.routes}
    assert "/experiments/{run_id}/questions/{question_id}/feedback" in paths
    assert "/experiments/{run_id}/feedback" in paths
    assert "/feedback" in paths


# ── PUT (upsert) ──────────────────────────────────────────────────────────────

def test_upsert_feedback_creates_new_doc_with_rating(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=_RUN))), \
         patch("adapters.mongodb.upsert_one", AsyncMock(return_value=None)) as mock_upsert:
        resp = client.put("/experiments/run1/questions/q1/feedback", json={"rating": "good"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["rating"] == "good"
    assert body["run_id"] == "run1"
    assert body["question_id"] == "q1"
    assert body["realm_id"] == "demo"
    assert body["id"]
    assert body["created_at"] == body["updated_at"]
    mock_upsert.assert_called_once()
    assert mock_upsert.call_args[0][0] == "answer_feedback"
    assert mock_upsert.call_args[0][1] == {"run_id": "run1", "question_id": "q1"}


def test_upsert_feedback_merges_without_clearing_existing_fields(client) -> None:
    """The most important behavior in this router — a partial PUT must not
    wipe fields it didn't touch."""
    existing = {
        "id": "fb1", "run_id": "run1", "question_id": "q1", "realm_id": "demo",
        "rating": "good", "scores": {"accuracy": 4.0}, "comment": "looks fine",
        "reviewer": "alice", "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=_RUN, feedback_doc=existing))), \
         patch("adapters.mongodb.upsert_one", AsyncMock(return_value=None)):
        resp = client.put("/experiments/run1/questions/q1/feedback", json={"scores": {"completeness": 3.0}})

    body = resp.json()
    assert body["scores"] == {"accuracy": 4.0, "completeness": 3.0}
    assert body["comment"] == "looks fine"
    assert body["reviewer"] == "alice"
    assert body["rating"] == "good"
    assert body["created_at"] == "2026-01-01T00:00:00+00:00"
    assert body["updated_at"] != body["created_at"]


def test_upsert_feedback_strips_stringified_id_before_writing_back(client) -> None:
    """Found live: adapters.mongodb.find_one stringifies `_id` for JSON
    safety, so `existing` always carries a str `_id` when a doc already
    exists. Copying that straight into the replace_one payload makes Mongo
    reject the write as an attempted change to the immutable `_id` field —
    every *update* to an existing feedback doc 500'd, while the first-ever
    create for a question (no `existing`, so no `_id` to copy) kept
    working. This only reproduces against real Mongo (mocked upsert_one
    doesn't enforce _id immutability), so the regression guard here is that
    the payload handed to upsert_one never contains `_id`."""
    existing = {
        "_id": "6a56565d078bbaa8a87ba3e3", "id": "fb1", "run_id": "run1", "question_id": "q1",
        "realm_id": "demo", "rating": "good", "scores": {}, "comment": None, "reviewer": None,
        "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
    }
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=_RUN, feedback_doc=existing))), \
         patch("adapters.mongodb.upsert_one", AsyncMock(return_value=None)) as mock_upsert:
        resp = client.put("/experiments/run1/questions/q1/feedback", json={"scores": {"completeness": 3.0}})

    assert resp.status_code == 200
    written_doc = mock_upsert.call_args[0][2]
    assert "_id" not in written_doc


def test_upsert_feedback_404_when_run_not_found(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=None))):
        resp = client.put("/experiments/nope/questions/q1/feedback", json={"rating": "good"})
    assert resp.status_code == 404


def test_upsert_feedback_404_on_realm_mismatch(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=_RUN))):
        resp = client.put("/experiments/run1/questions/q1/feedback?realm_id=other-realm", json={"rating": "good"})
    assert resp.status_code == 404


def test_upsert_feedback_404_when_question_not_in_run(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=_RUN))):
        resp = client.put("/experiments/run1/questions/nope/feedback", json={"rating": "good"})
    assert resp.status_code == 404


def test_upsert_feedback_422_on_invalid_rating(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=_RUN))):
        resp = client.put("/experiments/run1/questions/q1/feedback", json={"rating": "meh"})
    assert resp.status_code == 422


# ── GET per-run ───────────────────────────────────────────────────────────────

def test_get_run_feedback_returns_map_keyed_by_question_id(client) -> None:
    docs = [
        {"id": "fb1", "run_id": "run1", "question_id": "q1", "rating": "good"},
        {"id": "fb2", "run_id": "run1", "question_id": "q2", "rating": "bad"},
    ]
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=_RUN))), \
         patch("adapters.mongodb.find_many", AsyncMock(return_value=docs)):
        resp = client.get("/experiments/run1/feedback")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"q1", "q2"}
    assert body["q1"]["rating"] == "good"


def test_get_run_feedback_404_when_run_not_found(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=None))):
        resp = client.get("/experiments/nope/feedback")
    assert resp.status_code == 404


# ── DELETE ────────────────────────────────────────────────────────────────────

def test_delete_feedback_success(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=_RUN))), \
         patch("adapters.mongodb.delete_one", AsyncMock(return_value=1)) as mock_delete:
        resp = client.delete("/experiments/run1/questions/q1/feedback")

    assert resp.status_code == 204
    mock_delete.assert_called_once_with("answer_feedback", {"run_id": "run1", "question_id": "q1"})


def test_delete_feedback_404_when_not_found(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=_RUN))), \
         patch("adapters.mongodb.delete_one", AsyncMock(return_value=0)):
        resp = client.delete("/experiments/run1/questions/q1/feedback")
    assert resp.status_code == 404


# ── Cross-run analysis query ──────────────────────────────────────────────────

def test_list_feedback_applies_filters_to_query(client) -> None:
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=[])) as mock_find:
        client.get("/feedback?realm_id=demo&rating=bad&limit=50")

    assert mock_find.call_args.kwargs["query"] == {"realm_id": "demo", "rating": "bad"}
    assert mock_find.call_args.kwargs["limit"] == 50


def test_list_feedback_enriches_with_question_and_answer_text(client) -> None:
    docs = [{"id": "fb1", "run_id": "run1", "question_id": "q1", "rating": "bad", "realm_id": "demo"}]

    async def find_one_side_effect(collection, query):
        assert collection == "experiment_runs"
        return _RUN

    with patch("adapters.mongodb.find_many", AsyncMock(return_value=docs)), \
         patch("adapters.mongodb.find_one", AsyncMock(side_effect=find_one_side_effect)):
        resp = client.get("/feedback")

    assert resp.status_code == 200
    body = resp.json()[0]
    assert body["question"] == "Is X true?"
    assert body["generated_answer"] == "Yes, per article 5."
    assert body["run_config_name"] is None


def test_list_feedback_degrades_honestly_when_parent_run_missing(client) -> None:
    """A referenced run was deleted — feedback still returns, just without
    the enriched text, rather than erroring or fabricating it."""
    docs = [{"id": "fb1", "run_id": "gone", "question_id": "q1", "rating": "bad"}]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=docs)), \
         patch("adapters.mongodb.find_one", AsyncMock(return_value=None)):
        resp = client.get("/feedback")

    body = resp.json()[0]
    assert body["question"] is None
    assert body["generated_answer"] is None
