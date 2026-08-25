"""POST /experiments/{run_id}/questions/{question_id}/feedback/promote —
turning a reviewed question into a golden-dataset entry.
Calls `datasets.add_question` directly (not an HTTP self-call), so mocking
`adapters.mongodb` is enough — no second router needs to be mounted.
"""
from __future__ import annotations

import copy
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers.feedback import router

_RUN = {
    "run_id": "run1",
    "realm_id": "demo",
    "dataset_name": "handbook.v1.fast.jsonl",
    "question_results": [
        {
            "question_id": "q1", "question": "Is X true?",
            "reference_answer": "Yes, per article 5.", "generated_answer": "Yes, per article 5.",
        },
    ],
}

_SOURCE_DATASET_DOC = {
    "id": "ds-source", "filename": "handbook.v1.fast.jsonl", "realm_id": "demo",
    "questions": [
        {"id": "q1", "question": "Is X true?", "reference_answer": "Yes.", "article_refs": ["art-5"]},
    ],
    "n_questions": 1,
}

_TARGET_DATASET_DOC = {
    "id": "ds-target", "filename": "regression.v0.fast.jsonl", "realm_id": "demo",
    "questions": [], "n_questions": 0,
}

_FEEDBACK_DOC = {"id": "fb1", "run_id": "run1", "question_id": "q1", "rating": "bad"}


def _find_one_side_effect(*, run_doc=_RUN, feedback_doc=_FEEDBACK_DOC, dataset_docs=None):
    # add_question (datasets.py) mutates the dict find_one returns (sets
    # doc["questions"]/doc["n_questions"] in place, mirroring how a real
    # Mongo doc gets fetched fresh per request) — deep-copy the module-level
    # fixtures here so one test's mutation can't leak into the next.
    dataset_docs = copy.deepcopy(dataset_docs) if dataset_docs is not None else {
        "handbook.v1.fast.jsonl": copy.deepcopy(_SOURCE_DATASET_DOC),
        "ds-target": copy.deepcopy(_TARGET_DATASET_DOC),
    }

    async def _side_effect(collection, query):
        if collection == "experiment_runs":
            return run_doc
        if collection == "answer_feedback":
            return feedback_doc
        if collection == "datasets":
            # _load_dataset queries by filename; _get_dataset_or_404 (inside
            # add_question) queries by id — look up by whichever key is present.
            return dataset_docs.get(query.get("filename")) or dataset_docs.get(query.get("id"))
        return None
    return _side_effect


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def test_promote_creates_question_with_reviewer_feedback_provenance(client: TestClient) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect())), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        resp = client.post(
            "/experiments/run1/questions/q1/feedback/promote",
            json={"target_dataset_id": "ds-target"},
        )

    assert resp.status_code == 201
    assert resp.json()["created"] is True
    mock_update.assert_called_once()
    written_questions = mock_update.call_args[0][2]["$set"]["questions"]
    assert len(written_questions) == 1
    new_question = written_questions[0]
    assert new_question["question"] == "Is X true?"
    assert new_question["reference_answer"] == "Yes, per article 5."
    assert new_question["provenance"]["origin"] == "reviewer_feedback"
    assert new_question["provenance"]["source_run_id"] == "run1"
    assert new_question["provenance"]["source_question_id"] == "q1"
    assert new_question["provenance"]["source_feedback_id"] == "fb1"


def test_promote_allows_overriding_reference_answer(client: TestClient) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect())), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        client.post(
            "/experiments/run1/questions/q1/feedback/promote",
            json={"target_dataset_id": "ds-target", "reference_answer": "Corrected answer."},
        )

    new_question = mock_update.call_args[0][2]["$set"]["questions"][0]
    assert new_question["reference_answer"] == "Corrected answer."


def test_promote_updates_existing_question_instead_of_duplicating(client: TestClient) -> None:
    """Found live: promoting a question already present in the target
    dataset (e.g. it was auto-generated earlier) silently appended a
    duplicate — the reviewer confirms/corrects the same question, it must
    not become two rows."""
    target_with_existing = {
        "id": "ds-target", "filename": "regression.v0.fast.jsonl", "realm_id": "demo",
        "questions": [{
            "id": "existing-1", "question": "Is X true?", "reference_answer": "Old answer.",
            "article_refs": [], "provenance": {"origin": "generated", "model": "qwen3:8b", "created_at": "2026-01-01T00:00:00+00:00"},
        }],
        "n_questions": 1,
    }
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(
        dataset_docs={"handbook.v1.fast.jsonl": _SOURCE_DATASET_DOC, "ds-target": target_with_existing},
    ))), patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        resp = client.post(
            "/experiments/run1/questions/q1/feedback/promote",
            json={"target_dataset_id": "ds-target"},
        )

    assert resp.status_code == 201
    assert resp.json()["created"] is False
    mock_update.assert_called_once()
    written_questions = mock_update.call_args[0][2]["$set"]["questions"]
    assert len(written_questions) == 1  # updated in place, not appended
    updated = written_questions[0]
    assert updated["id"] == "existing-1"
    assert updated["reference_answer"] == "Yes, per article 5."
    assert updated["provenance"]["origin"] == "reviewer_feedback"
    assert updated["provenance"]["previous_origin"] == "generated"
    assert updated["provenance"]["created_at"] == "2026-01-01T00:00:00+00:00"
    assert updated["provenance"]["source_run_id"] == "run1"


def test_promote_autofills_article_refs_from_source_dataset(client: TestClient) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect())), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        client.post("/experiments/run1/questions/q1/feedback/promote", json={"target_dataset_id": "ds-target"})

    new_question = mock_update.call_args[0][2]["$set"]["questions"][0]
    assert new_question["article_refs"] == ["art-5"]


def test_promote_allows_overriding_article_refs(client: TestClient) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect())), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        client.post(
            "/experiments/run1/questions/q1/feedback/promote",
            json={"target_dataset_id": "ds-target", "article_refs": ["art-99"]},
        )

    new_question = mock_update.call_args[0][2]["$set"]["questions"][0]
    assert new_question["article_refs"] == ["art-99"]


def test_promote_works_without_existing_feedback_doc(client: TestClient) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(feedback_doc=None))), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        resp = client.post(
            "/experiments/run1/questions/q1/feedback/promote",
            json={"target_dataset_id": "ds-target"},
        )

    assert resp.status_code == 201
    new_question = mock_update.call_args[0][2]["$set"]["questions"][0]
    assert new_question["provenance"]["source_feedback_id"] is None


def test_promote_404_when_run_not_found(client: TestClient) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(run_doc=None))):
        resp = client.post("/experiments/nope/questions/q1/feedback/promote", json={"target_dataset_id": "ds-target"})
    assert resp.status_code == 404


def test_promote_404_when_question_not_in_run(client: TestClient) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect())):
        resp = client.post("/experiments/run1/questions/nope/feedback/promote", json={"target_dataset_id": "ds-target"})
    assert resp.status_code == 404


def test_promote_404_when_target_dataset_not_found(client: TestClient) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=_find_one_side_effect(dataset_docs={"handbook.v1.fast.jsonl": _SOURCE_DATASET_DOC}))):
        resp = client.post("/experiments/run1/questions/q1/feedback/promote", json={"target_dataset_id": "nope"})
    assert resp.status_code == 404
