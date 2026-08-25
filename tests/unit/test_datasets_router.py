"""GET/POST/DELETE /datasets — the unified dataset registry.

Locks in that platform-authored and RAG-authored datasets are genuinely one
collection now (no more `external_rag_datasets` split) — see
services/api_gateway/routers/datasets.py's module docstring for why.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers.datasets import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def test_routes_exist():
    paths = {r.path for r in router.routes}
    assert "/datasets" in paths
    assert any(p.endswith("/datasets/by-id/{dataset_id}") for p in paths)


def test_get_dataset_includes_id_and_realm_id(client) -> None:
    """Found live while adding per-question CRUD: this response never
    included the dataset's own `id` at all (unlike GET /datasets), so every
    question-write call built from it addressed the dataset as
    `/datasets/undefined/questions` and 404d."""
    doc = {
        "id": "ds1", "filename": "handbook.v1.fast.jsonl", "realm_id": "demo",
        "questions": [{"id": "q1", "question": "q?", "reference_answer": "a"}],
        "n_questions": 1,
    }
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=doc)):
        resp = client.get("/datasets/handbook.v1.fast.jsonl")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "ds1"
    assert body["realm_id"] == "demo"
    assert body["count"] == 1


def test_create_dataset_inserts_and_returns_parsed_metadata(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)) as mock_insert:
        resp = client.post("/datasets", json={
            "filename": "my_rag.v0.fast.jsonl",
            "realm_id": "demo",
            "questions": [{"id": "q1", "question": "..."}],
        })

    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "my_rag"
    assert body["version"] == "v0"
    assert body["speed"] == "fast"
    assert body["source_rag_id"] is None
    mock_insert.assert_called_once()


def test_create_dataset_records_source_rag_id_as_provenance_only(client) -> None:
    """A RAG-authored dataset is stored exactly like a platform one, just
    tagged with who registered it — no separate collection, no UI-visible
    'second class' marker in the stored doc itself."""
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)), \
         patch("adapters.mongodb.insert_one", AsyncMock(return_value=None)) as mock_insert:
        resp = client.post("/datasets", json={
            "filename": "ext_rag.v0.fast.jsonl",
            "realm_id": "demo",
            "questions": [],
            "source_rag_id": "bdc63487",
        })

    assert resp.status_code == 201
    assert resp.json()["source_rag_id"] == "bdc63487"
    inserted_doc = mock_insert.call_args[0][1]
    assert inserted_doc["source_rag_id"] == "bdc63487"


def test_create_dataset_409_on_filename_collision_within_same_realm(client) -> None:
    existing = {"id": "abc", "filename": "dup.v0.fast.jsonl", "realm_id": "demo"}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=existing)):
        resp = client.post("/datasets", json={
            "filename": "dup.v0.fast.jsonl", "realm_id": "demo", "questions": [],
        })
    assert resp.status_code == 409


def test_list_datasets_filters_by_source_rag_id(client) -> None:
    docs = [{"filename": "ext_rag.v0.fast.jsonl", "source_rag_id": "bdc63487", "id": "x1"}]
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=docs)) as mock_find:
        resp = client.get("/datasets?source_rag_id=bdc63487")

    assert resp.status_code == 200
    assert resp.json()[0]["source_rag_id"] == "bdc63487"
    assert mock_find.call_args.kwargs["query"] == {"source_rag_id": "bdc63487"}


def test_list_datasets_combines_realm_and_source_rag_filters(client) -> None:
    with patch("adapters.mongodb.find_many", AsyncMock(return_value=[])) as mock_find:
        client.get("/datasets?realm_id=demo&source_rag_id=bdc63487")

    assert mock_find.call_args.kwargs["query"] == {"realm_id": "demo", "source_rag_id": "bdc63487"}


def test_delete_dataset_by_id(client) -> None:
    with patch("adapters.mongodb.delete_one", AsyncMock(return_value=1)) as mock_delete:
        resp = client.delete("/datasets/by-id/abc123")
    assert resp.status_code == 204
    mock_delete.assert_called_once_with("datasets", {"id": "abc123"})


def test_delete_dataset_404_when_not_found(client) -> None:
    with patch("adapters.mongodb.delete_one", AsyncMock(return_value=0)):
        resp = client.delete("/datasets/by-id/nope")
    assert resp.status_code == 404


def test_delete_dataset_scopes_by_realm_id_when_given(client) -> None:
    """Found live while adding per-question endpoints: this had no
    realm_id check at all — any Realm could delete any other Realm's
    dataset by id."""
    with patch("adapters.mongodb.delete_one", AsyncMock(return_value=1)) as mock_delete:
        resp = client.delete("/datasets/by-id/abc123?realm_id=demo")
    assert resp.status_code == 204
    mock_delete.assert_called_once_with("datasets", {"id": "abc123", "realm_id": "demo"})


# ── Per-question CRUD (question-level write API) ─────────────────────────────

_DATASET = {
    "id": "ds1", "filename": "handbook.v1.fast.jsonl", "realm_id": "demo",
    "questions": [{"id": "q1", "question": "existing?", "reference_answer": "yes",
                   "provenance": {"origin": "manual", "created_at": "2026-01-01T00:00:00+00:00"}}],
    "n_questions": 1,
}


def test_add_question_generates_id_and_manual_provenance(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET, questions=list(_DATASET["questions"])))), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        resp = client.post("/datasets/ds1/questions", json={"question": "new?", "reference_answer": "42"})

    assert resp.status_code == 201
    body = resp.json()
    assert body["count"] == 2
    new_q = body["questions"][-1]
    assert new_q["id"] and new_q["id"] != "q1"
    assert new_q["provenance"]["origin"] == "manual"
    assert new_q["provenance"]["created_at"]
    set_arg = mock_update.call_args[0][2]["$set"]
    assert set_arg["n_questions"] == 2


def test_add_question_trusts_generated_provenance_but_overwrites_created_at(client) -> None:
    draft_provenance = {"origin": "generated", "model": "qwen3:8b", "corpus_id": "handbook",
                         "chunk_id": "c1", "preset_id": "p1", "created_at": "STALE"}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET, questions=[]))), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)):
        resp = client.post("/datasets/ds1/questions", json={
            "question": "q?", "reference_answer": "a", "provenance": draft_provenance,
        })

    prov = resp.json()["questions"][0]["provenance"]
    assert prov["origin"] == "generated"
    assert prov["model"] == "qwen3:8b"
    assert prov["created_at"] != "STALE"


def test_add_question_404_when_dataset_not_found(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=None)):
        resp = client.post("/datasets/nope/questions", json={"question": "q", "reference_answer": "a"})
    assert resp.status_code == 404


def test_add_question_404_on_realm_mismatch(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET))):
        resp = client.post("/datasets/ds1/questions?realm_id=other-realm", json={"question": "q", "reference_answer": "a"})
    assert resp.status_code == 404


def test_add_question_422_on_empty_question(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET))):
        resp = client.post("/datasets/ds1/questions", json={"question": "", "reference_answer": "a"})
    assert resp.status_code == 422


def test_add_questions_batch(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET, questions=[]))), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)):
        resp = client.post("/datasets/ds1/questions/batch", json=[
            {"question": "q1?", "reference_answer": "a1"},
            {"question": "q2?", "reference_answer": "a2"},
        ])
    assert resp.status_code == 201
    body = resp.json()
    assert body["count"] == 2
    ids = {q["id"] for q in body["questions"]}
    assert len(ids) == 2  # distinct generated ids


def test_update_question_replaces_content_and_stamps_updated_at(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET, questions=[dict(_DATASET["questions"][0])]))), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)):
        resp = client.put("/datasets/ds1/questions/q1", json={"question": "edited?", "reference_answer": "edited-answer"})

    assert resp.status_code == 200
    q = resp.json()["questions"][0]
    assert q["id"] == "q1"
    assert q["question"] == "edited?"
    assert q["provenance"]["origin"] == "manual"
    assert q["provenance"]["updated_at"]


def test_update_question_on_generated_question_marks_edited_manually(client) -> None:
    generated_q = {"id": "q1", "question": "gen?", "reference_answer": "gen-a",
                   "provenance": {"origin": "generated", "model": "qwen3:8b", "created_at": "..."}}
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET, questions=[generated_q]))), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)):
        resp = client.put("/datasets/ds1/questions/q1", json={"question": "human-edited?", "reference_answer": "a"})

    prov = resp.json()["questions"][0]["provenance"]
    assert prov["origin"] == "generated"
    assert prov["edited_manually"] is True
    assert prov["model"] == "qwen3:8b"  # original generation context preserved


def test_update_question_404_when_question_id_not_in_dataset(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET))):
        resp = client.put("/datasets/ds1/questions/nope", json={"question": "q", "reference_answer": "a"})
    assert resp.status_code == 404


def test_delete_question_removes_and_recomputes_count(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET, questions=[dict(_DATASET["questions"][0])]))), \
         patch("adapters.mongodb.update_one", AsyncMock(return_value=1)) as mock_update:
        resp = client.delete("/datasets/ds1/questions/q1")

    assert resp.status_code == 200
    assert resp.json()["count"] == 0
    assert mock_update.call_args[0][2]["$set"]["n_questions"] == 0


def test_delete_question_404_when_not_found(client) -> None:
    with patch("adapters.mongodb.find_one", AsyncMock(return_value=dict(_DATASET))):
        resp = client.delete("/datasets/ds1/questions/nope")
    assert resp.status_code == 404
