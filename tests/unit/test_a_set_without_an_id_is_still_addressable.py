"""A set that arrived without an id can still be read, changed and profiled.

Found live, opening the control questions of a Realm that was imported: the
line with the set's composition did not appear, and neither did any error.
A set written by an import carries no `id` (only the route that creates a
set mints one), the detail view returned `id: null`, and every call that
addresses a set by its id went to `/datasets/null/...` and got a 404. So a
question of such a set could not be edited, added or removed either, and the
form closed as if it had been.

Such a set is addressed by its file name within its Realm, which is what
makes it unique where it is stored. A set that has an id is addressed by the
id alone, as before.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers import datasets as D
from tests.unit._fake_mongo import FakeMongo

_IMPORTED = {"filename": "notes.v1.fast.jsonl", "realm_id": "acme", "questions": [
    {"id": "q1", "question": "How often?", "ground_truth": "Monthly.", "article_refs": ["A/1"]}]}
_CREATED = {"id": "abc12345", "filename": "made.v1.fast.jsonl", "realm_id": "acme", "n_questions": 0, "questions": []}
_SAME_NAME_ELSEWHERE = {"filename": "notes.v1.fast.jsonl", "realm_id": "other", "questions": [
    {"id": "q9", "question": "Another Realm's question?", "ground_truth": "Yes.", "article_refs": []}]}


@pytest.fixture
def served(monkeypatch) -> tuple[TestClient, FakeMongo]:
    store = FakeMongo({"datasets": [_SAME_NAME_ELSEWHERE, _IMPORTED, _CREATED]})
    store.install(monkeypatch)
    app = FastAPI()
    app.include_router(D.router)
    return TestClient(app), store


def _stored(store: FakeMongo, realm: str) -> dict:
    return next(d for d in store.data["datasets"] if d["filename"] == "notes.v1.fast.jsonl" and d["realm_id"] == realm)


def test_its_composition_can_be_asked_for_by_file_name_within_its_realm(served) -> None:
    client, _ = served
    reply = client.get("/datasets/notes.v1.fast.jsonl/profile", params={"realm_id": "acme"})
    assert reply.status_code == 200
    assert reply.json()["questions"] == 1


def test_a_question_of_it_can_be_changed_and_the_change_is_stored(served) -> None:
    client, store = served
    reply = client.put("/datasets/notes.v1.fast.jsonl/questions/q1", params={"realm_id": "acme"},
                       json={"question": "How often is it done?", "reference_answer": "Monthly."})
    assert reply.status_code == 200
    assert _stored(store, "acme")["questions"][0]["question"] == "How often is it done?"


def test_a_question_can_be_added_to_it_and_removed_from_it(served) -> None:
    client, store = served
    added = client.post("/datasets/notes.v1.fast.jsonl/questions", params={"realm_id": "acme"},
                        json={"question": "Who?", "reference_answer": "The owner."})
    assert added.status_code == 201
    assert len(_stored(store, "acme")["questions"]) == 2
    removed = client.delete("/datasets/notes.v1.fast.jsonl/questions/q1", params={"realm_id": "acme"})
    assert removed.status_code == 200
    assert [q["question"] for q in _stored(store, "acme")["questions"]] == ["Who?"]


def test_the_set_of_the_same_name_in_another_realm_is_left_alone(served) -> None:
    client, store = served
    client.put("/datasets/notes.v1.fast.jsonl/questions/q1", params={"realm_id": "acme"},
               json={"question": "Changed?", "reference_answer": "Monthly."})
    client.post("/datasets/notes.v1.fast.jsonl/questions", params={"realm_id": "acme"},
                json={"question": "Who?", "reference_answer": "The owner."})
    assert _stored(store, "other") == _SAME_NAME_ELSEWHERE


def test_a_file_name_without_a_realm_addresses_nothing(served) -> None:
    """Two Realms may hold a set of one name, so the name alone would pick
    whichever the store returned first."""
    client, _ = served
    assert client.get("/datasets/notes.v1.fast.jsonl/profile").status_code == 404


def test_a_set_that_has_an_id_is_addressed_by_it_as_before(served) -> None:
    client, store = served
    added = client.post("/datasets/abc12345/questions", params={"realm_id": "acme"},
                        json={"question": "Who?", "reference_answer": "The owner."})
    assert added.status_code == 201
    made = next(d for d in store.data["datasets"] if d.get("id") == "abc12345")
    assert len(made["questions"]) == 1 and made["n_questions"] == 1
    assert client.get("/datasets/abc12345/profile", params={"realm_id": "other"}).status_code == 404


def test_it_can_be_deleted_and_only_it_is(served) -> None:
    client, store = served
    assert client.delete("/datasets/by-id/notes.v1.fast.jsonl", params={"realm_id": "acme"}).status_code == 204
    left = [(d["filename"], d["realm_id"]) for d in store.data["datasets"]]
    assert left == [("notes.v1.fast.jsonl", "other"), ("made.v1.fast.jsonl", "acme")]


def test_a_set_with_an_id_is_deleted_by_it_and_another_realm_cannot(served) -> None:
    client, store = served
    assert client.delete("/datasets/by-id/abc12345", params={"realm_id": "other"}).status_code == 404
    assert client.delete("/datasets/by-id/abc12345", params={"realm_id": "acme"}).status_code == 204
    assert all(d.get("id") != "abc12345" for d in store.data["datasets"])
