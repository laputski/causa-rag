"""Deleting a run deletes the windows it moved out of its own document.

Found by reading, while planning the clean-up of old runs' windows:
`delete_experiment` removed the run document and its file and left every
window document behind, one per question, for good. No run pointed at them
and nothing ever read them again. None existed yet only because no run had
been deleted since the windows moved out.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers import experiments as E
from tests.unit._fake_mongo import FakeMongo


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "_STORE_DIR", tmp_path)
    app = FastAPI()
    app.include_router(E.router)
    with TestClient(app) as c:
        yield c


def _store() -> FakeMongo:
    return FakeMongo({
        "experiment_runs": [{"run_id": "gone"}, {"run_id": "kept"}],
        E._WINDOWS_COLLECTION: [
            {"run_id": "gone", "question_id": "q1", "position": 0, "candidate_source_refs": [{}]},
            {"run_id": "gone", "question_id": "q2", "position": 1, "candidate_source_refs": [{}]},
            {"run_id": "kept", "question_id": "q1", "position": 0, "candidate_source_refs": [{}]},
        ],
    })


def test_the_windows_of_a_deleted_run_go_with_it(client, monkeypatch) -> None:
    store = _store().install(monkeypatch)
    assert client.delete("/experiments/gone").status_code == 204
    left = {w["run_id"] for w in store.data[E._WINDOWS_COLLECTION]}
    assert left == {"kept"}


def test_another_run_s_windows_are_not_touched(client, monkeypatch) -> None:
    store = _store().install(monkeypatch)
    client.delete("/experiments/gone")
    assert [w["question_id"] for w in store.data[E._WINDOWS_COLLECTION]] == ["q1"]
