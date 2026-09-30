"""A run whose windows were cleared says what it said while it had them.

Clearing removes a run's retrieval windows: the candidate window, the list
before the reranker and each half's own list. Everything the run page
concluded from them has to survive, and so does the absence of anything they
used to rule out. Two things go wrong otherwise, and both look like findings
about the system under test, although the store caused them: a check that needed
a window falls silent, and a check the window used to hold back speaks up.

The run below is built so that each of those can happen: its lexical half
answered and its dense half did not, so a half is named as empty and the
check for a random-vector embedder, which reads the merged scores, stands
aside only because the halves say why those scores are zero.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api_gateway.routers import experiments as E
from tests.unit._fake_mongo import FakeMongo


def _ref(n: int, doc: str, **extra) -> dict:
    return {"chunk_id": f"{doc}-{n}", "chunk_text": f"text of {doc} {n}",
            "source_code": "base", "article_no": doc, **extra}


def _result(run_id: str = "r1", started_at: str = "2026-01-01T00:00:00+00:00"):
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult, QuestionResult
    questions = []
    for i in range(4):
        gold = _ref(9, f"gold{i}", dense_score=0.0, sparse_score=2.0)
        context = [_ref(j, f"other{i}", dense_score=0.0, sparse_score=3.0) for j in range(2)]
        questions.append(QuestionResult(
            question_id=f"q{i}", question=f"question {i}", reference_answer="r",
            generated_answer="an answer", expected_refs=[f"base/gold{i}"],
            metrics={"retrieval_recall_at_k": 0.0, "pre_rerank_recall_at_k": 1.0},
            answerability="answerable",
            source_refs=context,
            # The source it needed was found, ranked fourth, and cut.
            candidate_source_refs=context + [_ref(5, f"more{i}"), gold],
            pre_rerank_source_refs=[gold] + context,
            dense_source_refs=[],
            sparse_source_refs=[gold] + context,
        ))
    return ExperimentResult(
        config=ExperimentConfig(
            name="c", pipeline_id="hybrid_rrf", top_k=2,
            chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="ollama"),
        ),
        run_id=run_id, started_at=started_at, n_questions=4, dataset_name="ds",
        question_results=questions,
    )


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(E.router)
    with TestClient(app) as c:
        yield c


_CONCLUDED = ("diagnostics", "trace_gaps", "diagnosis_depth", "context_size_advice")


def _page(client) -> dict:
    return client.get("/experiments/r1").json()


def _cleared(client, monkeypatch) -> tuple[dict, dict, FakeMongo]:
    store = FakeMongo().install(monkeypatch)
    asyncio.run(E._save(_result()))
    before = _page(client)
    assert asyncio.run(E._clear_windows("r1", datetime.now(UTC).isoformat()))
    return before, _page(client), store


def test_the_fixture_provokes_what_it_is_meant_to(client, monkeypatch) -> None:
    before, _, _ = _cleared(client, monkeypatch)
    ids = {d["id"] for d in before["diagnostics"]}
    assert "half_returned_nothing" in ids and "stub_embedder" not in ids
    assert before["context_size_advice"], "the gold ranked past the cut, so there is advice"
    assert all(q["stages"]["lost_at"] for q in before["question_results"])


def test_every_conclusion_drawn_from_the_windows_survives_their_removal(client, monkeypatch) -> None:
    before, after, _ = _cleared(client, monkeypatch)
    for key in _CONCLUDED:
        assert after[key] == before[key], key
    for mine, theirs in zip(before["question_results"], after["question_results"], strict=True):
        assert theirs["stages"] == mine["stages"]
        assert theirs["funnel"] == mine["funnel"]


def test_the_windows_are_gone_from_the_store_and_the_file(client, monkeypatch) -> None:
    _, after, store = _cleared(client, monkeypatch)
    assert store.data[E._WINDOWS_COLLECTION] == []
    on_disk = json.loads((E._STORE_DIR / "r1.json").read_text(encoding="utf-8"))
    for holder in (after, on_disk):
        assert holder["debug_cleared_at"]
        for q in holder["question_results"]:
            assert not any(q.get(f) for f in E._HEAVY_FIELDS)
            assert q["source_refs"], "the answer's context is not a window and stays"


def test_the_context_keeps_its_texts(client, monkeypatch) -> None:
    before, after, _ = _cleared(client, monkeypatch)
    assert [r["chunk_text"] for q in after["question_results"] for r in q["source_refs"]] == \
        [r["chunk_text"] for q in before["question_results"] for r in q["source_refs"]]


def test_a_cleared_run_is_not_cleared_twice(monkeypatch) -> None:
    FakeMongo().install(monkeypatch)
    asyncio.run(E._save(_result()))
    now = datetime.now(UTC).isoformat()
    assert asyncio.run(E._clear_windows("r1", now))
    assert not asyncio.run(E._clear_windows("r1", now)), (
        "a second pass would freeze the run as it stands without its windows")


def test_a_cleared_run_is_not_reported_as_a_system_that_hides_its_windows(client, monkeypatch) -> None:
    """The trace gaps name what the system under test does not report. A
    run whose windows the store removed is not such a system."""
    _, after, _ = _cleared(client, monkeypatch)
    fields = {g["field"] for g in after["trace_gaps"]}
    assert "pre_rerank_source_refs" not in fields and "candidate_source_refs" not in fields
