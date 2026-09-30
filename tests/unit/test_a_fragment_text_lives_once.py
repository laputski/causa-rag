"""A fragment's text lives once, in its own store, and a run refers to it.

What has to hold: the page of a run shows exactly the texts the run saw;
two texts under one fragment identifier stay two; the file beside the run
refers to the store only when the store took the texts; a text the store
lost is said to be missing and not shown as empty; and the checks that read
texts reach the same findings from the identifiers alone.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.eval.detectors import run_detectors
from core.fragment_texts import externalise, internalise, text_id, text_ids_in
from services.api_gateway.routers import experiments as E
from tests.unit._fake_mongo import FakeMongo


def _ref(chunk_id: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"chunk_id": chunk_id, "chunk_text": text, "score": 0.5, **extra}


def _result():
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult, QuestionResult
    return ExperimentResult(
        config=ExperimentConfig(
            name="c",
            chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="ollama"),
        ),
        run_id="r1", n_questions=2, dataset_name="ds",
        question_results=[
            QuestionResult(
                question_id="q0", question="q", reference_answer="", generated_answer="a",
                source_refs=[_ref("c1", "first text"), _ref("c2", "second text")],
                candidate_source_refs=[_ref("c1", "first text"), _ref("c3", "third text")],
                dense_source_refs=[_ref("c3", "third text")],
            ),
            QuestionResult(
                question_id="q1", question="q", reference_answer="", generated_answer="a",
                # One identifier, two texts: the store must not merge them.
                source_refs=[_ref("c1", "first text, as another version says it")],
            ),
        ],
    )


# ── The pure half ─────────────────────────────────────────────────────────────

def test_what_goes_out_comes_back() -> None:
    data = _result().to_dict()
    shaped, texts = externalise(data, E._REF_FIELDS)
    assert internalise(shaped, E._REF_FIELDS, texts) == {
        **data, "question_results": [
            {**q, **{f: [{**r, "text_id": text_id(r["chunk_text"]),
                          "text_chars": len(r["chunk_text"].strip())} for r in q[f]]
                     for f in E._REF_FIELDS if q.get(f)}}
            for q in data["question_results"]]}


def test_a_reference_keeps_no_text_and_the_store_holds_each_once() -> None:
    shaped, texts = externalise(_result().to_dict(), E._REF_FIELDS)
    assert "chunk_text" not in json.dumps(shaped)
    assert sorted(texts.values()) == sorted({
        "first text", "second text", "third text", "first text, as another version says it"})


def test_one_fragment_identifier_with_two_texts_keeps_both() -> None:
    shaped, texts = externalise(_result().to_dict(), E._REF_FIELDS)
    back = internalise(shaped, E._REF_FIELDS, texts)
    assert back["question_results"][0]["source_refs"][0]["chunk_text"] == "first text"
    assert back["question_results"][1]["source_refs"][0]["chunk_text"] == \
        "first text, as another version says it"


def test_a_text_the_store_lost_is_marked_and_not_passed_off_as_empty() -> None:
    shaped, texts = externalise(_result().to_dict(), E._REF_FIELDS)
    lost = text_id("second text")
    back = internalise(shaped, E._REF_FIELDS, {k: v for k, v in texts.items() if k != lost})
    ref = back["question_results"][0]["source_refs"][1]
    assert ref["chunk_text"] == "" and ref["text_missing"] is True
    assert "text_missing" not in back["question_results"][0]["source_refs"][0]


def test_a_stored_length_is_the_length_without_surrounding_whitespace() -> None:
    """The check for fragments that carry only a heading measures after
    stripping, and a length taken before would move its threshold."""
    data = {"question_results": [{"source_refs": [_ref("c", "  heading \n")]}]}
    shaped, _ = externalise(data, E._REF_FIELDS)
    assert shaped["question_results"][0]["source_refs"][0]["text_chars"] == len("heading")


# ── The checks read identifiers ───────────────────────────────────────────────

def _run_that_every_text_check_fires_on() -> dict[str, Any]:
    long = "a body long enough to be more than a heading, by a margin"
    return {
        "run_id": "r", "config": {}, "aggregate_metrics": {},
        "question_results": [
            {"question_id": "q0", "metrics": {}, "generated_answer": "a",
             "source_refs": [_ref("c1", long), _ref("c2", long)]},
            {"question_id": "q1", "metrics": {}, "generated_answer": "a",
             "source_refs": [_ref("h1", "Section 1", structural_path="1"),
                             _ref("h2", "Section 2", structural_path="2"),
                             _ref("c1", long + " and a different ending", structural_path="3")]},
        ],
    }


@pytest.mark.parametrize("finding", ["duplicates", "header_only", "chunk_id_collision"])
def test_a_check_that_read_texts_reaches_the_same_finding_from_identifiers(finding: str) -> None:
    run = _run_that_every_text_check_fires_on()
    shaped, _ = externalise(run, E._REF_FIELDS)
    before = {d.id: d.params for d in run_detectors(run)}
    after = {d.id: d.params for d in run_detectors(shaped)}
    assert finding in before, "the fixture does not provoke the finding it is meant to"
    assert after.get(finding) == before[finding]


def test_a_collision_after_the_first_400_characters_is_one() -> None:
    """The check compared the first 400 characters, so two texts differing
    only after them were counted as one."""
    head = "x" * 400
    run = {"run_id": "r", "config": {}, "aggregate_metrics": {}, "question_results": [
        {"question_id": "q0", "metrics": {}, "source_refs": [_ref("c1", head + "one")]},
        {"question_id": "q1", "metrics": {}, "source_refs": [_ref("c1", head + "two")]},
    ]}
    assert "chunk_id_collision" in {d.id for d in run_detectors(run)}


# ── The store ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(E.router)
    with TestClient(app) as c:
        yield c


def test_the_page_of_a_run_shows_the_texts_the_run_saw(client, monkeypatch) -> None:
    store = FakeMongo().install(monkeypatch)
    result = _result()
    asyncio.run(E._save(result))

    assert "chunk_text" not in json.dumps(store.data["experiment_runs"])
    assert "chunk_text" not in json.dumps(store.data[E._WINDOWS_COLLECTION])
    held = [d["text"] for d in store.data[E._TEXTS_COLLECTION]]
    assert len(held) == len(set(held)) == 4

    served = client.get("/experiments/r1").json()
    for mine, theirs in zip(result.to_dict()["question_results"], served["question_results"],
                           strict=True):
        for field in E._REF_FIELDS:
            assert [r["chunk_text"] for r in theirs.get(field) or []] == \
                [r["chunk_text"] for r in mine.get(field) or []], field


def test_a_second_run_over_the_same_fragments_adds_no_text(monkeypatch) -> None:
    store = FakeMongo().install(monkeypatch)
    asyncio.run(E._save(_result()))
    second = _result()
    second.run_id = "r2"
    asyncio.run(E._save(second))
    assert len(store.data[E._TEXTS_COLLECTION]) == 4


def test_the_file_refers_to_the_store_when_the_store_took_the_texts(monkeypatch) -> None:
    FakeMongo().install(monkeypatch)
    asyncio.run(E._save(_result()))
    written = json.loads((E._STORE_DIR / "r1.json").read_text(encoding="utf-8"))
    assert "chunk_text" not in json.dumps(written)
    assert text_ids_in(written["question_results"], E._REF_FIELDS)


def test_the_file_keeps_the_texts_when_the_store_refused_them(monkeypatch) -> None:
    """The file is the copy that survives the database. A file of
    identifiers whose texts never reached the store would survive as a run
    nobody can read."""
    store = FakeMongo().install(monkeypatch)

    async def refuse(collection: str, ops: Any) -> None:
        raise ConnectionError("store unreachable")
    monkeypatch.setattr("adapters.mongodb.bulk_upsert", refuse)
    asyncio.run(E._save(_result()))
    written = json.loads((E._STORE_DIR / "r1.json").read_text(encoding="utf-8"))
    assert written["question_results"][0]["source_refs"][0]["chunk_text"] == "first text"
    assert store.data.get("experiment_runs") in (None, []), (
        "a run whose texts were not stored must not enter the database referring to them")


# ── A run as a file ───────────────────────────────────────────────────────────

def test_the_size_of_either_file_is_told_before_it_is_taken(client, monkeypatch) -> None:
    FakeMongo().install(monkeypatch)
    asyncio.run(E._save(_result()))
    size = client.get("/experiments/r1/export/size").json()
    bare = client.get("/experiments/r1/export").content
    full = client.get("/experiments/r1/export?with_texts=true").content
    assert size["bytes_without_texts"] == len(bare) and size["bytes_with_texts"] == len(full)
    assert size["bytes_with_texts"] > size["bytes_without_texts"]
    assert size["debug_cleared_at"] is None
    assert b"chunk_text" not in bare
    assert json.loads(full)["question_results"][1]["source_refs"][0]["chunk_text"] == \
        "first text, as another version says it"


def test_a_file_of_a_run_that_does_not_exist_is_a_404(client, monkeypatch) -> None:
    FakeMongo().install(monkeypatch)
    assert client.get("/experiments/nothing/export").status_code == 404
