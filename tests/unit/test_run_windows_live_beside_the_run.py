"""The retrieval windows leave the run document, and come back on the way out.

Measured over every run stored here: the candidate window is 55% of the whole
store and the pre-rerank list another 29%, because each entry carries the full
text of its fragment. A run of fifty questions with a wide window is 10 MiB of
a 16 MiB limit with those two lists and 1.5 MiB without, so the engine's own
ceiling was seventy-nine questions and is now several hundred.

One document per question and not one per run, which is the whole point. A
per-run windows document would carry the same 8.5 MiB and move the ceiling
from seventy-nine to ninety-five, which only delays the same refusal.

Three things have to hold, and the last is the one that would rot quietly:

- what goes out comes back, so no screen loses a window;
- a run written before the split carries its windows inline and is served
  exactly as it was written;
- what a run says about its own completeness is computed from whether those
  lists are present, so a run served without them would report two gaps in
  its trace that it does not have. Only the list may ask for a run without
  its windows, because a row of the list carries a count and never a fragment.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from services.api_gateway.routers import experiments as E


def _run(questions: int = 3, wide: bool = True) -> dict[str, Any]:
    return {
        "run_id": "r1",
        "config": {"pipeline_source": "in_process"},
        "aggregate_metrics": {},
        "question_results": [
            {
                "question_id": f"q{i}", "question": f"question {i}",
                "reference_answer": "", "generated_answer": "an answer",
                "metrics": {}, "source_refs": [{"chunk_id": f"c{i}", "chunk_text": "kept"}],
                "pre_rerank_source_refs": (
                    [{"chunk_id": f"c{i}", "chunk_text": "x" * 500}] if wide else []),
                "candidate_source_refs": (
                    [{"chunk_id": f"c{i}", "chunk_text": "y" * 500}] if wide else []),
            }
            for i in range(questions)
        ],
    }


def test_the_run_document_keeps_nothing_that_grows_with_a_window() -> None:
    light, windows = E._split_windows(_run())
    for question in light["question_results"]:
        assert "pre_rerank_source_refs" not in question
        assert "candidate_source_refs" not in question
        assert question["source_refs"], "the final context is not a window and stays"
    assert len(windows) == 3
    assert {w["question_id"] for w in windows} == {"q0", "q1", "q2"}
    assert all(w["run_id"] == "r1" for w in windows)


def test_a_question_with_no_window_writes_no_document() -> None:
    """A run with no reranker and no widened window has nothing to move, and
    a document per question of empty lists would cost a write each."""
    _, windows = E._split_windows(_run(wide=False))
    assert windows == []


def test_what_went_out_comes_back_on_the_question_it_belongs_to() -> None:
    original = _run()
    light, windows = E._split_windows(original)
    restored = E._merge_windows(light, windows)
    assert restored == original


def test_the_windows_find_their_questions_by_position_when_ids_are_missing() -> None:
    """A run written before question ids were stable still has to find its
    own windows, and the position is what it has."""
    original = _run()
    for question in original["question_results"]:
        question["question_id"] = ""
    light, windows = E._split_windows(original)
    assert all(w["question_id"] == "" for w in windows)
    restored = E._merge_windows(light, windows)
    for i, question in enumerate(restored["question_results"]):
        assert question["candidate_source_refs"][0]["chunk_text"] == "y" * 500, i


def test_a_run_stored_before_the_split_is_served_exactly_as_written() -> None:
    """Its windows are inline and it has no documents of its own, so merging
    nothing must leave it alone."""
    inline = _run()
    assert E._merge_windows(inline, []) == inline


def test_a_window_never_overwrites_one_the_run_already_carries() -> None:
    inline = _run()
    stale = [{"run_id": "r1", "question_id": "q0", "position": 0,
              "candidate_source_refs": [{"chunk_id": "other", "chunk_text": "stale"}]}]
    merged = E._merge_windows(inline, stale)
    assert merged["question_results"][0]["candidate_source_refs"][0]["chunk_id"] == "c0"


def test_a_second_write_of_one_run_replaces_its_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run written twice under one id would otherwise keep both sets and
    serve whichever the engine returned first, which is the re-ingestion
    failure this platform has a catalogue entry for."""
    calls: list[tuple[str, Any]] = []

    class _Mongo:
        async def upsert_one(self, collection: str, query: Any, doc: Any) -> None:
            calls.append(("upsert", collection))

        async def delete_many(self, collection: str, query: Any) -> int:
            calls.append(("delete", collection))
            return 0

        async def insert_one(self, collection: str, doc: Any) -> str:
            calls.append(("insert", collection))
            return "id"

    import adapters.mongodb as mdb
    fake = _Mongo()
    for name in ("upsert_one", "delete_many", "insert_one"):
        monkeypatch.setattr(mdb, name, getattr(fake, name))
    monkeypatch.setattr(E, "_STORE_DIR", E._STORE_DIR)

    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult, QuestionResult

    result = ExperimentResult(
        config=ExperimentConfig(
            name="c",
            chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="ollama"),
        ),
        run_id="r1", n_questions=1, dataset_name="ds",
        question_results=[QuestionResult(
            question_id="q0", question="q", reference_answer="", generated_answer="a",
            candidate_source_refs=[{"chunk_id": "c0", "chunk_text": "z" * 100}],
        )],
    )
    asyncio.run(E._save(result))
    kinds = [c[0] for c in calls]
    assert kinds.index("delete") < kinds.index("insert"), (
        "the windows of an earlier write are still there when the new ones arrive"
    )
    assert ("insert", E._WINDOWS_COLLECTION) in calls
