"""Async job-model for create_experiment.

POST /experiments must not block the gateway's event loop for the whole
run; GET /experiments/{run_id} must distinguish "still running" / "failed"
/ "done" without a 404 while in flight.
"""
from __future__ import annotations

import asyncio

import pytest

from adapters.bge_m3 import BgeM3Embedder
from adapters.generator_stub import GeneratorStub
from adapters.qdrant import QdrantRetrieverStub
from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult, ExperimentRunner, QuestionResult
from core.pipeline import NaivePipeline
from core.registry import ComponentRegistry
from eval.dataset import make_stub_dataset
from services.api_gateway.routers import experiments as exp_module


def _make_runner() -> ExperimentRunner:
    reg = ComponentRegistry()
    emb = BgeM3Embedder()
    ret = QdrantRetrieverStub()
    gen = GeneratorStub()
    pipeline = NaivePipeline(retriever=ret, embedder=emb, generator=gen)
    reg.register("embedder", "bge_m3", emb)
    reg.register("retriever", "qdrant_dense_stub", ret)
    reg.register("generator", "stub", gen)
    reg.register("pipeline", "naive", pipeline)
    return ExperimentRunner(registry=reg)


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="async_job_test",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        retrievers=[ComponentRef(kind="retriever", component_id="qdrant_dense_stub")],
        generator=ComponentRef(kind="generator", component_id="stub"),
    )


def test_background_run_populates_progress_then_clears_running(monkeypatch):
    run_id = "bgtest1"
    saved = {}

    async def _fake_save(result):
        saved["result"] = result

    monkeypatch.setattr(exp_module, "_save", _fake_save)
    exp_module._progress[run_id] = []
    exp_module._running.add(run_id)

    runner = _make_runner()
    dataset = make_stub_dataset(n=3)

    asyncio.run(exp_module._run_experiment_background(run_id, _cfg(), dataset, None, runner))

    assert run_id not in exp_module._running
    assert run_id not in exp_module._errors
    assert saved["result"].run_id == run_id
    events = exp_module._progress[run_id]
    assert [(e["processed"], e["total"]) for e in events] == [(1, 3), (2, 3), (3, 3)]


def test_background_run_failure_records_error_and_clears_running(monkeypatch):
    run_id = "bgtest2"

    def _boom(*args, **kwargs):
        raise RuntimeError("pipeline exploded")

    runner = _make_runner()
    monkeypatch.setattr(runner, "run", _boom)
    exp_module._progress[run_id] = []
    exp_module._running.add(run_id)

    asyncio.run(exp_module._run_experiment_background(run_id, _cfg(), make_stub_dataset(n=1), None, runner))

    assert run_id not in exp_module._running
    assert "pipeline exploded" in exp_module._errors[run_id]


@pytest.mark.asyncio
async def test_get_experiment_reports_running_status_before_completion(monkeypatch):
    run_id = "bgtest3"
    monkeypatch.setattr(exp_module, "_get_one_result", _no_such_run)
    exp_module._progress[run_id] = [{"type": "progress", "processed": 1, "total": 5}]
    exp_module._running.add(run_id)
    try:
        payload = await exp_module.get_experiment(run_id)
        assert payload["status"] == "running"
        assert payload["progress"] == {"type": "progress", "processed": 1, "total": 5}
    finally:
        exp_module._running.discard(run_id)
        exp_module._progress.pop(run_id, None)


@pytest.mark.asyncio
async def test_get_experiment_raises_500_with_detail_on_error(monkeypatch):
    from fastapi import HTTPException

    run_id = "bgtest4"
    monkeypatch.setattr(exp_module, "_get_one_result", _no_such_run)
    exp_module._errors[run_id] = "pipeline exploded"
    try:
        with pytest.raises(HTTPException) as exc_info:
            await exp_module.get_experiment(run_id)
        assert exc_info.value.status_code == 500
        assert "pipeline exploded" in exc_info.value.detail
    finally:
        exp_module._errors.pop(run_id, None)


@pytest.mark.asyncio
async def test_get_experiment_404_when_run_id_unknown_anywhere():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await exp_module.get_experiment("no-such-run-id")
    assert exc_info.value.status_code == 404


async def _no_such_run(run_id: str):
    return None


# ── Found live: a run picking a slow model (or hitting a stuck external RAG)
# had no way to be interrupted short of waiting out every remaining
# question — POST /{run_id}/stop + should_stop wiring ─────────────────────────

@pytest.mark.asyncio
async def test_stop_experiment_404_when_not_running():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await exp_module.stop_experiment("no-such-run-id")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_stop_experiment_marks_running_run_for_stopping():
    run_id = "stoptest1"
    exp_module._running.add(run_id)
    try:
        payload = await exp_module.stop_experiment(run_id)
        assert payload == {"run_id": run_id, "status": "stopping"}
        assert run_id in exp_module._stop_requested
    finally:
        exp_module._running.discard(run_id)
        exp_module._stop_requested.discard(run_id)


def test_background_run_honors_stop_requested_and_clears_it_after(monkeypatch):
    """End-to-end through _run_experiment_background: marking a run_id in
    _stop_requested before it starts must actually reach
    ExperimentRunner.run()'s should_stop and halt it early, and
    _stop_requested must be cleaned up afterward (mirrors _running/
    _running_meta's own finally-block cleanup)."""
    run_id = "bgtest5"
    saved = {}

    async def _fake_save(result):
        saved["result"] = result

    monkeypatch.setattr(exp_module, "_save", _fake_save)
    exp_module._progress[run_id] = []
    exp_module._running.add(run_id)
    exp_module._stop_requested.add(run_id)

    runner = _make_runner()
    dataset = make_stub_dataset(n=5)

    asyncio.run(exp_module._run_experiment_background(run_id, _cfg(), dataset, None, runner))

    assert run_id not in exp_module._running
    assert run_id not in exp_module._stop_requested
    assert saved["result"].stopped is True
    assert saved["result"].question_results == []


def test_stop_experiment_404s_once_loop_finished_even_during_save(monkeypatch):
    """Found live: a stop request landing after the question loop already
    exited normally, but before the still-in-flight `await _save(result)`
    completed, used to succeed with {"status": "stopping"} even though
    should_stop would never be consulted again — the caller was told a run
    was stopping when it never would be. `_running.discard(run_id)` must
    happen the moment runner.run() returns, not only in the `finally` after
    `_save`, so a stop request arriving during that window correctly 404s
    instead of silently doing nothing."""
    from fastapi import HTTPException

    run_id = "racetest1"
    seen: dict[str, int] = {}

    async def _fake_save(result):
        # Simulates a stop request arriving while the real Mongo/file write
        # this stands in for is still in flight.
        with pytest.raises(HTTPException) as exc_info:
            await exp_module.stop_experiment(run_id)
        seen["status_code"] = exc_info.value.status_code

    monkeypatch.setattr(exp_module, "_save", _fake_save)
    exp_module._progress[run_id] = []
    exp_module._running.add(run_id)

    runner = _make_runner()
    dataset = make_stub_dataset(n=2)

    asyncio.run(exp_module._run_experiment_background(run_id, _cfg(), dataset, None, runner))

    assert seen["status_code"] == 404


# Found live: GET /experiments (the list page) had no way to tell a stopped
# run apart from a normally-completed one — it showed the dataset's full
# planned n_questions and no `stopped` field at all, so a run stopped at
# 6/30 questions looked identical to one that genuinely answered all 30.
def _only_what_the_projection_asked_for(docs):
    """A run store that hands back the fields it was asked for and no others.

    The stand-in is strict about the projection because the projection is
    what the list rests on. A reader that stopped naming one would be handed
    every answer of every run here and would pass, and on the real store it
    would go back to parsing a quarter of a gigabyte to draw rows that carry
    no answer at all.
    """
    async def find_many(collection, query=None, sort=None, limit=0, projection=None):
        assert collection == "experiment_runs"
        assert projection, "a list of rows must not read whole runs"
        if projection.get("question_results") == 0:
            return [{k: v for k, v in d.items() if k != "question_results"} for d in docs]
        # The other read this store answers: how far a stopped run got,
        # asked as identifiers alone.
        assert projection == {"run_id": 1, "question_results.question_id": 1}
        wanted = set(((query or {}).get("run_id") or {}).get("$in") or [])
        return [
            {
                "run_id": d["run_id"],
                "question_results": [
                    {"question_id": q["question_id"]} for q in d.get("question_results") or []
                ],
            }
            for d in docs if d["run_id"] in wanted
        ]
    return find_many


def test_list_experiments_reflects_stopped_run(monkeypatch, tmp_path):
    stopped_result = ExperimentResult(
        config=_cfg(), run_id="stoppedrun1", n_questions=30, dataset_name="ds",
        stopped=True,
        question_results=[
            QuestionResult(question_id=str(i), question=f"q{i}", reference_answer="", generated_answer="a")
            for i in range(6)
        ],
    )
    finished_result = ExperimentResult(
        config=_cfg(), run_id="finishedrun1", n_questions=3, dataset_name="ds",
        stopped=False,
        question_results=[
            QuestionResult(question_id=str(i), question=f"q{i}", reference_answer="", generated_answer="a")
            for i in range(3)
        ],
    )

    import adapters.mongodb as mdb
    monkeypatch.setattr(exp_module, "_STORE_DIR", tmp_path)
    monkeypatch.setattr(
        mdb, "find_many",
        _only_what_the_projection_asked_for(
            [stopped_result.to_dict(), finished_result.to_dict()]),
    )

    items = asyncio.run(exp_module.list_experiments())
    by_id = {item.run_id: item for item in items}

    assert by_id["stoppedrun1"].stopped is True
    assert by_id["stoppedrun1"].n_questions == 6  # actual answered count, not the planned 30
    assert by_id["finishedrun1"].stopped is False
    assert by_id["finishedrun1"].n_questions == 3
