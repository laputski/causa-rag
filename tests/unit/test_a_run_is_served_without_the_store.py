"""One run, served without reading every other one.

Every endpoint about a single run used to load and parse the whole run store
to find it. Measured on the store this was written against: 1.88 s for the
whole store against 0.08 s for one run, paid on the page a reader opens most
often, and paid five times a second by the progress socket.

What those callers wanted was one of three things: that run, a bit saying it
exists, or a dozen top-level fields of every run. Three readers answer those
three questions now, and the guards below are what keep a fourth caller from
quietly going back to the first one.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult, QuestionResult
from services.api_gateway.routers import experiments as exp_module

_SOURCE = Path(exp_module.__file__).read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _functions_calling(name: str) -> set[str]:
    """Every function in the router whose body calls `name`."""
    found = set()
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) \
                    and inner.func.id == name:
                found.add(node.name)
    return found


def _cfg(**over: Any) -> ExperimentConfig:
    return ExperimentConfig(**{
        "name": "served_without_the_store",
        "dataset_name": "ds",
        "chunking_strategy": ComponentRef(kind="chunker", component_id="fixed"),
        "embedder": ComponentRef(kind="embedder", component_id="bge_m3"),
        "generator": ComponentRef(kind="generator", component_id="stub"),
        **over,
    })


def _a_run(run_id: str = "servedrun1", **over: Any) -> ExperimentResult:
    result = ExperimentResult(
        config=_cfg(), run_id=run_id, n_questions=2, dataset_name="ds",
        started_at="2026-09-08T10:00:00", aggregate_metrics={"retrieval_recall_at_k": 0.9},
        question_results=[
            QuestionResult(question_id=str(i), question=f"q{i}",
                           reference_answer="", generated_answer="a")
            for i in range(2)
        ],
    )
    for key, value in over.items():
        setattr(result, key, value)
    return result


# ── which callers may still read every run ───────────────────────────────────

# The comparison, and only it. It reads two runs and then resamples flipped
# questions through a live pipeline for minutes, so the store it loads is a
# rounding error against what it is about to do, and the paired diff it builds
# reads every answer of both runs anyway.
MAY_READ_EVERY_RUN = {"_compare_experiments"}


def test_only_the_comparison_still_reads_every_run() -> None:
    callers = _functions_calling("_get_results")
    assert callers == MAY_READ_EVERY_RUN, (
        "a caller reads the whole run store to serve one run again; "
        f"expected {sorted(MAY_READ_EVERY_RUN)}, found {sorted(callers)}"
    )


def test_a_summary_refuses_to_be_asked_for_an_answer() -> None:
    """A summary is a run without its answers, and nothing at the call site
    says so. A caller reaching for them would find an empty list where the
    run has two hundred, and would report a run that answered nothing: the
    read-path loss this platform keeps an entry for, committed by the reader
    built to make reading cheap. Both ways of asking say where they went."""
    summary = exp_module._summarise(_a_run().to_dict())
    with pytest.raises(KeyError, match="_get_one_result"):
        summary["question_results"]
    with pytest.raises(KeyError, match="_get_one_result"):
        summary.get("question_results")
    # Every other field answers as a mapping does, absent ones included.
    assert summary["run_id"] == "servedrun1"
    assert summary.get("nothing-here") is None


# ── the run page ─────────────────────────────────────────────────────────────

def _store_that_refuses_to_be_read_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("this endpoint read the whole run store")
    monkeypatch.setattr(exp_module, "_get_results", _refuse)


@pytest.mark.asyncio
async def test_the_run_page_is_served_by_the_one_run_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _a_run()
    _store_that_refuses_to_be_read_whole(monkeypatch)

    async def _one(run_id: str) -> ExperimentResult | None:
        return run if run_id == run.run_id else None

    async def _summaries() -> dict[str, dict[str, Any]]:
        return {run.run_id: exp_module._summarise(run.to_dict())}

    monkeypatch.setattr(exp_module, "_get_one_result", _one)
    monkeypatch.setattr(exp_module, "_get_summaries", _summaries)

    payload = await exp_module.get_experiment(run.run_id)
    assert payload["run_id"] == run.run_id
    assert payload["status"] == "done"
    assert len(payload["question_results"]) == 2
    assert payload["tuning_provenance"]["dataset_name"] == "ds"


@pytest.mark.asyncio
async def test_the_miss_diagnosis_is_served_by_the_one_run_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    _store_that_refuses_to_be_read_whole(monkeypatch)

    async def _none(run_id: str) -> ExperimentResult | None:
        return None

    monkeypatch.setattr(exp_module, "_get_one_result", _none)
    with pytest.raises(HTTPException) as raised:
        await exp_module.diagnose_retrieval_miss_endpoint("no-such-run", "q0")
    assert raised.value.status_code == 404


# ── existence ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_existence_is_answered_without_reading_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Deliberately unreadable, because the question is whether the run is
    there and not what it says. The socket asking this five times a second
    is the reason it must stay that cheap."""
    import adapters.mongodb as mdb

    async def _no_database(*_a: Any, **_k: Any) -> int:
        raise RuntimeError("no server")

    monkeypatch.setattr(mdb, "count", _no_database)
    monkeypatch.setattr(exp_module, "_STORE_DIR", tmp_path)
    (tmp_path / "onlyonfile1.json").write_text("{ this is not json", encoding="utf-8")

    assert await exp_module._run_exists("onlyonfile1") is True
    assert await exp_module._run_exists("never-existed") is False


def test_the_progress_socket_asks_existence_and_nothing_else() -> None:
    socket = next(n for n in ast.walk(_TREE)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "experiment_progress")
    body = ast.unparse(socket)
    assert "_run_exists" in body
    assert "_get_results" not in body and "_get_one_result" not in body, (
        "the socket polls this every 200 ms; reading a run there costs the "
        "whole run five times a second"
    )


# ── what a summary carries ───────────────────────────────────────────────────

def _a_store(docs: list[dict[str, Any]]):
    async def find_many(collection: str, query: dict[str, Any] | None = None,
                        sort: Any = None, limit: int = 0,
                        projection: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert projection, "a summary must not be read out of a whole run"
        if projection.get("question_results") == 0:
            return [{k: v for k, v in d.items() if k != "question_results"} for d in docs]
        wanted = set(((query or {}).get("run_id") or {}).get("$in") or [])
        return [
            {"run_id": d["run_id"],
             "question_results": [{"question_id": q["question_id"]}
                                  for q in d.get("question_results") or []]}
            for d in docs if d["run_id"] in wanted
        ]
    return find_many


@pytest.mark.asyncio
async def test_a_summary_leaves_the_answers_in_the_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import adapters.mongodb as mdb

    run = _a_run()
    monkeypatch.setattr(mdb, "find_many", _a_store([run.to_dict()]))
    monkeypatch.setattr(exp_module, "_STORE_DIR", tmp_path)

    summaries = await exp_module._get_summaries()
    summary = summaries[run.run_id]
    assert "question_results" not in summary
    # Everything a row, a point and a provenance count are built from.
    for field in ("run_id", "config", "config_hash", "config_name", "started_at",
                  "dataset_name", "realm_id", "aggregate_metrics", "avg_stage_trace"):
        assert field in summary, f"a summary lost {field}"


@pytest.mark.asyncio
async def test_a_summary_names_a_configuration_the_way_the_run_page_does(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Of the runs stored on the machine this was written on, forty-eight
    carry an identity written by a version that computed it differently. Read
    back as stored, a row and the page it opens would name one run's
    configuration two ways, and one configuration run in two eras would count
    as two configurations that somebody tried."""
    import adapters.mongodb as mdb

    run = _a_run()
    stored = {**run.to_dict(), "config_hash": "written-by-an-older-version"}
    monkeypatch.setattr(mdb, "find_many", _a_store([stored]))
    monkeypatch.setattr(exp_module, "_STORE_DIR", tmp_path)

    summaries = await exp_module._get_summaries()
    assert summaries[run.run_id]["config_hash"] == run.to_dict()["config_hash"]


@pytest.mark.asyncio
async def test_a_run_the_database_rejected_still_has_a_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The file copy is what a run too large for the database lives in, and a
    list that skipped it would be the invisible-run failure again, one reader
    along."""
    import adapters.mongodb as mdb

    async def _no_database(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
        raise RuntimeError("no server")

    run = _a_run("onlyonfile2")
    monkeypatch.setattr(mdb, "find_many", _no_database)
    monkeypatch.setattr(exp_module, "_STORE_DIR", tmp_path)
    (tmp_path / "onlyonfile2.json").write_text(
        json.dumps(run.to_dict(), ensure_ascii=False), encoding="utf-8")

    summaries = await exp_module._get_summaries()
    assert set(summaries) == {"onlyonfile2"}
    assert summaries["onlyonfile2"]["n_answered"] == 2
    assert "question_results" not in summaries["onlyonfile2"]


def test_the_frontier_reads_the_average_the_run_recorded() -> None:
    """A parsed run recomputes its averaged stage trace from its answers; a
    summary carries what the run wrote. Measured over every run in the store
    this was written against, no run's frontier point differs between the
    two, and a run carrying stage traces with no average of them does not
    exist. One arriving later reports no latency, which is the same answer it
    gives for a run that traced nothing."""
    from core.eval.frontier import point_from_run

    run = _a_run()
    stored = run.to_dict()
    stored["avg_stage_trace"] = {"total_ms": 120.0}
    point = point_from_run(exp_module._summarise(stored))
    assert point is not None
    assert point.latency_ms == 120.0
    assert point_from_run(exp_module._summarise(run.to_dict())) is None
