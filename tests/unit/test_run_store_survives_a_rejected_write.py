"""A run the database refuses must still be readable.

Found by reading the store's two halves together. `_save` wrote to the
database inside a bare `except: pass` and then wrote a file copy; `_get_results`
read the database and returned as soon as it had anything at all, reaching the
files only when the database was completely empty. So a run the database
rejected existed on disk and appeared nowhere, for as long as any other run
existed, which is always.

The rejection is not hypothetical. A run document larger than the engine's
limit is refused, and a run with a wide candidate window costs what
`WORST_BYTES_PER_QUESTION` says per question, which puts the limit where
`QUESTIONS_A_RUN_CAN_HOLD` does. Both are measured over the runs stored here
by the test at the end of this file, because the prose that carried those
numbers had nothing holding it to a measurement: the figure it stated was
right, and an audit that measured with the wrong encoding revised it to two
and a half times the truth and met nothing that argued back.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from services.api_gateway.routers import experiments as E

ROOT = Path(__file__).resolve().parents[2]


def _stored_run(run_id: str, n_questions: int = 2) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config": {
            "name": run_id,
            "chunking_strategy": {"kind": "chunker", "component_id": "fixed"},
            "embedder": {"kind": "embedder", "component_id": "bge_m3"},
            "generator": {"kind": "generator", "component_id": "ollama"},
        },
        "started_at": "2026-01-01T00:00:00+00:00",
        "n_questions": n_questions,
        "question_results": [
            {"question_id": f"q{i}", "question": "?", "reference_answer": "",
             "generated_answer": "a", "metrics": {}, "source_refs": []}
            for i in range(n_questions)
        ],
    }


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(E, "_STORE_DIR", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_a_run_only_on_disk_is_still_listed(store: Path) -> None:
    """The database holds one run and the disk holds two. Before the fix the
    second was unreachable."""
    (store / "rejected.json").write_text(
        json.dumps(_stored_run("rejected")), encoding="utf-8")

    with patch("adapters.mongodb.find_many", new=AsyncMock(return_value=[_stored_run("accepted")])):
        results = await E._get_results()

    assert set(results) == {"accepted", "rejected"}, (
        "a run the database rejected is invisible while any other run exists"
    )


@pytest.mark.asyncio
async def test_the_database_wins_where_both_hold_the_same_run(store: Path) -> None:
    """The file copy is only ever as new as the write that produced it."""
    stale = _stored_run("r1", n_questions=2)
    stale["dataset_name"] = "from-disk"
    (store / "r1.json").write_text(json.dumps(stale), encoding="utf-8")

    fresh = _stored_run("r1", n_questions=2)
    fresh["dataset_name"] = "from-database"

    with patch("adapters.mongodb.find_many", new=AsyncMock(return_value=[fresh])):
        results = await E._get_results()

    assert results["r1"].dataset_name == "from-database"


@pytest.mark.asyncio
async def test_the_disk_alone_still_answers_when_the_database_is_unreachable(store: Path) -> None:
    (store / "r1.json").write_text(json.dumps(_stored_run("r1")), encoding="utf-8")

    with patch("adapters.mongodb.find_many", new=AsyncMock(side_effect=RuntimeError("no server"))):
        results = await E._get_results()

    assert set(results) == {"r1"}


@pytest.mark.asyncio
async def test_a_rejected_write_is_reported_and_the_file_copy_is_still_written(
    store: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swallowing it is what made the run invisible; the copy is what saves it."""
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult

    cfg = ExperimentConfig(
        name="big",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="ollama"),
    )
    result = ExperimentResult(config=cfg, run_id="big")

    warnings: list[tuple[str, dict[str, Any]]] = []

    class _Log:
        def warning(self, event: str, **kw: Any) -> None:
            warnings.append((event, kw))

    import structlog
    monkeypatch.setattr(structlog, "get_logger", lambda *a, **k: _Log())

    with patch("adapters.mongodb.upsert_one",
               new=AsyncMock(side_effect=RuntimeError("document too large"))):
        await E._save(result)

    assert (store / "big.json").exists(), "the copy that survives a rejection was not written"
    assert warnings, "the rejection was swallowed, which is how the run went missing"
    event, fields = warnings[0]
    assert event == "experiment.save.database_rejected"
    assert fields["run_id"] == "big"
    assert "size_bytes" in fields, "the size is what tells a reader why it was rejected"


def test_the_worst_rate_named_here_is_the_worst_rate_stored_here() -> None:
    """The number in the prose above, tied to the runs it claims to describe.

    A rate written into a docstring is a measurement with no way of going
    stale loudly, and this one had no way of being wrong loudly either. An
    audit measured the store with the serialiser's default escaping, reported
    522 KB per question against the stated 215 KB, and moved the ceiling from
    seventy-eight questions to thirty-one. The stated figure was right and the
    audit was wrong, and neither the prose nor anything else could say which.
    Now the number is measured the way the engine stores it, and the decision
    the plan hangs on it, whether to split a run's storage per question, has
    something under it.

    A ratchet in the direction that matters: the stated worst case may exceed
    the measured one and may never fall below it, so a fatter run reddens this
    and whoever raises the number meets the ceiling it implies.
    """
    from services.api_gateway.routers.experiments import (
        QUESTIONS_A_RUN_CAN_HOLD,
        WORST_BYTES_PER_QUESTION,
    )

    LIMIT = 16 * 1024 * 1024
    files = sorted((ROOT / "eval" / "results" / "runs").glob("*.json"))
    if not files:
        pytest.skip("NOT RUN: no stored run to measure")

    worst = 0
    fattest = ""
    for path in files:
        run = json.loads(path.read_text(encoding="utf-8"))
        questions = len(run.get("question_results") or [])
        if not questions:
            continue
        # As the engine stores it, in two respects. UTF-8, because the
        # default escaping writes a Cyrillic character as six bytes where
        # UTF-8 writes two and reports two and a half times the real size.
        # And without the retrieval windows, because those are written to
        # documents of their own, one per question, and the run document is
        # what the ceiling is about.
        stored, windows = E._split_windows(run)
        rate = len(json.dumps(stored, separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8")) // questions
        widest = max((len(json.dumps(w, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8"))
                      for w in windows), default=0)
        assert widest < LIMIT, (
            f"{path.name} has a single question whose retrieval window is {widest} bytes, "
            "which is a document the engine will refuse on its own"
        )
        if rate > worst:
            worst, fattest = rate, path.name

    assert worst, "no stored run carries a question, so this measures nothing"
    assert worst <= WORST_BYTES_PER_QUESTION, (
        f"{fattest} costs {worst} bytes per question and the constant says "
        f"{WORST_BYTES_PER_QUESTION}. Raise it, and read the ceiling it implies: a run of "
        f"that shape then holds {16 * 1024 * 1024 // worst} questions."
    )
    assert QUESTIONS_A_RUN_CAN_HOLD == 16 * 1024 * 1024 // WORST_BYTES_PER_QUESTION
