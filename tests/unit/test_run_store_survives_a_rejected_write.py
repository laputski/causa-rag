"""A run the database refuses must still be readable.

Found by reading the store's two halves together. `_save` wrote to the
database inside a bare `except: pass` and then wrote a file copy; `_get_results`
read the database and returned as soon as it had anything at all, reaching the
files only when the database was completely empty. So a run the database
rejected existed on disk and appeared nowhere, for as long as any other run
existed, which is always.

The rejection is not hypothetical. A run document larger than the engine's
limit is refused, and measured on this machine a run with a wide candidate
window costs 215 KB per question, which puts the limit at roughly seventy-eight
questions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from services.api_gateway.routers import experiments as E


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
