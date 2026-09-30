"""Old runs lose their retrieval windows; the last runs of a realm never do.

The rule is the user's: runs are debugged soon after they are made, so an old
run's windows are cleared by age and, while the store is over its limit, by
volume; the last runs of each realm keep theirs so a person coming back after
a long break can carry on. The baseline every run is compared with is kept
too. A text is removed from the store only when nothing refers to it and
nothing has touched it for an hour.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from core.run_retention import Retention, StoredRun, protected_runs, runs_to_clear
from services.api_gateway import run_retention as R
from services.api_gateway.routers import experiments as E
from tests.unit._fake_mongo import FakeMongo

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def _run(run_id: str, realm: str, days_ago: float, window_bytes: int = 100,
         cleared: bool = False) -> StoredRun:
    return StoredRun(run_id=run_id, realm_id=realm,
                     started_at=(NOW - timedelta(days=days_ago)).isoformat(),
                     window_bytes=window_bytes, cleared=cleared)


# ── The rule ──────────────────────────────────────────────────────────────────

def test_the_last_runs_of_every_realm_are_kept_however_old() -> None:
    runs = [_run("a1", "a", 90), _run("a2", "a", 80), _run("a3", "a", 70), _run("b1", "b", 400)]
    chosen = dict(runs_to_clear(runs, Retention(), stored_bytes=0, now=NOW))
    assert chosen == {"a1": "age"}


def test_a_run_younger_than_the_age_limit_is_kept() -> None:
    runs = [_run("old", "a", 31), _run("young", "a", 29), _run("x", "a", 1), _run("y", "a", 0)]
    assert dict(runs_to_clear(runs, Retention(), stored_bytes=0, now=NOW)) == {"old": "age"}


def test_a_protected_run_is_kept() -> None:
    runs = [_run("baseline", "a", 90), _run("a2", "a", 80), _run("x", "a", 1), _run("y", "a", 0)]
    chosen = dict(runs_to_clear(runs, Retention(), stored_bytes=0, protected=["baseline"], now=NOW))
    assert chosen == {"a2": "age"}


def test_over_the_limit_the_oldest_go_first_until_it_fits() -> None:
    mib = 1024 * 1024
    runs = [_run(f"r{i}", "a", 10 - i, window_bytes=mib) for i in range(6)]
    chosen = runs_to_clear(runs, Retention(max_mib=3), stored_bytes=6 * mib, now=NOW)
    assert chosen == [("r0", "volume"), ("r1", "volume"), ("r2", "volume")]


def test_a_run_already_cleared_is_not_chosen_again() -> None:
    runs = [_run("done", "a", 90, cleared=True), _run("x", "a", 1), _run("y", "a", 0)]
    assert runs_to_clear(runs, Retention(), stored_bytes=0, now=NOW) == []


def test_a_start_written_without_a_zone_is_read_as_utc() -> None:
    run = StoredRun("naive", "a", (NOW - timedelta(days=40)).replace(tzinfo=None).isoformat(), 1, False)
    runs = [run, _run("x", "a", 1), _run("y", "a", 0)]
    assert dict(runs_to_clear(runs, Retention(), stored_bytes=0, now=NOW)) == {"naive": "age"}


def test_a_start_that_cannot_be_read_is_never_cleared_by_age() -> None:
    runs = [StoredRun("undated", "a", "", 1, False), _run("x", "a", 1), _run("y", "a", 0)]
    assert runs_to_clear(runs, Retention(), stored_bytes=0, now=NOW) == []


def test_runs_without_a_realm_count_as_one_realm() -> None:
    runs = [_run("n1", "", 90), _run("n2", "", 80), _run("n3", "", 70)]
    assert protected_runs(runs, 2) == {"n2", "n3"}


# ── The pass over the store ───────────────────────────────────────────────────

def _question(i: int, window_text: str) -> object:
    from core.experiment.runner import QuestionResult
    return QuestionResult(
        question_id=f"q{i}", question="q", reference_answer="", generated_answer="a",
        source_refs=[{"chunk_id": "ctx", "chunk_text": "the context everyone keeps"}],
        candidate_source_refs=[{"chunk_id": f"w{i}", "chunk_text": window_text}],
    )


def _result(run_id: str, realm: str, days_ago: float, window_text: str):
    from core.experiment.config import ComponentRef, ExperimentConfig
    from core.experiment.runner import ExperimentResult
    return ExperimentResult(
        config=ExperimentConfig(
            name="c",
            chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
            embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
            generator=ComponentRef(kind="generator", component_id="ollama"),
        ),
        run_id=run_id, realm_id=realm, n_questions=1, dataset_name="ds",
        started_at=(NOW - timedelta(days=days_ago)).isoformat(),
        question_results=[_question(0, window_text)],
    )


@pytest.fixture
def store(monkeypatch) -> FakeMongo:
    store = FakeMongo().install(monkeypatch)
    for run_id, days, text in (("old", 90, "only in the old window"),
                               ("mid", 60, "in two windows"),
                               ("new1", 2, "in two windows"),
                               ("new2", 1, "only in the newest window")):
        asyncio.run(E._save(_result(run_id, "a", days, text)))
    # Every text as if written two hours ago, so the grace period is past.
    for doc in store.data[E._TEXTS_COLLECTION]:
        doc["seen_at"] = (NOW - timedelta(hours=2)).isoformat()
    return store


def _texts(store: FakeMongo) -> set[str]:
    return {d["text"] for d in store.data[E._TEXTS_COLLECTION]}


def test_a_pass_clears_the_old_and_keeps_what_is_still_referred_to(store) -> None:
    report = asyncio.run(R.prune(now=NOW))
    assert {c["run_id"] for c in report["cleared"]} == {"old", "mid"}
    assert {w["run_id"] for w in store.data[E._WINDOWS_COLLECTION]} == {"new1", "new2"}
    assert _texts(store) == {"the context everyone keeps", "in two windows",
                             "only in the newest window"}
    assert report["texts_removed"] == 1


def test_a_text_touched_within_the_hour_is_left_alone(store) -> None:
    for doc in store.data[E._TEXTS_COLLECTION]:
        doc["seen_at"] = (NOW - timedelta(minutes=5)).isoformat()
    asyncio.run(R.prune(now=NOW))
    assert "only in the old window" in _texts(store)


def test_the_baseline_keeps_its_windows(store) -> None:
    store.data["settings"] = [{"_id": "global", "baseline_run_id": "old"}]
    report = asyncio.run(R.prune(now=NOW))
    assert {c["run_id"] for c in report["cleared"]} == {"mid"}
    assert "only in the old window" in _texts(store)


def test_a_preview_changes_nothing(store) -> None:
    before = {k: [dict(d) for d in v] for k, v in store.data.items()}
    report = asyncio.run(R.prune(write=False, now=NOW))
    assert {c["run_id"] for c in report["cleared"]} == {"old", "mid"}
    assert store.data == before


def test_the_rule_is_read_from_the_settings(store) -> None:
    store.data["settings"] = [{"_id": "global", "debug_retention": {"keep_last_per_realm": 3}}]
    report = asyncio.run(R.prune(now=NOW))
    assert {c["run_id"] for c in report["cleared"]} == {"old"}


def test_a_run_that_exists_only_as_a_file_keeps_its_texts(store) -> None:
    """A run the database refused lives in its file alone, and the texts it
    refers to are in the store all the same."""
    asyncio.run(E._save(_result("file-only", "b", 0, "only in a file")))
    store.data["experiment_runs"] = [d for d in store.data["experiment_runs"]
                                     if d["run_id"] != "file-only"]
    store.data[E._WINDOWS_COLLECTION] = [d for d in store.data[E._WINDOWS_COLLECTION]
                                         if d["run_id"] != "file-only"]
    for doc in store.data[E._TEXTS_COLLECTION]:
        doc["seen_at"] = (NOW - timedelta(hours=2)).isoformat()
    asyncio.run(R.prune(now=NOW))
    assert "only in a file" in _texts(store)


# ── Off unless the installation says so ──────────────────────────────────────

def test_the_pass_does_not_run_by_itself_unless_switched_on(store, monkeypatch) -> None:
    """A gateway reloading from a working tree ran this pass on its own store
    the moment the file was saved, twice, once in a deliberately broken
    version. It runs by itself only when the installation asks."""
    monkeypatch.delenv("CAUSA_RUN_RETENTION", raising=False)
    windows = len(store.data[E._WINDOWS_COLLECTION])
    asyncio.run(R.keep_pruning())
    assert len(store.data[E._WINDOWS_COLLECTION]) == windows


def test_switched_on_the_pass_runs_at_start(store, monkeypatch) -> None:
    monkeypatch.setenv("CAUSA_RUN_RETENTION", "on")

    async def one_pass_then_stop(_: float) -> None:
        raise asyncio.CancelledError
    monkeypatch.setattr(R.asyncio, "sleep", one_pass_then_stop)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(R.keep_pruning())
    assert {w["run_id"] for w in store.data[E._WINDOWS_COLLECTION]} == {"new1", "new2"}


def test_the_tool_shows_and_changes_nothing_unless_told_to_write(store, monkeypatch, capsys) -> None:
    from tools import prune_run_debug
    before = {k: [dict(d) for d in v] for k, v in store.data.items()}
    monkeypatch.setattr("sys.argv", ["prune_run_debug"])
    monkeypatch.setattr(R, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: NOW)}))
    prune_run_debug.main()
    assert "nothing was changed" in capsys.readouterr().out
    assert store.data == before
