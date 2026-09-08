"""Rewriting a stored run into the shape today's writer produces.

The split and the deduplication happen when a run is written, so a run written
before them keeps the shape it had. Seventy-four of the hundred and twenty-nine
runs on the machine this was written on were in that state, and the store was
296 MiB where the same runs rewritten are 211.

A one-line loop would do the work. What makes it worth a tool is that a
rewrite is destructive and a run is worth hours. So the tool proves twice, per
run: that taking the run apart and putting it back gives what a reader had, and
that what came back out of the store afterwards is that same thing. Run as a
check alone, before anything was written, the first of those found two ways the
reassembly did not return what it was given, and a third came out of comparing
the rewritten store against a snapshot taken first.
"""
from __future__ import annotations

import copy
from typing import Any

import pytest

from services.api_gateway.routers import experiments as E
from tools import compact_run_storage as tool


class _Store:
    """A run store that answers and remembers, with no engine under it."""

    def __init__(self, runs: list[dict[str, Any]],
                 windows: list[dict[str, Any]] | None = None) -> None:
        self.runs = {r["run_id"]: copy.deepcopy(r) for r in runs}
        self.windows = copy.deepcopy(windows or [])
        self.writes = 0

    async def find_many(self, collection: str, query: Any = None, sort: Any = None,
                        limit: int = 0, projection: Any = None) -> list[dict[str, Any]]:
        if collection == "experiment_runs":
            return [copy.deepcopy(r) for r in self.runs.values()]
        wanted = ((query or {}).get("run_id") or {}).get("$in") or []
        return [copy.deepcopy(w) for w in self.windows if w.get("run_id") in wanted]

    async def find_one(self, collection: str, query: Any,
                       projection: Any = None) -> dict[str, Any] | None:
        found = self.runs.get(query.get("run_id"))
        return copy.deepcopy(found) if found else None

    async def upsert_one(self, collection: str, query: Any, doc: dict[str, Any]) -> None:
        self._refuse_the_engines_own_key(doc)
        self.writes += 1
        self.runs[query["run_id"]] = {**copy.deepcopy(doc), "_id": "an id of its own"}

    @staticmethod
    def _refuse_the_engines_own_key(doc: dict[str, Any]) -> None:
        """What the engine does, and the reason this stand-in has to do it.

        A document read out of the store carries `_id` as a string, and
        writing that string back is refused: the key is the engine's and
        immutable, and a replacement is a new document to it. A stand-in that
        accepted it passed while the real store answered `WriteError` on the
        first run of the rewrite.
        """
        assert "_id" not in doc, "the engine's own key was written back"

    async def delete_many(self, collection: str, query: Any) -> int:
        run_id = query.get("run_id")
        kept = [w for w in self.windows if w.get("run_id") != run_id]
        removed = len(self.windows) - len(kept)
        self.windows = kept
        return removed

    async def insert_one(self, collection: str, doc: dict[str, Any]) -> str:
        self._refuse_the_engines_own_key(doc)
        self.writes += 1
        self.windows.append({**copy.deepcopy(doc), "_id": "an id of its own"})
        return "id"


def _ref(chunk_id: str, text: str) -> dict[str, Any]:
    return {"chunk_id": chunk_id, "chunk_text": text, "score": 0.5}


def _a_run_in_the_old_shape(run_id: str = "r1") -> dict[str, Any]:
    """Windows written inside the run, and every text written three times."""
    ref = _ref("c1", "the same eight hundred characters " * 20)
    return {
        "_id": "the engine's own key", "run_id": run_id,
        "started_at": "2026-01-01T00:00:00Z", "config": {},
        "question_results": [{
            "question_id": "q0", "generated_answer": "a",
            "source_refs": [dict(ref)],
            "pre_rerank_source_refs": [dict(ref), _ref("c2", "another one")],
            "candidate_source_refs": [dict(ref), _ref("c2", "another one")],
        }],
    }


def _served(store: _Store, run_id: str) -> dict[str, Any]:
    """What a reader of the store gets for one run, without the engine's own
    key, which is the engine's to choose and no part of the run."""
    windows = [w for w in store.windows if w.get("run_id") == run_id]
    served = E._restore_texts(E._merge_windows(store.runs[run_id], windows))
    return {k: v for k, v in served.items() if k != "_id"}


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    import adapters.mongodb as mdb

    fake = _Store([_a_run_in_the_old_shape()])
    for name in ("find_many", "find_one", "upsert_one", "delete_many", "insert_one"):
        monkeypatch.setattr(mdb, name, getattr(fake, name))
    return fake


def test_a_check_alone_writes_nothing(store: _Store) -> None:
    """The default, because the other one cannot be undone."""
    before = copy.deepcopy(store.runs)
    assert tool.main([]) == 0
    assert store.writes == 0
    assert store.runs == before
    assert store.windows == []


def test_a_rewrite_serves_the_reader_exactly_what_it_served_before(store: _Store) -> None:
    was = _served(store, "r1")
    assert tool.main(["--write"]) == 0
    assert store.writes > 0, "nothing was written, so nothing was compacted"
    assert _served(store, "r1") == was


def test_a_rewrite_moves_the_windows_out_and_writes_a_text_once(store: _Store) -> None:
    tool.main(["--write"])
    question = store.runs["r1"]["question_results"][0]
    assert "candidate_source_refs" not in question, "the widest list is still inside the run"
    assert store.windows, "the windows were taken out and written nowhere"
    carrying = [r for f in E._REF_FIELDS for r in (question.get(f) or []) if "chunk_text" in r]
    assert [r["chunk_id"] for r in carrying] == ["c1"], (
        "a fragment's text is still written more than once inside its question"
    )


def test_a_run_already_in_the_new_shape_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise every run would be rewritten on every pass, and a run whose
    windows had already moved would have them deleted and not written back."""
    import adapters.mongodb as mdb

    whole = _a_run_in_the_old_shape("r2")
    light, windows = E._split_windows(E._dedupe_texts(whole))
    fake = _Store([light], [{**w, "_id": "the engine's own key"} for w in windows])
    for name in ("find_many", "find_one", "upsert_one", "delete_many", "insert_one"):
        monkeypatch.setattr(mdb, name, getattr(fake, name))

    assert tool.main(["--write"]) == 0
    assert fake.writes == 0, "a run in the new shape was rewritten"
    served = _served(fake, "r2")
    assert served == {k: v for k, v in E._restore_texts(whole).items() if k != "_id"}


def test_a_run_the_writer_cannot_reproduce_is_reported_and_left(
    store: _Store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check that made the tool worth writing. A writer that loses
    something has to stop the rewrite before the loss reaches the whole
    store, and the run has to be named so somebody can look at it."""
    def _lossy(data: dict[str, Any]) -> dict[str, Any]:
        stripped = copy.deepcopy(data)
        for question in stripped.get("question_results") or []:
            question["source_refs"] = []
        return stripped

    monkeypatch.setattr(E, "_dedupe_texts", _lossy)
    was = copy.deepcopy(store.runs)
    assert tool.main(["--write"]) == 1
    assert store.writes == 0
    assert store.runs == was


def test_an_empty_store_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    import adapters.mongodb as mdb

    fake = _Store([])
    for name in ("find_many", "find_one", "upsert_one", "delete_many", "insert_one"):
        monkeypatch.setattr(mdb, name, getattr(fake, name))
    assert tool.main([]) == 0
    assert fake.writes == 0
