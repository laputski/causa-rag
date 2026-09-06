"""Reporting a failure without letting anyone edit the catalogue.

The catalogue is read-only from the interface, and that is the whole reason
its guarantees hold: an entry claims a signal catches a failure, and the
build refuses such a claim without a bait. Let the interface write entries
and the refusal cannot hold.

So a person who meets a failure writes a candidate instead. It claims
nothing about detection, it says on every path out of here that no bait has
confirmed it, and it becomes an entry only through a change to the
repository, which is the only place a bait can be written.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException

from services.api_gateway.routers.atlas import (
    CandidateDecision,
    CandidatePromotion,
    CandidateRequest,
    create_candidate,
    decide_candidate,
    list_candidates,
    record_promotion,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _Store:
    """One collection, in memory, behaving as the adapter does."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []

    async def insert_one(self, collection: str, doc: dict[str, Any]) -> str:
        self.docs.append(dict(doc))
        return "id"

    async def find_one(self, collection: str, query: dict[str, Any]) -> dict[str, Any] | None:
        return next((d for d in self.docs if all(d.get(k) == v for k, v in query.items())), None)

    async def find_many(
        self, collection: str, query: dict[str, Any] | None = None, **kw: Any
    ) -> list[dict[str, Any]]:
        query = query or {}
        return [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]

    async def upsert_one(
        self, collection: str, query: dict[str, Any], doc: dict[str, Any]
    ) -> None:
        self.docs = [d for d in self.docs if not all(d.get(k) == v for k, v in query.items())]
        self.docs.append(dict(doc))


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    import adapters.mongodb as mdb

    fake = _Store()
    for name in ("insert_one", "find_one", "find_many", "upsert_one"):
        monkeypatch.setattr(mdb, name, getattr(fake, name))
    return fake


def _report(**overrides: Any) -> CandidateRequest:
    return CandidateRequest(**{
        "title": "The reranker drops the only fragment that answers",
        "looked_like": "Retrieval found the right section every time and the answer never used it.",
        "observed_on": "run 9da7259a, corpus handbook",
        "suspected_signal": "pre_rerank_recall_at_k",
        **overrides,
    })


def test_a_report_says_on_its_face_that_no_bait_confirmed_it(store: _Store) -> None:
    """A candidate and an entry look alike on a screen, and only one of them
    has been proven to be caught by anything."""
    created = _run(create_candidate(_report(), realm_id="demo"))
    assert created["confirmed_by_a_bait"] is False
    assert created["status"] == "proposed"
    listed = _run(list_candidates(realm_id="demo"))["candidates"]
    assert [c["confirmed_by_a_bait"] for c in listed] == [False]


def test_a_report_is_scoped_to_the_realm_it_was_written_in(store: _Store) -> None:
    _run(create_candidate(_report(), realm_id="demo"))
    _run(create_candidate(_report(title="Both halves return the same ten"), realm_id="proving-ground"))
    assert len(_run(list_candidates(realm_id="demo"))["candidates"]) == 1
    assert len(_run(list_candidates())["candidates"]) == 2


def test_a_rejection_without_a_reason_is_refused(store: _Store) -> None:
    """The reporter needs to know why, and so does the next reader who meets
    the same thing and wonders whether it was already dismissed."""
    created = _run(create_candidate(_report(), realm_id="demo"))
    with pytest.raises(HTTPException) as raised:
        _run(decide_candidate(created["candidate_id"], CandidateDecision(status="rejected")))
    assert raised.value.status_code == 400
    assert "reason" in str(raised.value.detail)


def test_a_decision_is_recorded_with_what_it_was_decided_for(store: _Store) -> None:
    created = _run(create_candidate(_report(), realm_id="demo"))
    decided = _run(decide_candidate(
        created["candidate_id"],
        CandidateDecision(status="rejected", note="This is our own async call in a sync handler."),
    ))
    assert decided["status"] == "rejected"
    assert "async call" in decided["note"]
    assert decided["confirmed_by_a_bait"] is False


def test_a_candidate_cannot_claim_to_have_become_an_entry_nobody_wrote(store: _Store) -> None:
    """Promotion is a change to the repository, because an entry arrives with
    a bait and a bait is a test. Recording the pointer is all that happens
    here, and only once the entry exists."""
    created = _run(create_candidate(_report(), realm_id="demo"))
    _run(decide_candidate(created["candidate_id"], CandidateDecision(status="accepted")))
    with pytest.raises(HTTPException) as raised:
        _run(record_promotion(created["candidate_id"], CandidatePromotion(failure_id="F99")))
    assert raised.value.status_code == 400
    assert "F99" in str(raised.value.detail)


def test_a_candidate_nobody_accepted_cannot_be_promoted(store: _Store) -> None:
    created = _run(create_candidate(_report(), realm_id="demo"))
    with pytest.raises(HTTPException) as raised:
        _run(record_promotion(created["candidate_id"], CandidatePromotion(failure_id="F01")))
    assert raised.value.status_code == 400
    assert "proposed" in str(raised.value.detail)


def test_an_accepted_candidate_records_the_entry_it_became(store: _Store) -> None:
    created = _run(create_candidate(_report(), realm_id="demo"))
    _run(decide_candidate(created["candidate_id"], CandidateDecision(status="accepted")))
    promoted = _run(record_promotion(created["candidate_id"], CandidatePromotion(failure_id="F01")))
    assert promoted["promoted_to"] == "F01"


def test_an_unknown_candidate_is_a_refusal_and_not_a_new_one(store: _Store) -> None:
    with pytest.raises(HTTPException) as raised:
        _run(decide_candidate("C0badbad", CandidateDecision(status="accepted")))
    assert raised.value.status_code == 404


def test_the_literal_paths_are_declared_before_the_one_that_matches_anything() -> None:
    """FastAPI takes the first route that matches, so `/atlas/{failure_id}`
    declared above these would answer "the failure whose id is candidates"
    and return a 404 for the whole mechanism. The same trap caught
    DELETE /experiments/baseline once already, which is why it is a test.
    """
    from services.api_gateway.routers.atlas import router

    paths = [route.path for route in router.routes]  # type: ignore[attr-defined]
    catch_all = paths.index("/atlas/{failure_id}")
    for literal in ("/atlas/candidates", "/atlas/signals"):
        assert paths.index(literal) < catch_all, f"{literal} is declared after the catch-all"


# ── the scaffold ──────────────────────────────────────────────────────────────

def _reported() -> dict[str, Any]:
    return {
        "candidate_id": "C1a2b3c4d",
        "created_at": "2026-09-06T10:00:00Z",
        "observed_on": "run 9da7259a",
        "title": "The reranker drops the only fragment that answers",
        "looked_like": "Retrieval found the right section and the answer never used it.",
        "suspected_signal": "pre_rerank_recall_at_k",
    }


def test_the_scaffold_carries_over_what_the_reporter_wrote() -> None:
    from tools.atlas_report import scaffold

    drafted = scaffold(_reported())
    assert "The reranker drops the only fragment that answers" in drafted
    assert "run 9da7259a" in drafted
    assert "pre_rerank_recall_at_k" in drafted


def test_the_scaffold_decides_nothing_a_person_has_to_decide() -> None:
    """A scaffold that guessed the severity, the coordinates and above all
    the detection would be a way of adding an unproven claim to the
    catalogue with less typing, which is the one thing this prevents."""
    from tools.atlas_report import scaffold

    drafted = scaffold(_reported())
    assert 'detection="none"' in drafted
    assert 'severity=Severity(0, 0, 0)' in drafted
    assert 'applies_when=()' in drafted
    assert drafted.count("DECIDE") >= 6


def test_the_scaffold_takes_an_identifier_no_entry_holds() -> None:
    """An identifier is issued once and never reused, so a draft carrying a
    live one would collide the moment it was pasted."""
    from core.eval.atlas import FAILURES
    from tools.atlas_report import _next_free_id

    assert _next_free_id() not in {f.id for f in FAILURES}


def test_the_scaffold_says_a_bait_needs_both_halves() -> None:
    from tools.atlas_report import scaffold

    drafted = scaffold(_reported())
    assert "silent" in drafted and "fire" in drafted
