"""core/eval/drift.py.

Every failure the platform sees comes from a golden set, so it only ever sees
questions somebody thought of in advance. These tests pin the measurement
that finds the questions nobody did.
"""
from __future__ import annotations

import pytest

from core.eval.drift import coverage_report, nearest_similarities, uncovered_count


def test_coverage_is_the_nearest_golden_question_not_the_average() -> None:
    """A question is covered when *some* golden question resembles it.
    Averaging over a large golden set would acme every value toward zero
    and report a well-covered corpus as uncovered."""
    production = [[1.0, 0.0]]
    golden = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    assert nearest_similarities(production, golden)[0] == 1.0


def test_an_empty_golden_set_covers_nothing_rather_than_erroring() -> None:
    # "No golden set yet" is a real state a new Realm passes through.
    assert nearest_similarities([[1.0, 0.0]], []) == [0.0]


def test_uncovered_questions_come_worst_first() -> None:
    """The question furthest from anything tested is the one most worth
    turning into a test, and a reader skimming reads the first few."""
    report = coverage_report(
        ["near", "far"], [0.55, 0.10], n_golden=3, threshold=0.6,
    )
    assert report.uncovered == ("far", "near")


def test_the_share_counts_every_uncovered_question_not_the_capped_list() -> None:
    """A cap that silently improved the number reported beside it would be
    the worst kind of summary."""
    nearest = [0.1] * 50
    report = coverage_report(["q"] * 50, nearest, n_golden=1, threshold=0.6, max_examples=5)
    assert len(report.uncovered) == 5
    assert uncovered_count(nearest, 0.6) == 50


def test_the_distribution_is_reported_beside_the_share() -> None:
    """A set covering most questions poorly and one covering most well but
    missing a few produce similar shares and call for opposite work."""
    report = coverage_report(["a", "b", "c"], [0.1, 0.5, 0.9], n_golden=2)
    assert len(report.deciles) == 11
    assert report.deciles[0] == 0.1


def test_a_fully_covered_stream_reports_nothing_to_do() -> None:
    report = coverage_report(["a", "b"], [0.95, 0.88], n_golden=5, threshold=0.6)
    assert report.uncovered == ()
    assert report.uncovered_share == 0.0


def test_the_threshold_travels_so_a_reader_can_disagree_with_it() -> None:
    assert coverage_report(["a"], [0.5], n_golden=1, threshold=0.7).to_dict()["threshold"] == 0.7


# ── Found by re-verification: a missing dataset silently became a stub ───

async def test_coverage_refuses_a_dataset_that_does_not_exist() -> None:
    """`_load_dataset` falls back to a five-question stub of invented text
    when a name matches nothing. Coverage then compared real traffic against
    fabricated questions and reported "100% uncovered" — a confidently wrong
    number that reads as a finding rather than as a caller mistake."""
    from unittest.mock import AsyncMock, patch

    from fastapi import HTTPException

    from eval.dataset import make_stub_dataset
    from services.api_gateway.routers import production

    with patch("adapters.mongodb.find_many", AsyncMock(return_value=[{"query": "q"}])), \
         patch("services.api_gateway.routers.experiments._load_dataset",
               AsyncMock(return_value=make_stub_dataset(n=5))):
        with pytest.raises(HTTPException) as exc:
            await production.golden_set_coverage(
                realm_id="r", corpus_id="c", dataset_name="nosuch.jsonl",
            )
    assert exc.value.status_code == 404


# ── Found live: a deleted question left its trace permanently unpromotable ──


async def test_a_trace_whose_question_was_deleted_can_be_promoted_again() -> None:
    """The mark on a trace records that a promotion happened once. It cannot
    know that somebody later deleted the question, and treating it as proof of
    a live question turned an ordinary deletion into a permanent refusal — the
    production failure became untestable forever, with no way back short of
    editing the database by hand.

    Found on the platform's own data: the question created while verifying
    phase 7 was removed afterwards, and its trace kept refusing.
    """
    from unittest.mock import AsyncMock, patch

    from services.api_gateway.routers import production

    trace = {"trace_id": "t1", "query": "a question", "promoted_question_id": "gone42"}
    added = {"name": "Cosmos", "count": 51, "questions": [{"id": "new7", "trace_id": "t1"}]}

    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=[trace, None])), \
         patch("adapters.mongodb.update_one", AsyncMock()), \
         patch("services.api_gateway.routers.datasets.add_question",
               AsyncMock(return_value=added)):
        result = await production.promote_trace(production.PromoteRequest(
            trace_id="t1", dataset_id="ds1", reference_answer="an answer", realm_id="r",
        ))
    assert result["question_id"] == "new7"


async def test_a_trace_whose_question_still_exists_is_still_refused() -> None:
    """The negative control. Without it the fix above would read as "always
    allow", which is the duplicate-question defect this endpoint was built to
    prevent."""
    from unittest.mock import AsyncMock, patch

    from fastapi import HTTPException

    from services.api_gateway.routers import production

    trace = {"trace_id": "t1", "query": "a question", "promoted_question_id": "alive9"}
    dataset = {"id": "ds1", "questions": [{"id": "alive9"}]}

    with patch("adapters.mongodb.find_one", AsyncMock(side_effect=[trace, dataset])):
        with pytest.raises(HTTPException) as exc:
            await production.promote_trace(production.PromoteRequest(
                trace_id="t1", dataset_id="ds1", reference_answer="an answer", realm_id="r",
            ))
    assert exc.value.status_code == 409
    assert "alive9" in exc.value.detail


async def test_unreachable_storage_refuses_rather_than_risking_a_duplicate() -> None:
    """A storage failure answers "the question is still there". Guessing the
    other way would let a reader create a duplicate golden question during an
    outage, and a duplicate is harder to notice and undo than a refusal."""
    from unittest.mock import AsyncMock, patch

    from fastapi import HTTPException

    from services.api_gateway.routers import production

    trace = {"trace_id": "t1", "query": "a question", "promoted_question_id": "maybe1"}

    with patch("adapters.mongodb.find_one",
               AsyncMock(side_effect=[trace, RuntimeError("mongo down")])):
        with pytest.raises(HTTPException) as exc:
            await production.promote_trace(production.PromoteRequest(
                trace_id="t1", dataset_id="ds1", reference_answer="an answer", realm_id="r",
            ))
    assert exc.value.status_code == 409


async def test_the_list_says_whether_a_promotion_still_points_at_anything() -> None:
    """Two states the page must not collapse: "became a test" and "became a
    test that no longer exists". Rendering them identically is what left a
    dead reference on screen with no way to act on it."""
    from unittest.mock import AsyncMock, patch

    from services.api_gateway.routers import production

    traces = [
        {"trace_id": "t1", "promoted_question_id": "alive9", "created_at": "2026-08-01"},
        {"trace_id": "t2", "promoted_question_id": "gone42", "created_at": "2026-08-02"},
        {"trace_id": "t3", "promoted_question_id": None, "created_at": "2026-08-03"},
    ]
    datasets = [{"id": "ds1", "questions": [{"id": "alive9"}]}]

    with patch("adapters.mongodb.find_many", AsyncMock(side_effect=[traces, datasets])):
        rows = await production.list_traces(realm_id="r", corpus_id="c")

    by_id = {r["trace_id"]: r for r in rows}
    assert by_id["t1"]["promotion_live"] is True
    assert by_id["t2"]["promotion_live"] is False
    assert by_id["t3"]["promotion_live"] is False
