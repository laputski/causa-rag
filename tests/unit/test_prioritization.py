"""core/eval/prioritization.py —.

Phase 1 gave every failed question a cause, which is still a list of
complaints. These tests pin the step that makes it a list of work: failures
that share a cause and a document become one task, and tasks are comparable
by how many questions each would close.
"""
from __future__ import annotations

from core.eval.prioritization import (
    clusterable_share,
    entity_for_ref,
    group_failures,
)


def _failed(qid: str, cause: str, refs: list[str], lever: str = "ingest") -> dict:
    return {
        "question_id": qid,
        "root_cause": {"cause": cause, "lever": lever, "evidence": {"refs": refs}},
    }


# ── The entity a task is about ──────────────────────────────────────────


def test_the_document_is_the_entity_not_the_article() -> None:
    """Grouping by the whole ref would give every question its own task,
    which is the list of complaints this module replaces."""
    assert entity_for_ref("SRC001/44") == "SRC001"
    assert entity_for_ref("SRC001/12") == "SRC001"


def test_every_ref_id_form_yields_a_document() -> None:
    assert entity_for_ref("D1#Ch 2 > Art 10") == "D1"
    assert entity_for_ref("S#Section 3") == "S"
    assert entity_for_ref("bare_doc_id") == "bare_doc_id"


# ── Grouping ────────────────────────────────────────────────────────────


def test_questions_naming_the_same_document_and_cause_become_one_task() -> None:
    tasks = group_failures([
        _failed("q1", "data_missing", ["HK/44"]),
        _failed("q2", "data_missing", ["HK/12"]),
        _failed("q3", "data_missing", ["HK/8"]),
    ])
    assert len(tasks) == 1
    assert tasks[0].entity == "HK"
    assert tasks[0].questions == 3


def test_the_same_document_under_different_causes_stays_two_tasks() -> None:
    """Loading a missing document and re-chunking an indexed one are
    different work, so merging them would produce a task nobody can do."""
    tasks = group_failures([
        _failed("q1", "data_missing", ["HK/44"]),
        _failed("q2", "chunking", ["HK/12"], lever="chunking"),
    ])
    assert len(tasks) == 2


def test_tasks_are_ordered_by_how_many_questions_each_would_close() -> None:
    tasks = group_failures([
        _failed("q1", "data_missing", ["SMALL/1"]),
        _failed("q2", "data_missing", ["BIG/1"]),
        _failed("q3", "data_missing", ["BIG/2"]),
        _failed("q4", "data_missing", ["BIG/3"]),
    ])
    assert [t.entity for t in tasks] == ["BIG", "SMALL"]


def test_the_order_is_stable_when_payoffs_tie() -> None:
    # An ordering that shuffles between reads of the same run is one a
    # reader cannot trust.
    rows = [_failed("q1", "data_missing", ["B/1"]), _failed("q2", "data_missing", ["A/1"])]
    assert [t.entity for t in group_failures(rows)] == [t.entity for t in group_failures(rows)]
    assert [t.entity for t in group_failures(rows)] == ["A", "B"]


def test_a_question_naming_two_documents_joins_both_tasks() -> None:
    """It really would be closed by fixing either, so counting it once would
    understate the payoff of whichever task the reader then skipped."""
    tasks = group_failures([_failed("q1", "data_missing", ["A/1", "B/1"])])
    assert sorted(t.entity for t in tasks) == ["A", "B"]
    assert all(t.questions == 1 for t in tasks)


def test_a_failure_with_no_refs_is_still_accounted_for() -> None:
    # Dropping it would hide a failure from the very list meant to account
    # for all of them.
    tasks = group_failures([_failed("q1", "unknown", [], lever="verify_index")])
    assert len(tasks) == 1
    assert tasks[0].questions == 1


def test_healthy_questions_produce_no_tasks() -> None:
    assert group_failures([{"question_id": "q1"}, {"question_id": "q2", "root_cause": None}]) == []


# ── The share that decides whether phase 6 starts ────────────


def test_the_share_counts_questions_that_share_a_cause_with_another() -> None:
    tasks = group_failures([
        _failed("q1", "data_missing", ["BIG/1"]),
        _failed("q2", "data_missing", ["BIG/2"]),
        _failed("q3", "data_missing", ["LONE/1"]),
    ])
    assert clusterable_share(tasks) == 2 / 3


def test_entirely_isolated_failures_give_a_share_of_zero() -> None:
    """The case that says generalisation has nothing to work with, so the
    effort belongs in the data instead of in the last mile."""
    tasks = group_failures([
        _failed("q1", "data_missing", ["A/1"]),
        _failed("q2", "data_missing", ["B/1"]),
    ])
    assert clusterable_share(tasks) == 0.0


def test_a_run_with_no_failures_reports_zero_rather_than_a_verdict() -> None:
    assert clusterable_share([]) == 0.0


def test_the_share_is_measured_over_questions_not_over_tasks() -> None:
    # A task-based ratio would reward splitting the same failures more
    # finely, which changes the number without changing the world.
    tasks = group_failures([
        _failed("q1", "data_missing", ["BIG/1"]),
        _failed("q2", "data_missing", ["BIG/2"]),
        _failed("q3", "data_missing", ["BIG/3"]),
        _failed("q4", "data_missing", ["LONE/1"]),
    ])
    assert clusterable_share(tasks) == 0.75


# ── Runs stored before phase 2 ──────────────────────────────────────────


def test_an_older_verdict_still_names_the_document_to_fix() -> None:
    """Runs stored between phase 1 and phase 2 carry only the cause-specific
    ref list. Ignoring it would leave their task list showing an anonymous
    group, losing most of what makes the list useful. Found by reading a
    real run rather than by a test."""
    tasks = group_failures([{
        "question_id": "q1",
        "root_cause": {
            "cause": "data_missing", "lever": "ingest",
            "evidence": {"absent_refs": ["RC/1"]},
        },
    }])
    assert tasks[0].entity == "RC"


def test_an_unverifiable_older_verdict_names_its_document_too() -> None:
    tasks = group_failures([{
        "question_id": "q1",
        "root_cause": {
            "cause": "unknown", "lever": "verify_index",
            "evidence": {"unknown_refs": ["D9#Ch 1"]},
        },
    }])
    assert tasks[0].entity == "D9"


def test_the_current_refs_field_wins_over_the_fallback() -> None:
    tasks = group_failures([{
        "question_id": "q1",
        "root_cause": {
            "cause": "data_missing", "lever": "ingest",
            "evidence": {"refs": ["NEW/1"], "absent_refs": ["OLD/1"]},
        },
    }])
    assert tasks[0].entity == "NEW"
