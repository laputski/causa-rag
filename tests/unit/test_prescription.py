"""core/eval/prescription.py and trace_completeness.py — phase 4.

The exit criterion of the phase is that the owner of a system the platform
does not own receives a document to work from rather than a score. These
tests pin what makes it a document: it asserts nothing the run did not
measure, it says what could not be checked at all, and its verification list
is what acceptance is later judged over.
"""
from __future__ import annotations

from core.eval.prescription import (
    build_prescription,
    judge_acceptance,
    render_markdown,
)
from core.eval.trace_completeness import assess_trace_completeness, diagnosis_depth


def _q(qid: str, **extra) -> dict:
    base = {"question_id": qid, "question": f"question {qid}"}
    base.update(extra)
    return base


def _payload(**overrides) -> dict:
    base = {
        "run_id": "run1",
        "dataset_name": "Cosmos 1",
        "aggregate_metrics": {"retrieval_recall_at_k": 0.3},
        "question_results": [
            _q("q1", source_refs=[{"chunk_id": "c"}], stage_trace={"total_ms": 1}),
            _q("q2", source_refs=[{"chunk_id": "c"}], stage_trace={"total_ms": 1}),
        ],
        "fix_tasks": [{
            "cause": "data_missing", "lever": "ingest", "entity": "DOC1",
            "questions": 2, "question_ids": ["q1", "q2"],
        }],
    }
    base.update(overrides)
    return base


# ── What could not be checked ────────────────────────────────


def test_a_missing_capability_names_the_verdict_it_makes_unreachable() -> None:
    """A diagnostic that cannot run and one that ran and found nothing are
    both absent from a report, so only naming the gap separates them."""
    gaps = {g.field: g for g in assess_trace_completeness([_q("q1", source_refs=[{"chunk_id": "c"}])])}
    assert "pre_rerank_source_refs" in gaps
    assert "rerank" in gaps["pre_rerank_source_refs"].unavailable.lower()


def test_every_gap_names_what_the_owner_would_change() -> None:
    # A gap nobody can act on is a complaint rather than a finding.
    for gap in assess_trace_completeness([_q("q1")]):
        assert gap.remedy


def test_a_run_with_no_questions_reports_no_gaps() -> None:
    """Nothing was observed, so nothing is missing. Reporting every gap for
    an empty run would drown the real ones the first time anyone looked."""
    assert assess_trace_completeness([]) == []
    assert diagnosis_depth([]) == "unknown"


def test_depth_is_named_rather_than_scored() -> None:
    # A percentage would invite comparing two systems by it, and a system
    # reporting sources but no trace is not "more complete" than one
    # reporting a trace but no sources.
    assert diagnosis_depth([_q("q1")]) == "answer_only"
    assert diagnosis_depth([_q("q1", source_refs=[{"chunk_id": "c"}])]) == "partial"
    assert diagnosis_depth([_q("q1", source_refs=[{"c": 1}], stage_trace={"t": 1},
                               pre_rerank_source_refs=[{"c": 1}],
                               candidate_source_refs=[{"c": 1}])]) == "full"


# ── The document ─────────────────────────────────────────────


def test_the_prescription_asserts_nothing_the_run_did_not_measure() -> None:
    """Assembly only. A prescription able to assert something unmeasured
    would be a recommendation dressed as a finding."""
    p = build_prescription(_payload(context_size_advice=None))
    assert p.context_size_advice is None
    assert p.metrics == {"retrieval_recall_at_k": 0.3}


def test_each_fix_carries_examples_and_the_full_verification_list() -> None:
    fix = build_prescription(_payload()).fixes[0]
    assert fix.entity == "DOC1"
    assert fix.examples  # enough to show the pattern
    # Every question, not a sample: checking a sample would let a partial
    # fix pass as a whole one.
    assert fix.verification_question_ids == ("q1", "q2")


def test_the_document_says_what_could_not_be_checked_before_what_was_found() -> None:
    """A reader who does not know what was invisible reads a short findings
    list as good news."""
    text = render_markdown(build_prescription(_payload()))
    assert text.index("could not check") < text.index("What to fix")


def test_a_run_with_no_diagnosed_failures_says_so_rather_than_inventing_work() -> None:
    text = render_markdown(build_prescription(_payload(fix_tasks=[])))
    assert "No work assigned" in text


def test_a_measured_counterfactual_is_carried_verbatim_into_the_document() -> None:
    text = render_markdown(build_prescription(_payload(context_size_advice={
        "current_k": 5, "recommended_k": 39, "questions_gained": 23, "unreachable": 12,
    })))
    assert "from 5 to 39" in text
    assert "23" in text
    assert "12" in text


# ── Works 4.2 and 4.3 — acceptance ──────────────────────────────────────


def _funnel(qid: str, layer: str) -> dict:
    return {"question_id": qid, "funnel": {"layer": layer, "detail": ""}}


def test_acceptance_needs_every_named_question_to_pass() -> None:
    verdict = judge_acceptance(
        ["q1", "q2"],
        before=[_funnel("q1", "retrieval"), _funnel("q2", "retrieval")],
        after=[_funnel("q1", "ok"), _funnel("q2", "ok")],
    )
    assert verdict.accepted is True
    assert set(verdict.fixed) == {"q1", "q2"}


def test_closing_some_while_breaking_others_is_not_acceptance() -> None:
    """The owner was asked for a specific outcome and did not deliver it, so
    this is not a partial success at this boundary."""
    verdict = judge_acceptance(
        ["q1", "q2"],
        before=[_funnel("q1", "retrieval"), _funnel("q2", "ok")],
        after=[_funnel("q1", "ok"), _funnel("q2", "retrieval")],
    )
    assert verdict.accepted is False
    assert verdict.fixed == ("q1",)
    assert verdict.regressed == ("q2",)


def test_a_question_missing_from_a_run_blocks_acceptance() -> None:
    """Treating it as passed would let a shrunken dataset read as a
    completed prescription."""
    verdict = judge_acceptance(
        ["q1", "q2"],
        before=[_funnel("q1", "retrieval"), _funnel("q2", "retrieval")],
        after=[_funnel("q1", "ok")],
    )
    assert verdict.accepted is False
    assert verdict.unchecked == ("q2",)


def test_only_the_named_questions_are_judged() -> None:
    # The acceptance set is fixed when the prescription is written; a
    # question outside it neither helps nor hurts.
    verdict = judge_acceptance(
        ["q1"],
        before=[_funnel("q1", "retrieval"), _funnel("other", "ok")],
        after=[_funnel("q1", "ok"), _funnel("other", "retrieval")],
    )
    assert verdict.accepted is True


def test_the_summary_states_all_four_counts() -> None:
    verdict = judge_acceptance(["q1"], before=[_funnel("q1", "retrieval")], after=[])
    assert "unchecked 1" in verdict.summary
