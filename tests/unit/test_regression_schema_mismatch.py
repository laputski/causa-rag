"""core/eval/regression.py — cross-schema mismatch detection added for Eval
Measurement Trustworthiness, Phase 0. Without this, comparing a run on the
new metric schema (retrieval_recall_at_k, ...) against a baseline still on
the old one (faithfulness, ...) silently produces zero deltas and reads as
"passed" — there's nothing comparable, not nothing wrong.
"""
from __future__ import annotations

from core.eval.regression import compare


def test_disjoint_schemas_fail_with_explicit_violation() -> None:
    report = compare(
        current_metrics={"retrieval_recall_at_k": 0.8, "correct_refusal": 0.9},
        baseline_metrics={"faithfulness": 0.04, "answer_relevancy": 0.13},
    )
    assert report.passed is False
    assert report.deltas == []
    assert len(report.violations) == 1
    assert "disjoint" in report.violations[0]


def test_overlapping_schemas_compare_normally() -> None:
    report = compare(
        current_metrics={"retrieval_recall_at_k": 0.5, "correct_refusal": 0.9},
        baseline_metrics={"retrieval_recall_at_k": 0.8},
    )
    assert len(report.deltas) == 1
    assert report.deltas[0].metric == "retrieval_recall_at_k"


def test_empty_current_does_not_trigger_mismatch() -> None:
    report = compare(current_metrics={}, baseline_metrics={"retrieval_recall_at_k": 0.8})
    assert report.passed is True
    assert report.violations == []


def test_empty_baseline_does_not_trigger_mismatch() -> None:
    report = compare(current_metrics={"retrieval_recall_at_k": 0.5}, baseline_metrics={})
    assert report.passed is True
    assert report.violations == []


def test_a_metric_the_baseline_pins_and_the_run_lost_is_named() -> None:
    """A guard that stops guarding without saying so.

    The baseline pins four metrics, the run produces three, and the report
    comes back green over the shorter list. Nothing was wrong with the
    three; the fourth simply stopped being checked, and until now that was
    indistinguishable from passing.
    """
    report = compare(
        current_metrics={"retrieval_recall_at_k": 0.9, "retrieval_precision_at_k": 0.4},
        baseline_metrics={"retrieval_recall_at_k": 0.9, "retrieval_precision_at_k": 0.4,
                          "correct_refusal": 0.8, "answer_similarity": 0.7},
    )

    assert report.passed is True
    assert sorted(report.unchecked) == ["answer_similarity", "correct_refusal"]
    assert "unchecked" in report.to_dict()


def test_nothing_is_unchecked_when_the_run_covers_the_baseline() -> None:
    """The bait. A list that is never empty stops being read."""
    report = compare(
        current_metrics={"retrieval_recall_at_k": 0.9},
        baseline_metrics={"retrieval_recall_at_k": 0.9},
    )
    assert report.unchecked == []


def test_an_unchecked_metric_does_not_fail_the_gate_on_its_own() -> None:
    """Deciding whether a vanished metric is a problem belongs to the
    caller: a retrieval-only run has no answer to score, and that is not a
    regression."""
    report = compare(
        current_metrics={"retrieval_recall_at_k": 0.95},
        baseline_metrics={"retrieval_recall_at_k": 0.9, "answer_similarity": 0.7},
    )
    assert report.passed is True
    assert report.unchecked == ["answer_similarity"]
