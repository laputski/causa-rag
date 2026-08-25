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
