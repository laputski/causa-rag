"""A merge that loses the sources its halves found is reported on the run.

Found live. A stored run of fifty questions lost twenty-five sources between
a half's own list and the merged one, and the verdict written for that case
never appeared: thirty-six of the fifty were already named "an answer that
resembles the right one without its source", which is checked first and is
true of them too. A cause on the retrieval side that the per-question verdict
cannot reach is said about the run instead.

Measured beside it: a healthy hybrid run on the proving ground lost none of
seventeen.
"""
from __future__ import annotations

from core.eval.detectors import detect_merge_lost_what_a_half_found, run_detectors
from core.eval.funnel import diagnose_question


def _run(kept: list[float | None]) -> dict:
    return {"config": {"pipeline_id": "hybrid_rrf"}, "question_results": [
        {"question_id": f"q{i}", "metrics": ({} if k is None else {"kept_through_fusion": k})}
        for i, k in enumerate(kept)]}


def test_half_of_the_found_sources_lost_is_reported() -> None:
    finding = detect_merge_lost_what_a_half_found(_run([0.0] * 25 + [1.0] * 13 + [None] * 12))
    assert finding is not None
    assert finding.params == {"lost": 25, "found_by_a_half": 38}
    assert "merge_lost_what_a_half_found" in {d.id for d in run_detectors(_run([0.0] * 25 + [1.0] * 13))}


def test_a_merge_that_keeps_what_it_is_given_is_silent() -> None:
    assert detect_merge_lost_what_a_half_found(_run([1.0] * 17 + [None] * 2)) is None


def test_one_loss_among_many_is_what_a_merge_does() -> None:
    """A merge ranks, and a source a half ranked last may fall below the cut
    once. That is not a merge working against its halves."""
    assert detect_merge_lost_what_a_half_found(_run([0.0] + [1.0] * 16)) is None


def test_too_few_questions_to_say_anything() -> None:
    assert detect_merge_lost_what_a_half_found(_run([0.0, 0.0])) is None


def test_a_run_that_recorded_no_halves_is_silent() -> None:
    assert detect_merge_lost_what_a_half_found(_run([None] * 20)) is None


def test_the_verdict_that_is_checked_first_still_says_what_the_merge_did() -> None:
    """The question keeps its one layer, and its words carry the cause the
    layer has no room for."""
    verdict = diagnose_question("answerable", {
        "retrieval_recall_at_k": 0.0, "answer_similarity": 0.9, "kept_through_fusion": 0.0})
    assert verdict.layer == "suspected_ungrounded_answer"
    assert "merged list" in verdict.detail
    kept = diagnose_question("answerable", {
        "retrieval_recall_at_k": 0.0, "answer_similarity": 0.9, "kept_through_fusion": 1.0})
    assert "merged list" not in kept.detail
