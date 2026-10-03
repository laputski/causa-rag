"""core/eval/funnel.py — per-question layer attribution, Eval Measurement
Trustworthiness Phase 1. Fixtures use the exact pattern found in run
44329138's analysis (recall=0, similarity high) to lock in the combined
check that motivated this module.
"""
from __future__ import annotations

from core.eval.funnel import diagnose_question


def test_uncovered_question_is_not_applicable() -> None:
    v = diagnose_question("uncovered", metrics={"correct_refusal": 1.0})
    assert v.layer == "not_applicable"
    assert "uncovered" in v.detail


def test_out_of_scope_question_is_not_applicable() -> None:
    v = diagnose_question("out_of_scope", metrics={"correct_refusal": 1.0})
    assert v.layer == "not_applicable"
    assert "out_of_scope" in v.detail


def test_suspected_ungrounded_answer_pattern_from_real_run_44329138() -> None:
    """The exact pattern found in 12/75 answerable questions of run
    44329138: recall_at_k=0.0, answer_similarity=0.92 — must NOT be read as
    "retrieval bad, generation fine" by isolated per-layer checks."""
    v = diagnose_question(
        "answerable",
        metrics={"retrieval_recall_at_k": 0.0, "answer_similarity": 0.92, "correct_refusal": 1.0},
    )
    assert v.layer == "suspected_ungrounded_answer"


def test_low_recall_low_similarity_is_plain_retrieval_failure() -> None:
    """Low recall WITHOUT a suspiciously good answer is just retrieval
    failing honestly — no combined-check false alarm."""
    v = diagnose_question(
        "answerable",
        metrics={"retrieval_recall_at_k": 0.0, "answer_similarity": 0.2, "correct_refusal": 1.0},
    )
    assert v.layer == "retrieval"


def test_rerank_attributed_when_pre_rerank_recall_was_good() -> None:
    """Retrieval found it (pre-rerank recall high), but the reranker
    demoted it out of top-k (post-rerank recall low) — must attribute to
    rerank, not retrieval."""
    v = diagnose_question(
        "answerable",
        metrics={"retrieval_recall_at_k": 0.0, "answer_similarity": 0.3, "correct_refusal": 1.0},
        pre_rerank_recall_at_k=1.0,
    )
    assert v.layer == "rerank"


def test_low_recall_without_pre_rerank_data_falls_back_to_retrieval() -> None:
    """No reranker ran (pre_rerank_recall_at_k=None) — can't distinguish
    retrieval from rerank, so must default to the honest combined bucket,
    not invent a rerank verdict without evidence."""
    v = diagnose_question(
        "answerable",
        metrics={"retrieval_recall_at_k": 0.0, "answer_similarity": 0.3, "correct_refusal": 1.0},
        pre_rerank_recall_at_k=None,
    )
    assert v.layer == "retrieval"


def test_good_recall_but_not_grounded_in_correct_chunk_is_generation() -> None:
    v = diagnose_question(
        "answerable",
        metrics={
            "retrieval_recall_at_k": 1.0, "answer_similarity": 0.8,
            "grounded_in_correct_source": 0.1, "correct_refusal": 1.0,
        },
    )
    assert v.layer == "generation"


def test_good_recall_low_similarity_is_generation() -> None:
    v = diagnose_question(
        "answerable",
        metrics={"retrieval_recall_at_k": 1.0, "answer_similarity": 0.1, "correct_refusal": 1.0},
    )
    assert v.layer == "generation"


def test_everything_good_is_ok() -> None:
    v = diagnose_question(
        "answerable",
        metrics={
            "retrieval_recall_at_k": 1.0, "answer_similarity": 0.85,
            "grounded_in_correct_source": 0.9, "correct_refusal": 1.0,
        },
    )
    assert v.layer == "ok"


def test_missing_metrics_does_not_crash() -> None:
    """Metrics dict missing keys entirely (e.g. legacy pre-Phase-0 run) must
    not raise — degrade to 'ok' rather than crash the whole diagnosis."""
    v = diagnose_question("answerable", metrics={})
    assert v.layer == "ok"


def test_to_dict_shape() -> None:
    v = diagnose_question("uncovered", metrics={})
    d = v.to_dict()
    assert set(d) == {"layer", "detail"}


# ── A source one half found and the merge lost ────────────────────────────────

def test_a_source_a_half_found_and_the_merge_lost_is_named_as_such() -> None:
    """Measured on a stored run of fifty questions: half of them had the
    source they needed in one half's own list and not in the merged one, and
    every one of them was reported as "retrieval never found it"."""
    v = diagnose_question("answerable", metrics={
        "retrieval_recall_at_k": 0.0, "answer_similarity": 0.2, "kept_through_fusion": 0.0})
    assert v.layer == "fusion"


def test_a_source_the_merge_kept_and_the_cut_dropped_is_still_retrieval() -> None:
    v = diagnose_question("answerable", metrics={
        "retrieval_recall_at_k": 0.0, "answer_similarity": 0.2, "kept_through_fusion": 1.0})
    assert v.layer == "retrieval"


def test_a_run_that_recorded_no_halves_is_diagnosed_as_before() -> None:
    v = diagnose_question("answerable", metrics={"retrieval_recall_at_k": 0.0, "answer_similarity": 0.2})
    assert v.layer == "retrieval"


def test_a_confident_answer_without_its_source_is_still_suspected_first() -> None:
    """The merge losing the source does not make the answer grounded."""
    v = diagnose_question("answerable", metrics={
        "retrieval_recall_at_k": 0.0, "answer_similarity": 0.9, "kept_through_fusion": 0.0})
    assert v.layer == "suspected_ungrounded_answer"
