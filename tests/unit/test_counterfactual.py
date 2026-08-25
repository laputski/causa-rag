"""core/eval/counterfactual.py —.

Phase 2 says which work pays off most; it cannot say by how much. These
tests pin the claim that makes "how much" answerable without a second run:
if the run recorded where the expected source actually sat, recall at any
smaller cut-off is arithmetic rather than retrieval.
"""
from __future__ import annotations

from core.eval.counterfactual import (
    context_size_curve,
    first_hit_rank,
    payoff_of_context_size,
    recommend_context_size,
)


def _ref(source_code: str, article_no: str) -> dict:
    return {"source_code": source_code, "article_no": article_no, "doc_id": "d"}


# ── Where the expected source actually sat ──────────────────────────────


def test_the_rank_is_one_based_so_it_reads_as_a_position() -> None:
    ranked = [_ref("S", "9"), _ref("S", "1"), _ref("S", "3")]
    assert first_hit_rank(["S/1"], ranked) == 2


def test_the_first_matching_entry_wins_when_several_are_expected() -> None:
    ranked = [_ref("S", "9"), _ref("S", "3"), _ref("S", "1")]
    assert first_hit_rank(["S/1", "S/3"], ranked) == 2


def test_never_present_is_none_rather_than_a_large_rank() -> None:
    """No cut-off would have helped, which is a different statement from
    "the cut-off was too small" — and reporting it as a big number would
    make a curve show a payoff that does not exist."""
    assert first_hit_rank(["S/1"], [_ref("S", "9")]) is None
    assert first_hit_rank(["S/1"], []) is None
    assert first_hit_rank([], [_ref("S", "1")]) is None


# ── The curve ───────────────────────────────────────────────────────────


def test_the_curve_counts_questions_answerable_at_each_cut_off() -> None:
    points = context_size_curve([1, 3, 7], current_k=5, max_k=8)
    by_k = {p.top_k: p.questions_hit for p in points}
    assert by_k[1] == 1
    assert by_k[3] == 2
    assert by_k[5] == 2
    assert by_k[7] == 3


def test_the_curve_reports_the_change_rather_than_leaving_subtraction() -> None:
    points = context_size_curve([1, 3, 7], current_k=5, max_k=8)
    by_k = {p.top_k: p.delta_vs_current for p in points}
    assert by_k[5] == 0
    assert by_k[7] == 1
    assert by_k[1] == -1


def test_a_question_never_retrieved_contributes_at_no_cut_off() -> None:
    points = context_size_curve([1, None, None], current_k=1, max_k=50)
    assert {p.questions_hit for p in points} == {1}


def test_every_integer_is_evaluated_not_only_round_numbers() -> None:
    """The useful value is usually just past a cluster of ranks, and
    rounding to 10/20/50 steps over it."""
    assert [p.top_k for p in context_size_curve([2], current_k=1, max_k=4)] == [1, 2, 3, 4]


# ── Payoff of one proposal ──────────────────────────────────────────────


def test_a_wider_cut_off_reports_what_it_would_close() -> None:
    assert payoff_of_context_size([2, 8, 12], current_k=5, proposed_k=12) == 2


def test_a_wider_cut_off_can_never_lose_a_question() -> None:
    assert payoff_of_context_size([2, 8], current_k=5, proposed_k=20) >= 0


def test_a_narrower_proposal_reports_the_loss_which_is_the_point_of_asking() -> None:
    assert payoff_of_context_size([2, 8], current_k=10, proposed_k=5) == -1


# ── Advice ──────────────────────────────────────────────────────────────


def test_the_smallest_cut_off_capturing_the_whole_payoff_is_recommended() -> None:
    """Every extra chunk costs tokens, latency and a chance of distracting
    the generator, so a value buying nothing beyond a smaller one is
    strictly worse."""
    advice = recommend_context_size([7, 9, 30], current_k=5, max_k=50)
    assert advice is not None
    assert advice.recommended_k == 30
    assert advice.questions_gained == 3


def test_nothing_to_gain_yields_no_advice_rather_than_the_status_quo() -> None:
    assert recommend_context_size([1, 2, 3], current_k=5, max_k=50) is None
    assert recommend_context_size([], current_k=5, max_k=50) is None


def test_questions_beyond_the_observed_window_are_named_not_hidden() -> None:
    """Otherwise a reader is left thinking the remaining failures are a
    tuning problem, when no cut-off would reach them."""
    advice = recommend_context_size([7, None, None], current_k=5, max_k=50)
    assert advice is not None
    assert advice.questions_gained == 1
    assert advice.unreachable == 2


def test_a_rank_past_the_observed_window_is_not_recommended() -> None:
    # Recommending past what was actually observed would be extrapolation,
    # and the whole point of this answer is that it is measured.
    advice = recommend_context_size([7, 80], current_k=5, max_k=50)
    assert advice is not None
    assert advice.recommended_k == 7
    assert advice.unreachable == 1
