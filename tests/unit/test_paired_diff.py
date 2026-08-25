"""core/eval/regression.py:paired_diff — per-question before/after comparison. The aggregate-only `compare`/`check_gate`
answers "did the average move"; this answers "did any individual question
that used to pass now fail" — the check a targeted fix (pin, per-class
profile, ...) is judged against.
"""
from __future__ import annotations

from core.eval.regression import confirm_flips, paired_diff


def _qr(question_id: str, layer: str, **metrics: float) -> dict:
    return {
        "question_id": question_id,
        "metrics": metrics,
        "funnel": {"layer": layer, "detail": ""},
    }


def test_question_moving_from_not_ok_to_ok_is_fixed() -> None:
    before = [_qr("q1", "retrieval", retrieval_recall_at_k=0.1)]
    after = [_qr("q1", "ok", retrieval_recall_at_k=0.9)]
    report = paired_diff(before, after)
    assert report.fixed == ["q1"]
    assert report.flips == []
    assert report.unchanged == []


def test_question_moving_from_ok_to_not_ok_is_a_flip() -> None:
    before = [_qr("q1", "ok", retrieval_recall_at_k=0.9)]
    after = [_qr("q1", "generation", retrieval_recall_at_k=0.9)]
    report = paired_diff(before, after)
    assert report.flips == ["q1"]
    assert report.fixed == []


def test_question_ok_both_times_is_unchanged() -> None:
    before = [_qr("q1", "ok", retrieval_recall_at_k=0.9)]
    after = [_qr("q1", "ok", retrieval_recall_at_k=0.95)]
    report = paired_diff(before, after)
    assert report.unchanged == ["q1"]
    assert report.fixed == []
    assert report.flips == []


def test_not_applicable_counts_as_not_ok() -> None:
    # A question outside the diagnosable funnel (out_of_scope/uncovered)
    # moving to a real "ok" verdict still counts as fixed — and the reverse
    # still counts as a flip — since not_applicable is not "ok" either way.
    before = [_qr("q1", "not_applicable")]
    after = [_qr("q1", "ok", retrieval_recall_at_k=0.9)]
    assert paired_diff(before, after).fixed == ["q1"]

    before2 = [_qr("q1", "ok", retrieval_recall_at_k=0.9)]
    after2 = [_qr("q1", "not_applicable")]
    assert paired_diff(before2, after2).flips == ["q1"]


def test_question_missing_from_one_side_is_skipped_entirely() -> None:
    before = [_qr("q1", "ok", retrieval_recall_at_k=0.9), _qr("q2", "retrieval")]
    after = [_qr("q1", "ok", retrieval_recall_at_k=0.9)]  # q2 absent — dataset changed
    report = paired_diff(before, after)
    assert report.fixed == []
    assert report.flips == []
    assert report.unchanged == ["q1"]


def test_metric_deltas_computed_only_for_common_metrics() -> None:
    before = [_qr("q1", "ok", retrieval_recall_at_k=0.5, answer_similarity=0.6)]
    after = [_qr("q1", "ok", retrieval_recall_at_k=0.8, context_support=0.9)]
    report = paired_diff(before, after)
    assert report.metric_deltas == {"q1": {"retrieval_recall_at_k": 0.30000000000000004}}


def test_multiple_questions_sorted_output() -> None:
    before = [_qr("qb", "retrieval"), _qr("qa", "ok", retrieval_recall_at_k=0.9)]
    after = [_qr("qb", "ok", retrieval_recall_at_k=0.9), _qr("qa", "generation")]
    report = paired_diff(before, after)
    assert report.fixed == ["qb"]
    assert report.flips == ["qa"]


def test_to_dict_shape() -> None:
    report = paired_diff(
        [_qr("q1", "retrieval")], [_qr("q1", "ok", retrieval_recall_at_k=0.9)],
    )
    assert report.to_dict() == {
        "fixed": ["q1"], "flips": [], "unchanged": [],
        "metric_deltas": {},
    }


class TestConfirmFlips:
    """core/eval/regression.py:confirm_flips — resample-based noise filter
    for paired_diff's flips (the acceptance criterion
    left open when paired_diff itself shipped: "a noisy flip is extinguished
    on repeat run")."""

    def test_consistently_not_ok_confirms_a_real_flip(self) -> None:
        confirmed, noise = confirm_flips({"q1": [False, False, False]})
        assert confirmed == ["q1"]
        assert noise == []

    def test_majority_ok_on_resample_demotes_to_noise(self) -> None:
        # Original sample said not-ok (that's why it was a flip); both
        # resamples came back ok — the original was generation-metric noise.
        confirmed, noise = confirm_flips({"q1": [False, True, True]})
        assert confirmed == []
        assert noise == ["q1"]

    def test_majority_not_ok_stays_a_confirmed_flip_despite_one_ok_resample(self) -> None:
        confirmed, noise = confirm_flips({"q1": [False, False, True]})
        assert confirmed == ["q1"]
        assert noise == []

    def test_exact_tie_counts_as_noise(self) -> None:
        # Documented behavior: `ok_count * 2 >= len(samples)` treats a tie
        # as noise — benefit of the doubt goes to "not a real regression"
        # only on a genuine majority-or-tie of ok samples.
        confirmed, noise = confirm_flips({"q1": [False, True]})
        assert confirmed == []
        assert noise == ["q1"]

    def test_multiple_questions_classified_independently_and_sorted(self) -> None:
        confirmed, noise = confirm_flips({
            "qb": [False, False, False],
            "qa": [False, True, True],
        })
        assert confirmed == ["qb"]
        assert noise == ["qa"]

    def test_a_question_with_no_samples_is_skipped_not_misclassified(self) -> None:
        # The caller's own job to decide what happens to an id it couldn't
        # resample at all (e.g. no longer in the dataset) — this function
        # only classifies ids it actually received samples for.
        confirmed, noise = confirm_flips({"q1": []})
        assert confirmed == []
        assert noise == []
