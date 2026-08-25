"""core/eval/frontier.py.

Three questions that share a module because they share a subject: which
configuration to choose, whether the metric choosing it can be trusted, and
one specific way it can quietly fail.
"""
from __future__ import annotations

from core.eval.frontier import (
    ConfigPoint,
    calibrate_metric,
    measure_position_bias,
    pareto_frontier,
    point_from_run,
)


def _p(run_id: str, quality: float, latency: float) -> ConfigPoint:
    return ConfigPoint(run_id=run_id, label=run_id, quality=quality, latency_ms=latency, config={})


# ── The frontier ─────────────────────────────────────────────


def test_a_configuration_beaten_on_both_counts_is_dropped() -> None:
    """Nothing could justify choosing it: whatever you valued, the other
    point gives you more of it."""
    frontier = pareto_frontier([_p("good", 0.8, 100), _p("worse", 0.5, 200)])
    assert [p.run_id for p in frontier] == ["good"]


def test_a_genuine_trade_off_keeps_both() -> None:
    frontier = pareto_frontier([_p("fast", 0.5, 100), _p("accurate", 0.9, 900)])
    assert {p.run_id for p in frontier} == {"fast", "accurate"}


def test_the_frontier_reads_fastest_first() -> None:
    # Reading down the list is then reading the price of each increment.
    frontier = pareto_frontier([_p("accurate", 0.9, 900), _p("fast", 0.5, 100)])
    assert [p.run_id for p in frontier] == ["fast", "accurate"]


def test_an_equally_good_but_cheaper_configuration_is_not_hidden() -> None:
    """Two points at the same place do not dominate each other; reporting
    one would hide that a cheaper configuration reached it."""
    frontier = pareto_frontier([_p("a", 0.7, 300), _p("b", 0.7, 300)])
    assert len(frontier) == 2


def test_a_run_missing_a_number_is_excluded_rather_than_placed_at_zero() -> None:
    """At the origin an unmeasured configuration would look like the
    cheapest good one."""
    assert point_from_run({"run_id": "r", "aggregate_metrics": {}, "avg_stage_trace": {"total_ms": 5}}) is None
    assert point_from_run({"run_id": "r", "aggregate_metrics": {"retrieval_recall_at_k": 1.0}}) is None


def test_a_point_carries_the_config_that_produced_it() -> None:
    # A frontier of anonymous points tells a reader what is possible but not
    # what to do about it.
    point = point_from_run({
        "run_id": "r1", "config_name": "run",
        "aggregate_metrics": {"retrieval_recall_at_k": 0.7},
        "avg_stage_trace": {"total_ms": 120.0},
        "config": {"top_k": 5, "merge_strategy": "weighted", "merge_alpha": 0.3},
    })
    assert point is not None
    assert point.config["merge_strategy"] == "weighted"
    assert point.config["merge_alpha"] == 0.3


# ── Position bias ────────────────────────────────────────────


def test_quality_is_averaged_per_position_bucket() -> None:
    bias = measure_position_bias([(1, 0.9), (1, 0.7), (5, 0.4), (5, 0.6)])
    assert bias.by_position["1"] == 0.8
    assert bias.by_position["4-10"] == 0.5


def test_the_spread_says_whether_position_matters_at_all() -> None:
    indifferent = measure_position_bias([(1, 0.5), (1, 0.5), (5, 0.5), (5, 0.5)])
    assert indifferent.spread == 0.0


def test_a_bucket_too_small_to_be_a_mean_is_dropped() -> None:
    """One observation moving the spread would make noise look like a
    finding, which is what this measurement exists to rule out."""
    bias = measure_position_bias([(1, 0.9), (1, 0.7), (11, 0.1)])
    assert "11+" not in bias.by_position
    assert bias.spread == 0.0


def test_buckets_are_used_rather_than_a_single_coefficient() -> None:
    # The documented shape is not linear: the ends behave differently from
    # the middle, which one coefficient would average into nothing.
    bias = measure_position_bias([(1, 0.9), (1, 0.9), (2, 0.4), (3, 0.4), (12, 0.9), (13, 0.9)])
    assert set(bias.by_position) == {"1", "2-3", "11+"}
    assert bias.spread == 0.5


# ── Calibration against the reviewer ─────────────────────────


def _q(qid: str, value: float) -> dict:
    return {"question_id": qid, "metrics": {"answer_similarity": value}}


def test_the_two_disagreement_directions_are_reported_separately() -> None:
    """A metric calling a bad answer good is what lets a regression ship; a
    metric calling a good answer bad only wastes an investigation. One
    accuracy figure would let the cheap error mask the expensive one."""
    cal = calibrate_metric(
        [_q("q1", 0.9), _q("q2", 0.2)],
        {"q1": {"rating": "bad"}, "q2": {"rating": "good"}},
    )
    assert cal.false_good == ("q1",)
    assert cal.false_bad == ("q2",)
    assert cal.agree == 0


def test_agreement_counts_only_questions_carrying_both() -> None:
    """A question with a metric and no verdict says nothing about their
    agreement; counting it either way would move the number with no
    evidence behind the movement."""
    cal = calibrate_metric(
        [_q("q1", 0.9), _q("q2", 0.9)],
        {"q1": {"rating": "good"}},
    )
    assert cal.compared == 1
    assert cal.agreement == 1.0


def test_a_rating_that_is_neither_good_nor_bad_is_not_compared() -> None:
    cal = calibrate_metric([_q("q1", 0.9)], {"q1": {"rating": None, "comment": "unclear"}})
    assert cal.compared == 0


def test_no_overlap_at_all_reports_zero_agreement_rather_than_perfect() -> None:
    cal = calibrate_metric([_q("q1", 0.9)], {})
    assert cal.compared == 0
    assert cal.agreement == 0.0


def test_the_threshold_is_explicit_so_a_reader_can_disagree_with_it() -> None:
    lenient = calibrate_metric([_q("q1", 0.5)], {"q1": {"rating": "good"}}, threshold=0.4)
    strict = calibrate_metric([_q("q1", 0.5)], {"q1": {"rating": "good"}}, threshold=0.9)
    assert lenient.agree == 1
    assert strict.false_bad == ("q1",)


# ── Found live: latency is not comparable across pipeline sources ───────


def _src(run_id: str, quality: float, latency: float, source: str) -> ConfigPoint:
    return ConfigPoint(run_id=run_id, label=run_id, quality=quality,
                       latency_ms=latency, config={}, pipeline_source=source)


def test_an_external_run_never_dominates_an_in_process_one_on_speed() -> None:
    """Found live: an external RAG reported 0.15 ms against an in-process
    run's 24 521 ms at the same quality. The platform's stage trace measures
    the platform's own work, and for an HTTP call that is almost nothing —
    the remote system's real cost is invisible. Mixing the two produced a
    confident recommendation to adopt the external system on speed grounds
    nobody had measured."""
    from core.eval.frontier import frontier_by_source

    grouped = frontier_by_source([
        _src("external", 0.75, 0.15, "http"),
        _src("in_process", 0.76, 24520.0, "in_process"),
    ])
    # The in-process run survives in its own group instead of being wiped
    # out by an artefact from the other side.
    assert [p.run_id for p in grouped["in_process"]] == ["in_process"]
    assert [p.run_id for p in grouped["http"]] == ["external"]


def test_runs_of_one_source_are_still_compared_normally() -> None:
    from core.eval.frontier import frontier_by_source

    grouped = frontier_by_source([
        _src("slow_bad", 0.5, 900, "in_process"),
        _src("fast_good", 0.9, 100, "in_process"),
    ])
    assert [p.run_id for p in grouped["in_process"]] == ["fast_good"]


def test_a_point_records_which_source_it_came_from() -> None:
    point = point_from_run({
        "run_id": "r", "aggregate_metrics": {"retrieval_recall_at_k": 0.5},
        "avg_stage_trace": {"total_ms": 1.0}, "config": {"pipeline_source": "http"},
    })
    assert point is not None and point.pipeline_source == "http"


# ── Tokens as the third axis ─────────────────────────────────


def _t(run_id: str, quality: float, latency: float, tokens: float | None) -> ConfigPoint:
    return ConfigPoint(run_id=run_id, label=run_id, quality=quality,
                       latency_ms=latency, config={}, tokens=tokens)


def test_a_configuration_beaten_on_all_three_counts_is_dropped() -> None:
    frontier = pareto_frontier([_t("good", 0.8, 100, 500), _t("worse", 0.5, 200, 900)])
    assert [p.run_id for p in frontier] == ["good"]


def test_spending_fewer_tokens_at_equal_quality_and_speed_survives() -> None:
    """Without the third axis these two are indistinguishable, and the
    cheaper one would be dropped as a tie or kept by accident. With it, the
    expensive one is genuinely beaten."""
    frontier = pareto_frontier([_t("cheap", 0.7, 300, 400), _t("costly", 0.7, 300, 1200)])
    assert [p.run_id for p in frontier] == ["cheap"]


def test_more_tokens_bought_more_quality_keeps_both() -> None:
    frontier = pareto_frontier([_t("frugal", 0.6, 300, 400), _t("lavish", 0.9, 300, 2000)])
    assert {p.run_id for p in frontier} == {"frugal", "lavish"}


def test_one_run_without_token_counts_takes_the_axis_out_of_the_comparison() -> None:
    """The alternative is worse than not comparing: a run with no counts would
    read as the cheapest of the set, exactly the mistake that keeps an
    unmeasured quality metric off the frontier entirely."""
    from core.eval.frontier import tokens_comparable

    points = [_t("measured", 0.7, 300, 1200), _t("unmeasured", 0.7, 300, None)]
    assert tokens_comparable(points) is False
    # Both survive: on two axes they tie, and a tie is not domination.
    assert len(pareto_frontier(points)) == 2


def test_zero_tokens_is_a_measurement_and_not_a_missing_value() -> None:
    """An external system reports no counts of its own, so its runs sit at
    zero together and the axis simply does not separate them. That is not the
    same as the axis being absent, and the two must not collapse."""
    from core.eval.frontier import tokens_comparable

    points = [_t("ext1", 0.7, 1, 0.0), _t("ext2", 0.75, 2, 0.0)]
    assert tokens_comparable(points) is True
    assert [p.run_id for p in pareto_frontier(points)] == ["ext1", "ext2"]


def test_a_point_carries_the_token_sum_of_its_run() -> None:
    point = point_from_run({
        "run_id": "r1",
        "aggregate_metrics": {"retrieval_recall_at_k": 0.7},
        "avg_stage_trace": {"total_ms": 120.0, "input_tokens": 800.0, "output_tokens": 150.0},
    })
    assert point is not None
    assert point.tokens == 950.0
    assert point.to_dict()["tokens"] == 950.0


def test_a_run_whose_trace_has_no_token_keys_leaves_the_axis_unmeasured() -> None:
    point = point_from_run({
        "run_id": "r1",
        "aggregate_metrics": {"retrieval_recall_at_k": 0.7},
        "avg_stage_trace": {"total_ms": 120.0},
    })
    assert point is not None
    assert point.tokens is None
