"""Choosing between configurations, and checking the metric that chooses.

Three questions that look unrelated and are not. Which configuration to run
(5.2) depends on a metric being trustworthy; whether it is trustworthy
depends on how it compares against a human verdict (5.7); and one specific
way it can be untrustworthy is that a chunk's *position* changes the answer's
quality independently of its content (5.6).

Pure: takes finished runs and stored verdicts, returns what they imply. No
run is started here — a module that could launch experiments would make the
cost of asking a question invisible at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ── The quality/latency frontier ─────────────────────────────


@dataclass(frozen=True)
class ConfigPoint:
    """One finished run, reduced to the numbers a choice is made on."""

    run_id: str
    label: str
    quality: float
    latency_ms: float
    config: dict[str, Any]
    # The third axis, measured in tokens rather than in money.
    #
    # Tokens are what the platform actually observes; money is tokens times a
    # price it has no source for. A locally hosted generator costs no money and
    # plenty of compute, so reporting zero currency for it and a real figure
    # for an API model would put two different quantities on one axis. Tokens
    # per question are comparable across both and convert to money by
    # multiplication whenever an operator knows their own price.
    #
    # None means the run carried no token counts at all. Zero means it carried
    # them and they were zero, which is a measurement — the same distinction
    # that keeps an unmeasured quality metric off the frontier entirely.
    tokens: float | None = None
    # Found live: an external RAG run reported 0.15 ms against an in-process
    # run's 24 521 ms at the same quality — a 160000x lead that is entirely
    # an artefact. The platform's stage trace measures the platform's own
    # work, and for an HTTP call that is almost nothing; the remote system's
    # actual cost is invisible unless it returns a trace of its own.
    #
    # Latencies are therefore comparable only within one source, and the
    # frontier is computed per source rather than across all runs. Mixing
    # them produced a confident recommendation to adopt the external system
    # on speed grounds that had never been measured.
    pipeline_source: str = "in_process"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline_source": self.pipeline_source,
            "label": self.label,
            "quality": round(self.quality, 4),
            "latency_ms": round(self.latency_ms, 1),
            "tokens": None if self.tokens is None else round(self.tokens, 1),
            "config": self.config,
        }


def tokens_comparable(points: list[ConfigPoint]) -> bool:
    """Whether the token axis can take part in this group's comparison.

    Only when *every* point carries a count. One run without token data among
    ten with it would otherwise be treated as the cheapest of the ten, which
    is the same mistake as placing an unmeasured quality metric at zero.
    """
    return bool(points) and all(p.tokens is not None for p in points)


def frontier_by_source(points: list[ConfigPoint]) -> dict[str, list[ConfigPoint]]:
    """One frontier per pipeline source, never one across all of them.

    Latency means a different thing on each side: for an in-process run it
    is the whole pipeline, for an external one it is what the platform spent
    handing the work over. Comparing across the two ranks a measurement
    artefact, not a system.
    """
    grouped: dict[str, list[ConfigPoint]] = {}
    for point in points:
        grouped.setdefault(point.pipeline_source, []).append(point)
    return {source: pareto_frontier(group) for source, group in grouped.items()}


def pareto_frontier(points: list[ConfigPoint]) -> list[ConfigPoint]:
    """The configurations that are not beaten outright.

    A point is dominated when another is at least as good on every axis and
    strictly better on one. Dominated points are dropped because nothing could
    justify choosing one: whatever you valued, the dominating point gives you
    more of it.

    What survives is exactly the set where a choice is a genuine trade-off,
    which is the only place a human judgement about latency and token budgets
    is worth asking for. Returned fastest-first so reading down the list is
    reading the price of each quality increment.

    **The axes are quality, latency and tokens, and the third is used only
    when every point in the set carries it.** Dropping it silently for a set
    where one run lacks counts would compare the others on two axes while the
    reader believes it was three; `tokens_comparable` is exported so a caller
    can say which happened.

    Ties matter here. Two points identical on every axis do not dominate each
    other, so both survive — reporting one would hide that a second
    configuration reached the same place.
    """
    with_tokens = tokens_comparable(points)

    def dominates(q: ConfigPoint, p: ConfigPoint) -> bool:
        better_or_equal = q.quality >= p.quality and q.latency_ms <= p.latency_ms
        strictly_better = q.quality > p.quality or q.latency_ms < p.latency_ms
        if with_tokens:
            qt, pt = q.tokens or 0.0, p.tokens or 0.0
            better_or_equal = better_or_equal and qt <= pt
            strictly_better = strictly_better or qt < pt
        return better_or_equal and strictly_better

    frontier = [p for p in points if not any(dominates(q, p) for q in points)]
    return sorted(frontier, key=lambda p: (p.latency_ms, -p.quality))


def point_from_run(
    payload: dict[str, Any], quality_metric: str = "retrieval_recall_at_k",
) -> ConfigPoint | None:
    """Reduces a stored run to a point, or None when it lacks either number.

    None rather than a zero: a run whose quality metric is absent has not
    scored zero, and letting it sit at the origin of the frontier would make
    an unmeasured configuration look like the cheapest good one.
    """
    metrics = payload.get("aggregate_metrics") or {}
    quality = metrics.get(quality_metric)
    trace = payload.get("avg_stage_trace") or {}
    latency = trace.get("total_ms")
    if quality is None or latency is None:
        return None
    # Absent keys leave the axis unmeasured; present keys reading zero are a
    # measurement of zero. The two must not collapse into one number.
    tokens_in, tokens_out = trace.get("input_tokens"), trace.get("output_tokens")
    tokens = (
        None if tokens_in is None and tokens_out is None
        else float(tokens_in or 0) + float(tokens_out or 0)
    )
    config = payload.get("config") or {}
    return ConfigPoint(
        pipeline_source=str(config.get("pipeline_source") or "in_process"),
        run_id=str(payload.get("run_id") or ""),
        label=str(payload.get("config_name") or payload.get("run_id") or ""),
        quality=float(quality),
        latency_ms=float(latency),
        tokens=tokens,
        config={
            "pipeline_id": config.get("pipeline_id"),
            "top_k": config.get("top_k"),
            "fetch_k": config.get("fetch_k"),
            "merge_strategy": config.get("merge_strategy"),
            "merge_alpha": config.get("merge_alpha"),
            "reranker": (config.get("reranker") or {}).get("component_id"),
        },
    )


# ── Position bias ────────────────────────────────────────────


@dataclass(frozen=True)
class PositionBias:
    """How answer quality varies with where the correct chunk sat.

    Content is held roughly constant by construction: every observation here
    is a question whose expected source *was* in context, so the difference
    between buckets is about placement rather than about whether the answer
    was available at all.
    """

    by_position: dict[str, float]
    counts: dict[str, int]

    @property
    def spread(self) -> float:
        """Best bucket minus worst. The number that says whether position
        matters at all; near zero means the pipeline is indifferent to it and
        no work here would pay."""
        if len(self.by_position) < 2:
            return 0.0
        return max(self.by_position.values()) - min(self.by_position.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_position": {k: round(v, 4) for k, v in self.by_position.items()},
            "counts": self.counts,
            "spread": round(self.spread, 4),
        }


_BUCKETS = (("1", 1, 1), ("2-3", 2, 3), ("4-10", 4, 10), ("11+", 11, 10**9))


def _bucket(rank: int) -> str:
    for name, lo, hi in _BUCKETS:
        if lo <= rank <= hi:
            return name
    return "11+"


def measure_position_bias(
    observations: list[tuple[int, float]], min_per_bucket: int = 2,
) -> PositionBias:
    """Mean quality per position bucket.

    ``observations`` are ``(rank_of_expected_source, quality)`` pairs for
    questions where the source was in context.

    Buckets rather than a correlation, because the effect is not expected to
    be linear: the well-documented shape is that first and last positions
    behave differently from the middle, which a single coefficient would
    average into nothing.

    A bucket below ``min_per_bucket`` is dropped rather than reported. One
    observation is not a mean, and a bucket of size one moving the spread
    would make noise look like a finding — the very thing this measurement
    exists to rule out.
    """
    grouped: dict[str, list[float]] = {}
    for rank, quality in observations:
        if rank < 1:
            continue
        grouped.setdefault(_bucket(rank), []).append(quality)
    kept = {k: v for k, v in grouped.items() if len(v) >= min_per_bucket}
    return PositionBias(
        by_position={k: sum(v) / len(v) for k, v in kept.items()},
        counts={k: len(v) for k, v in kept.items()},
    )


# ── Calibrating a metric against a reviewer ──────────────────


@dataclass(frozen=True)
class Calibration:
    """Agreement between a cheap metric and the verdict it stands in for.

    The reviewer is taken as ground truth and the metric as its cheap
    approximation, so the two disagreement directions are not symmetric and
    are never summed into one accuracy number.
    """

    metric: str
    threshold: float
    agree: int = 0
    # Metric says good, reviewer says bad. The dangerous direction: these
    # are the failures a green dashboard hides.
    false_good: tuple[str, ...] = ()
    # Metric says bad, reviewer says good. Costly rather than dangerous —
    # wasted investigation, not shipped harm.
    false_bad: tuple[str, ...] = ()

    @property
    def compared(self) -> int:
        return self.agree + len(self.false_good) + len(self.false_bad)

    @property
    def agreement(self) -> float:
        return self.agree / self.compared if self.compared else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "compared": self.compared,
            "agreement": round(self.agreement, 4),
            "false_good": list(self.false_good),
            "false_bad": list(self.false_bad),
        }


def calibrate_metric(
    question_results: list[dict[str, Any]],
    feedback_by_question: dict[str, Any],
    metric: str = "answer_similarity",
    threshold: float = 0.6,
) -> Calibration:
    """Where the metric and the reviewer disagree, and how.

    Only questions carrying both a metric value and a reviewer rating are
    compared; a question with one and not the other says nothing about their
    agreement, and counting it either way would move the number without
    evidence.

    The two disagreement directions are reported separately rather than as
    one accuracy figure, because they cost differently. A metric calling a
    bad answer good is what lets a regression ship; a metric calling a good
    answer bad only wastes an investigation. A single number would let the
    cheap error mask the expensive one.
    """
    agree, false_good, false_bad = 0, [], []
    for qr in question_results:
        qid = str(qr.get("question_id") or "")
        value = (qr.get("metrics") or {}).get(metric)
        rating = (feedback_by_question.get(qid) or {}).get("rating")
        if value is None or rating not in ("good", "bad"):
            continue
        metric_says_good = float(value) >= threshold
        reviewer_says_good = rating == "good"
        if metric_says_good == reviewer_says_good:
            agree += 1
        elif metric_says_good:
            false_good.append(qid)
        else:
            false_bad.append(qid)
    return Calibration(
        metric=metric, threshold=threshold, agree=agree,
        false_good=tuple(false_good), false_bad=tuple(false_bad),
    )
