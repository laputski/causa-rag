"""Regression guard — baseline-vs-run comparison.

Shared by the CI eval gate (`eval/gate.py`) and the interactive gateway so an
engineer sees "this run regressed against your baseline" inside the debug loop,
not only in CI. Pure / deterministic — no LLM, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Allowed relative drop vs baseline, per metric (default 5%).
DEFAULT_THRESHOLDS: dict[str, float] = {
    # Eval Measurement Trustworthiness, Phase 0 — current metric schema
    # (services/api_gateway/routers/experiments.py:_CompositeEvaluator).
    "retrieval_recall_at_k": 0.05,
    "retrieval_precision_at_k": 0.05,
    "answer_similarity": 0.05,
    "context_support": 0.05,
    "correct_refusal": 0.05,
    # Legacy keys — kept so historical baselines pinned before Phase 0
    # still get a threshold if ever read by this module directly (the
    # gateway compares like-for-like by key name; see compare() below for
    # the explicit cross-schema mismatch this can otherwise mask).
    "faithfulness": 0.05,
    "recall": 0.05,
    "precision": 0.05,
    "accuracy_semantic": 0.05,
    "contextual_recall": 0.05,
    "answer_relevancy": 0.05,
    "reference_overlap": 0.05,
}

# Absolute floors — a metric must never fall below this regardless of baseline.
SLA_ABSOLUTE_FLOORS: dict[str, float] = {
    "accuracy_semantic": 0.85,
}


@dataclass
class MetricDelta:
    metric: str
    baseline: float
    current: float
    delta: float          # current - baseline
    allowed_min: float
    regressed: bool


@dataclass
class RegressionReport:
    passed: bool
    deltas: list[MetricDelta] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "deltas": [
                {
                    "metric": d.metric,
                    "baseline": round(d.baseline, 4),
                    "current": round(d.current, 4),
                    "delta": round(d.delta, 4),
                    "allowed_min": round(d.allowed_min, 4),
                    "regressed": d.regressed,
                }
                for d in self.deltas
            ],
        }


def compare(
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> RegressionReport:
    """Compare a run's metrics to a baseline.

    A metric regresses if current < baseline * (1 - threshold), or falls below an
    absolute SLA floor.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    deltas: list[MetricDelta] = []
    violations: list[str] = []

    common_metrics = set(current_metrics) & set(baseline_metrics)
    if baseline_metrics and current_metrics and not common_metrics:
        # Eval Measurement Trustworthiness, Phase 0 — comparing by matching
        # key name means a run on the NEW metric schema vs a baseline still
        # on the OLD (or vice versa) silently produces zero deltas and
        # reads as "passed" — there is nothing to compare, not nothing
        # wrong. Surface that explicitly instead of a misleading green.
        violations.append(
            "no comparable metrics: current run and baseline use disjoint "
            f"metric schemas (current={sorted(current_metrics)}, "
            f"baseline={sorted(baseline_metrics)}) — re-pin baseline after "
            "an evaluator/metric-schema change"
        )
        return RegressionReport(passed=False, deltas=[], violations=violations)

    for metric, baseline_val in baseline_metrics.items():
        current_val = current_metrics.get(metric)
        if current_val is None:
            continue
        threshold = thresholds.get(metric, 0.05)
        allowed_min = baseline_val * (1 - threshold)
        regressed = current_val < allowed_min
        deltas.append(
            MetricDelta(
                metric=metric,
                baseline=baseline_val,
                current=current_val,
                delta=current_val - baseline_val,
                allowed_min=allowed_min,
                regressed=regressed,
            )
        )
        if regressed:
            violations.append(
                f"{metric}: {current_val:.4f} < {allowed_min:.4f} "
                f"(baseline={baseline_val:.4f}, threshold={threshold:.0%})"
            )

    # Absolute SLA floors — independent of baseline.
    for metric, floor in SLA_ABSOLUTE_FLOORS.items():
        current_val = current_metrics.get(metric)
        if current_val is not None and current_val < floor:
            violations.append(f"{metric}: {current_val:.4f} < SLA floor {floor:.4f}")

    return RegressionReport(passed=len(violations) == 0, deltas=deltas, violations=violations)


def check_gate(
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """Backwards-compatible tuple API used by ``eval/gate.py``."""
    report = compare(current_metrics, baseline_metrics, thresholds)
    return report.passed, report.violations


@dataclass
class PairedDiffReport:
    """Per-question before/after comparison — the
    aggregate-only ``compare``/``check_gate`` above answer "did the average
    move", not "did any individual question that used to pass now fail".
    A targeted fix that helps three questions and breaks two others can
    leave every average metric within threshold; this is the check that
    catches the two broken ones by name instead of averaging them away.

    A question is "ok" iff its funnel verdict layer is exactly ``"ok"`` —
    ``"not_applicable"`` (out-of-scope/uncovered questions, outside the
    diagnosable funnel) counts as not-ok for this classification, same as
    any failing layer. This keeps the rule simple (one boolean per side)
    at the cost of also flagging a question that moved from "answerable
    and ok" to "no longer answerable" (e.g. the dataset itself changed) as
    a flip — an honest signal (something about this question's evaluation
    changed), even if the root cause isn't a retrieval/rerank/generation
    regression.
    """
    fixed: list[str] = field(default_factory=list)
    flips: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    metric_deltas: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fixed": self.fixed,
            "flips": self.flips,
            "unchanged": self.unchanged,
            "metric_deltas": self.metric_deltas,
        }


def _is_ok(question_result: dict) -> bool:
    funnel = question_result.get("funnel") or {}
    return funnel.get("layer") == "ok"


def paired_diff(
    before: list[dict],
    after: list[dict],
) -> PairedDiffReport:
    """Per-question before/after comparison, keyed by ``question_id``.

    ``before``/``after`` are ``question_results``-shaped dicts (the same
    shape ``ExperimentResult.to_dict()["question_results"]`` produces) that
    already carry a ``funnel`` key (see ``core/eval/funnel.py``:
    ``diagnose_question().to_dict()``) — attaching it is the caller's job
    (mirrors how ``GET /experiments/{run_id}`` already attaches it before
    display), so this function stays pure: no I/O, no funnel computation
    of its own.

    A question present in only one of the two lists is skipped entirely —
    it isn't a fix or a regression, it's a question the comparison can't
    speak to (e.g. the dataset changed between runs). Pure / deterministic,
    no LLM — this function never re-runs anything; a caller that wants to
    rule out generation-metric noise before trusting a flip is expected to
    re-run that one question and compare again, not inside this function.
    """
    before_by_id = {qr["question_id"]: qr for qr in before if qr.get("question_id")}
    after_by_id = {qr["question_id"]: qr for qr in after if qr.get("question_id")}
    common_ids = set(before_by_id) & set(after_by_id)

    fixed: list[str] = []
    flips: list[str] = []
    unchanged: list[str] = []
    metric_deltas: dict[str, dict[str, float]] = {}

    for qid in sorted(common_ids):
        before_qr = before_by_id[qid]
        after_qr = after_by_id[qid]
        was_ok = _is_ok(before_qr)
        is_ok_now = _is_ok(after_qr)

        if not was_ok and is_ok_now:
            fixed.append(qid)
        elif was_ok and not is_ok_now:
            flips.append(qid)
        else:
            unchanged.append(qid)

        before_metrics = before_qr.get("metrics") or {}
        after_metrics = after_qr.get("metrics") or {}
        common_metrics = set(before_metrics) & set(after_metrics)
        if common_metrics:
            metric_deltas[qid] = {
                m: after_metrics[m] - before_metrics[m] for m in sorted(common_metrics)
            }

    return PairedDiffReport(
        fixed=fixed, flips=flips, unchanged=unchanged, metric_deltas=metric_deltas,
    )


def confirm_flips(resampled_ok: dict[str, list[bool]]) -> tuple[list[str], list[str]]:
    """Separates a genuine regression from generation-metric noise.

    Generative metrics (``answer_similarity``, judge-based scores) are not
    perfectly deterministic — the same question against the same pipeline
    can occasionally score just below the "ok" funnel threshold once and
    comfortably above it the next time. ``paired_diff`` above reports a
    flip from a single before/after sample; before a flip is trusted as a
    real regression, the caller is expected to re-run that one question a
    few more times and pass every sample's ok/not-ok verdict here (this
    function stays pure — it never re-runs anything itself, see
    ``paired_diff``'s own docstring for the same convention).

    ``resampled_ok`` maps a flipped ``question_id`` to the list of ok/not-ok
    booleans collected across every sample taken for it (including the
    original sample that made it a flip in the first place — the caller
    decides how many resamples to add on top of that). A question is
    demoted back to "noise" (not a real flip) when at least half its
    samples came back ok; otherwise it is confirmed as a genuine flip. With
    an odd sample count (the expected case — one original sample plus two
    resamples) this never ties.

    Returns ``(confirmed_flips, noise_ids)`` — both sorted for a stable,
    testable ordering. A question_id absent from ``resampled_ok`` entirely
    (the caller couldn't resample it — e.g. it no longer exists in the
    dataset) is the caller's own concern to keep in the original flip list;
    this function only ever classifies the ids it was actually given.
    """
    confirmed: list[str] = []
    noise: list[str] = []
    for qid in sorted(resampled_ok):
        samples = resampled_ok[qid]
        if not samples:
            continue
        ok_count = sum(1 for s in samples if s)
        if ok_count * 2 >= len(samples):
            noise.append(qid)
        else:
            confirmed.append(qid)
    return confirmed, noise
