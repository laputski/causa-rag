"""Experiment comparison: config diff + metric deltas between two runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.eval.regression import PairedDiffReport
from core.experiment.runner import ExperimentResult


@dataclass
class MetricDelta:
    metric: str
    # None means the metric was never measured on that side, which is
    # different from measuring zero. Filling the gap with 0.0 was the
    # previous behaviour, and it fabricated verdicts: the UI classifies a
    # move away from zero as an improvement, so a metric absent from the
    # "before" run rendered as a green rise from 0.000.
    before: float | None
    after: float | None

    @property
    def delta(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def delta_pct(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        if self.before == 0:
            return float("inf") if self.after > 0 else 0.0
        return (self.after - self.before) / abs(self.before) * 100


@dataclass
class CompatWarning:
    """One reason the pair being compared may not be comparable.

    Same shape as core/eval/detectors.py:DiagnosticItem, with `params` added.
    The interface is fully localised while a detector's `title`/`detail` are
    English prose composed server-side, so those two strings would have shown
    up untranslated in a Russian UI. `id` and `params` let the client render
    its own sentence, and the English text stays as the fallback for any id
    the client does not know yet.
    """
    id: str
    severity: str
    title: str
    detail: str
    params: dict[str, Any] = field(default_factory=dict)
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "params": self.params,
            "action": self.action,
        }


@dataclass
class CompareReport:
    config_diff: dict[str, dict[str, Any]]
    metric_deltas: list[MetricDelta]
    compatibility: list[CompatWarning] = field(default_factory=list)

    def summary(self) -> str:
        lines: list[str] = []
        # Warnings lead. This text is what gets exported and pasted into
        # tickets, so a caveat printed under the numbers would be a caveat
        # nobody reads.
        if self.compatibility:
            lines.append("Comparability:")
            for w in self.compatibility:
                lines.append(f"  [{w.severity}] {w.title}: {w.detail}")
            lines.append("")

        if self.config_diff:
            lines.append("Config changes:")
            for field_name, change in self.config_diff.items():
                lines.append(f"  {field_name}: {change['before']!r} → {change['after']!r}")
        else:
            lines.append("Config: identical")

        lines.append("\nMetric deltas (after - before):")
        for d in sorted(self.metric_deltas, key=lambda x: x.metric):
            if d.before is None or d.after is None:
                side = "before" if d.before is None else "after"
                lines.append(f"  {d.metric}: not measured in the {side} run")
                continue
            delta = d.after - d.before
            sign = "+" if delta >= 0 else ""
            lines.append(f"  {d.metric}: {d.before:.4f} → {d.after:.4f}  ({sign}{delta:.4f})")
        return "\n".join(lines)


def _answered(result: ExperimentResult) -> int:
    """How many questions this run actually answered.

    The stored `n_questions` is the dataset's planned total, which a stopped
    run legitimately never reaches. Every count here is of answered questions,
    so a stopped run reads as the short run it is.
    """
    return len(result.question_results)


def _errored(result: ExperimentResult) -> int:
    return sum(1 for qr in result.question_results if qr.error)


def check_comparability(
    before: ExperimentResult,
    after: ExperimentResult,
    paired: PairedDiffReport | None = None,
) -> list[CompatWarning]:
    """Every reason these two runs may not be measuring the same thing.

    `paired` is optional. Without it the rules that need a question-level
    overlap are skipped and the rest still apply, which is what lets the
    pre-flight check reuse this function before any comparison has run.

    Ordered by severity, because the caller renders the list as it arrives.

    One kind of incompatibility is still undetectable from run data alone,
    and this function stays silent about it, so it never implies it checked:
    a question whose text was edited while keeping its id (question ids are
    the only key the pairing has, see core/eval/regression.py:paired_diff).

    A corpus reloaded between the two runs under the same `corpus_id` used to
    be the second of those, and is not any more: a run records the manifest of
    the corpus it queried, so two runs naming one corpus can be asked whether
    they queried the same documents. When either run carries no manifest the
    silence returns, and it is the same silence as before, never a verdict
    of "unchanged".
    """
    errors: list[CompatWarning] = []
    warns: list[CompatWarning] = []
    infos: list[CompatWarning] = []

    b_n, a_n = _answered(before), _answered(after)

    if b_n == 0 or a_n == 0:
        empty = "before" if b_n == 0 else "after"
        empty_label = "A" if b_n == 0 else "B"
        errors.append(CompatWarning(
            id="empty_run", severity="error",
            title="One of the runs answered no questions",
            detail=(
                f"The {empty} run holds no question results, so there is nothing to "
                "compare against. Every number below rests on the other run alone."
            ),
            params={"side": empty_label},
            action="Check why that run produced no results, then run it again.",
        ))

    before_corpus = (before.corpus_manifest or {}).get("documents_digest")
    after_corpus = (after.corpus_manifest or {}).get("documents_digest")
    if (before_corpus and after_corpus and before_corpus != after_corpus
            and before.config.corpus_id == after.config.corpus_id):
        errors.append(CompatWarning(
            id="corpus_changed", severity="error",
            title="The corpus changed between the two runs",
            detail=(
                f"Both runs name corpus {before.config.corpus_id!r} and the documents behind "
                f"that name are not the same set: "
                f"{(before.corpus_manifest or {}).get('document_count', '?')} documents loaded at "
                f"{(before.corpus_manifest or {}).get('loaded_at', 'an unrecorded time')} against "
                f"{(after.corpus_manifest or {}).get('document_count', '?')} loaded at "
                f"{(after.corpus_manifest or {}).get('loaded_at', 'an unrecorded time')}. "
                "A difference in the numbers below may be a difference in the documents."
            ),
            params={
                "corpus_id": before.config.corpus_id,
                "loaded_before": (before.corpus_manifest or {}).get("loaded_at", ""),
                "loaded_after": (after.corpus_manifest or {}).get("loaded_at", ""),
            },
            action="Compare runs made against one loading of the corpus, or load it once and "
                   "run both configurations again.",
        ))

    if before.config.dataset_name != after.config.dataset_name:
        errors.append(CompatWarning(
            id="different_dataset", severity="error",
            title="The runs were made on different datasets",
            detail=(
                f"A: {before.config.dataset_name or 'unnamed'}, {b_n} questions. "
                f"B: {after.config.dataset_name or 'unnamed'}, {a_n} questions. "
                "Aggregate metrics are averages over different question sets."
            ),
            params={
                "dataset_before": before.config.dataset_name, "dataset_after": after.config.dataset_name,
                "count_before": b_n, "count_after": a_n,
            },
        ))

    if before.config.corpus_id != after.config.corpus_id:
        errors.append(CompatWarning(
            id="different_corpus", severity="error",
            title="The runs were made against different corpora",
            detail=(
                f"A: {before.config.corpus_id}. B: {after.config.corpus_id}. "
                "Retrieval metrics measure two different indexes."
            ),
            params={"corpus_before": before.config.corpus_id, "corpus_after": after.config.corpus_id},
        ))

    if paired is not None:
        matched = len(paired.fixed) + len(paired.flips) + len(paired.unchanged)
        if matched == 0:
            errors.append(CompatWarning(
                id="no_overlap", severity="error",
                title="No question is shared by the two runs",
                detail=(
                    f"{len(paired.only_in_before)} questions exist only in A and "
                    f"{len(paired.only_in_after)} only in B. The per-question comparison "
                    "is empty because there was nothing to pair, which is a different "
                    "statement from a change that moved no question."
                ),
                params={
                    "only_in_before": len(paired.only_in_before),
                    "only_in_after": len(paired.only_in_after),
                },
            ))
        elif (paired.only_in_before or paired.only_in_after) \
                and before.config.dataset_name == after.config.dataset_name:
            warns.append(CompatWarning(
                id="partial_overlap", severity="warn",
                title="The dataset changed between the two runs",
                detail=(
                    f"Both runs name the same dataset, yet {len(paired.only_in_before)} "
                    f"questions appear only in A and {len(paired.only_in_after)} only in B. "
                    f"The comparison covers the {matched} they share."
                ),
                params={
                    "matched": matched,
                    "only_in_before": len(paired.only_in_before),
                    "only_in_after": len(paired.only_in_after),
                },
                action="Version the dataset so a run records which revision it measured.",
            ))

    if before.realm_id != after.realm_id:
        warns.append(CompatWarning(
            id="different_realm", severity="warn",
            title="The runs belong to different realms",
            detail=(
                f"A: {before.realm_id or 'no realm'}. B: {after.realm_id or 'no realm'}. "
                "A realm carries its own corpus and its own backends, and it is absent "
                "from the configuration diff below."
            ),
            params={"realm_before": before.realm_id, "realm_after": after.realm_id},
        ))

    # A run stored before these fields were captured carries an empty string,
    # and an empty side is not a difference. Found while sweeping every pair of
    # a real store: 80% of the prompt findings and 99% of the model findings
    # came from one side holding no record, where the honest claim is that the
    # question cannot be answered from what was stored.
    prompt_params = {
        "prompt_before": before.prompt_id, "version_before": before.prompt_version,
        "prompt_after": after.prompt_id, "version_after": after.prompt_version,
    }
    if before.prompt_id and after.prompt_id:
        if (before.prompt_id, before.prompt_version) != (after.prompt_id, after.prompt_version):
            warns.append(CompatWarning(
                id="different_prompt", severity="warn",
                title="The runs used different prompts",
                detail=(
                    f"A: {before.prompt_id} v{before.prompt_version}. "
                    f"B: {after.prompt_id} v{after.prompt_version}. "
                    "The prompt is recorded on the run and is absent from the "
                    "configuration diff below, so the diff can read as identical while "
                    "the answers were generated by different instructions."
                ),
                params=prompt_params,
            ))
    elif before.prompt_id or after.prompt_id:
        blank = "A" if not before.prompt_id else "B"
        warns.append(CompatWarning(
            id="prompt_unrecorded", severity="warn",
            title=f"Run {blank} did not record which prompt it used",
            detail=(
                f"The other run used {before.prompt_id or after.prompt_id}. Whether the "
                "prompt changed between the two cannot be answered from what was stored, "
                "and the configuration diff below is silent about prompts either way."
            ),
            params={
                **prompt_params, "side": blank,
                "other": before.prompt_id or after.prompt_id,
                "other_version": before.prompt_version or after.prompt_version,
            },
        ))

    model_params = {
        "model_before": before.generator_model, "model_after": after.generator_model,
    }
    if before.generator_model and after.generator_model:
        if before.generator_model != after.generator_model:
            warns.append(CompatWarning(
                id="different_generator_model", severity="warn",
                title="The answers were generated by different models",
                detail=(
                    f"A: {before.generator_model}. B: {after.generator_model}. "
                    "The model is recorded on the run and is absent from the "
                    "configuration diff below."
                ),
                params=model_params,
            ))
    elif before.generator_model or after.generator_model:
        blank = "A" if not before.generator_model else "B"
        warns.append(CompatWarning(
            id="generator_model_unrecorded", severity="warn",
            title=f"Run {blank} did not record which model generated its answers",
            detail=(
                f"The other run used {before.generator_model or after.generator_model}. "
                "Whether the model changed between the two cannot be answered from what "
                "was stored."
            ),
            params={
                **model_params, "side": blank,
                "other": before.generator_model or after.generator_model,
            },
        ))

    if before.config.retrieval_only != after.config.retrieval_only:
        warns.append(CompatWarning(
            id="retrieval_only_mismatch", severity="warn",
            title="One run skipped generation",
            detail=(
                "A retrieval-only run scores retrieval and never pays for an answer, so "
                "the two runs carry different metrics by construction."
            ),
            params={
                "retrieval_only_before": before.config.retrieval_only,
                "retrieval_only_after": after.config.retrieval_only,
            },
        ))

    if before.stopped or after.stopped:
        stopped_side = "before" if before.stopped else "after"
        stopped_label = "A" if before.stopped else "B"
        warns.append(CompatWarning(
            id="partial_run", severity="warn",
            title="One of the runs was stopped early",
            detail=(
                f"The {stopped_side} run answered {b_n if before.stopped else a_n} questions "
                "of the dataset's total, so its averages were taken over fewer questions."
            ),
            params={"side": stopped_label},
        ))

    b_err, a_err = _errored(before), _errored(after)
    if b_err or a_err:
        warns.append(CompatWarning(
            id="errored_questions", severity="warn",
            title="Some questions failed to be answered",
            detail=(
                f"A: {b_err} of {b_n}. B: {a_err} of {a_n}. A failed question carries no "
                "metrics, so the averages cover fewer questions than the pairing does."
            ),
            params={"errored_before": b_err, "errored_after": a_err},
        ))

    # An empty coverage_check means a run stored before the field existed,
    # which the run detectors already treat as nothing to report.
    for side, label, run in (("before", "A", before), ("after", "B", after)):
        coverage = run.coverage_check or {}
        if coverage and not coverage.get("checked"):
            warns.append(CompatWarning(
                id="coverage_unverified", severity="warn",
                title="Corpus coverage was not verified for one run",
                detail=(
                    f"The {side} run took its answerability classes from the dataset's own "
                    f"refs without checking them against the index "
                    f"({coverage.get('reason') or 'no reason given'}). Which questions count "
                    "toward its retrieval metrics is therefore unverified."
                ),
                params={"side": label, "reason": coverage.get("reason", "")},
            ))

    missing = set(before.aggregate_metrics) ^ set(after.aggregate_metrics)
    if missing:
        warns.append(CompatWarning(
            id="metric_missing_in_one_run", severity="warn",
            title="Some metrics were measured in only one run",
            detail=(
                f"{', '.join(sorted(missing))}. Those rows carry no delta, because an "
                "unmeasured metric has no value to compare."
            ),
            params={"metrics": sorted(missing), "metrics_list": ", ".join(sorted(missing))},
        ))

    if before.run_id and before.run_id == after.run_id:
        infos.append(CompatWarning(
            id="same_run", severity="info",
            title="A run compared against itself",
            detail=(
                "Both sides are the same run, so every delta is zero by construction. "
                "This is a useful way to see what the comparison reports when nothing "
                "changed at all."
            ),
            params={"run_id": before.run_id},
        ))

    return errors + warns + infos


def compare(before: ExperimentResult, after: ExperimentResult) -> CompareReport:
    """Compare two experiment results: config diff + metric deltas."""
    config_diff = before.config.diff(after.config)

    all_metrics = set(before.aggregate_metrics) | set(after.aggregate_metrics)
    deltas = [
        MetricDelta(
            metric=m,
            before=before.aggregate_metrics.get(m),
            after=after.aggregate_metrics.get(m),
        )
        for m in all_metrics
    ]
    return CompareReport(config_diff=config_diff, metric_deltas=deltas)
