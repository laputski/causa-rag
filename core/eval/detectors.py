"""Silent-degradation detectors.

Deterministic, rule-based checks over a finished run. Each detector encodes a
real debugging incident (see docs/DEBUGGING.md) so the platform flags the failure
automatically instead of the engineer finding it by hand after a complaint.

Pure functions — no LLM, no I/O. Mirrored in ui/src/lib/diagnostics.ts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

Severity = str  # "ok" | "info" | "warn" | "error"

# A refusal phrase, in either of the two languages this platform has served so
# far. This is domain logic rather than presentation: it decides whether an
# answer counts as a refusal, so it stays in code and cannot move to the UI's
# translation files. Extending it for a new language means adding that
# language's phrasing here.
#
# The English half was too narrow to catch the demo realm's own refusals. Its
# prompt tells the model to say so when the context does not hold the answer,
# and its two out-of-scope questions exist so `correct_refusal` has something to
# count; a model writing "the handbook does not cover this" or "I could not find
# this" matched nothing, so the metric read zero refusals on a run that had
# refused twice. Measured against those wordings before and after.
#
# `services/api_gateway/routers/experiments.py` used to keep a second, strictly
# weaker copy of this pattern. It imports this one now: one rule, one place.
NOT_FOUND_RE = re.compile(
    r"(не\s+найден|отсутству|не\s+содержит|нет\s+информац|не\s+указан"
    r"|not\s+found|no\s+information|does\s+not\s+(contain|cover|say|include|mention)"
    r"|not\s+contain|could\s+not\s+find|cannot\s+(find|answer)|unable\s+to\s+answer"
    r"|is\s+not\s+covered|no\s+relevant\s+information)",
    re.IGNORECASE,
)

# The old private name, kept so nothing in this module has to change at once.
_NOT_FOUND = NOT_FOUND_RE


@dataclass
class DiagnosticItem:
    id: str
    severity: Severity
    title: str
    detail: str
    action: str = ""
    # Which entries of the failure catalogue this finding is evidence for.
    # Filled by the services layer, never here: the catalogue knows which
    # signals evidence a failure and the signals know nothing about the
    # catalogue, so a detector is never edited because a description of it
    # changed. Empty on a finding no entry names, which is a fact worth seeing.
    failure_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "action": self.action,
            "failure_ids": list(self.failure_ids),
        }


def _all_source_refs(run: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for qr in run.get("question_results", []):
        refs.extend(qr.get("source_refs", []) or [])
    return refs


def _generated_answers(run: dict[str, Any]) -> list[str]:
    return [qr.get("generated_answer", "") for qr in run.get("question_results", [])]


# ── individual detectors ───────────────────────────────────────────────────────

def detect_stub_embedder(refs: list[dict[str, Any]]) -> DiagnosticItem | None:
    """Dense scores at the floor ⇒ the corpus was indexed with random vectors.

    The check is only possible when the run recorded a per-signal score split
    at all, and that is not a technicality. Only the hybrid retriever writes
    `dense_score`/`sparse_score`; a dense-only pipeline leaves both at the
    field's default and carries its retrieval score in `score` instead, and an
    external system reports no split of its own either way.

    Found live, and this is what the rule below is for: the old version read
    the absent field as evidence and reported "indexed with random vectors" on
    **every one of the eighteen stored dense-only runs**, each of which has a
    recall@k between 0.81 and 1.00. Eighteen out of eighteen, at severity
    error, on healthy runs.

    Two fixes were considered and rejected. Skipping the check for a non-hybrid
    pipeline trades a false alarm for a permanent silence. Reading `score`
    instead does not restore the check either: after a reranker `score` carries
    the reranker's output and not any similarity, and even without one, a
    stub embedder over a few thousand random vectors of this width yields top
    cosines around 0.1, which the 0.01 floor does not separate from health.

    So the check says which of the three it is. Split recorded and dense side
    collapsed: a fault. Split recorded and dense side alive: silence. No split
    recorded: an explicit `embedder_unverified`, because a reader must be able
    to tell "not checked" from "checked and fine". The reliable evidence is a
    record of which model the corpus was actually indexed with, which nothing
    keeps yet.

    The rule reads the data instead of the configuration, so it needs no
    special case for an external system: a run that reports no split gets the
    same honest "could not check" whoever produced it.
    """
    dense = [r.get("dense_score") or 0.0 for r in refs]
    sparse = [r.get("sparse_score") or 0.0 for r in refs]
    if not dense:
        return None

    split_recorded = any(d > 0 for d in dense) or any(s > 0 for s in sparse)
    if not split_recorded:
        return DiagnosticItem(
            id="embedder_unverified",
            severity="info",
            title="Whether the embedder is real could not be checked",
            detail=f"None of the {len(dense)} retrieved chunks carries a per-signal score "
            "split, which is what a dense-only pipeline and an external system both look "
            "like. Nothing here distinguishes a corpus indexed with a real model from one "
            "indexed with random vectors.",
            action="Read the run's recall against its reference sources: a corpus indexed "
                   "with random vectors cannot reach a high one.",
        )

    nonzero = [d for d in dense if d > 0.01]
    if len(nonzero) / len(dense) < 0.1:
        return DiagnosticItem(
            id="stub_embedder",
            severity="error",
            title="Looks like a stub embedder",
            detail=f"{len(dense) - len(nonzero)}/{len(dense)} chunks have a dense score at "
            "the floor while the sparse side reports real values, so the corpus was probably "
            "indexed with random vectors.",
            action="Re-index the corpus with the real embedding model switched on.",
        )
    return None


def detect_duplicates(run: dict[str, Any]) -> DiagnosticItem | None:
    """The same text twice in one question's context.

    Counted per question, and never across the run. It used to read every
    question's sources pooled into one list, where a chunk that answers two
    questions is indistinguishable from a chunk stored twice, so the detector
    was measuring how useful a chunk is and calling that a defect.

    Measured on the proving ground's healthy corpus: fifteen questions returned
    seventy-five sources of twenty-eight distinct texts, with the most useful
    chunk answering six questions and not one repeat inside any single
    question. The pooled rule reported forty-seven duplicates. Across the
    forty-one stored runs it fired on thirty-two, and on every one of those
    thirty-two the repeats existed only in the pool.

    The failure it exists for is a context holding the same passage twice,
    which wastes the window and lets one source outvote the rest. That is a
    property of one context, so it is asked of one context.
    """
    per_question = 0
    affected = 0
    for question in run.get("question_results") or []:
        texts = [r.get("chunk_text", "") for r in (question.get("source_refs") or [])
                 if r.get("chunk_text")]
        repeats = len(texts) - len(set(texts))
        if repeats:
            per_question += repeats
            affected += 1
    if per_question > 0:
        return DiagnosticItem(
            id="duplicates",
            severity="warn",
            title="Duplicates in the context",
            detail=f"{per_question} repeated chunks inside a single question's context, "
                   f"across {affected} question(s).",
            action="Check that chunk_id is deterministic, then drop the collection and re-index.",
        )
    return None


def detect_header_only(refs: list[dict[str, Any]]) -> DiagnosticItem | None:
    """Chunks with a structural_path but (near-)empty body — chunker split headers."""
    candidates = [r for r in refs if r.get("structural_path")]
    if not candidates:
        return None
    header_only = [r for r in candidates if len((r.get("chunk_text") or "").strip()) < 40]
    if header_only and len(header_only) / len(candidates) > 0.5:
        return DiagnosticItem(
            id="header_only",
            severity="warn",
            title="Chunks carry headings only",
            detail=f"{len(header_only)}/{len(candidates)} chunks carry almost no body "
            "(structural_path only).",
            action="Raise min_chars, or switch to sentence or paragraph chunking.",
        )
    return None


def detect_bm25_dominance(refs: list[dict[str, Any]]) -> DiagnosticItem | None:
    """One half of a hybrid retriever supplies almost the whole context.

    Measured by where the chunks came from, not by how large their scores are.
    That distinction is the whole of this function's history.

    The previous rule compared raw scores: sparse above three times dense for
    most chunks. Measured over the 4664 scored pairs on this machine, it holds
    for **100%** of them, on every hybrid run, always. Not because one half
    rules the ranking but because the two numbers are not on one scale: dense
    is a cosine between 0 and 0.89, sparse is a raw lexical score between 0 and
    58.1. The comparison is arithmetic, not evidence. Worse, 95.4% of those
    comparisons were against a dense score of exactly zero, which does not mean
    "the model scored this chunk zero" but "this chunk was not in the dense
    list at all", so the rule was reading an absence as a low value.

    The rank-based question has an answer on the same data: between 11% and
    47% of the final context comes from the sparse half alone, across all 23
    hybrid runs. That is a balanced merge, which is what rank fusion is for,
    and the opposite of what the old rule reported.

    Symmetric on purpose: a dense half supplying everything is the same failure
    as a sparse half supplying everything, and the old rule could only ever see
    one of the two.
    """
    contributed = [
        (r.get("dense_score") or 0.0, r.get("sparse_score") or 0.0)
        for r in refs
    ]
    # A chunk with neither signal reported says nothing about the merge.
    scored = [(d, s) for d, s in contributed if d > 0 or s > 0]
    if len(scored) < _MIN_MERGE_OBSERVATIONS:
        return None

    sparse_only = sum(1 for d, s in scored if d == 0 and s > 0)
    dense_only = sum(1 for d, s in scored if s == 0 and d > 0)
    for count, half, other in (
        (sparse_only, "keyword", "semantic"),
        (dense_only, "semantic", "keyword"),
    ):
        share = count / len(scored)
        if share > _MERGE_DOMINANCE_SHARE:
            return DiagnosticItem(
                id="bm25_dominance",
                severity="info",
                title=f"The {half} half rules the merge",
                detail=f"{count}/{len(scored)} of the chunks that reached the answer were "
                f"contributed by the {half} half alone ({share:.0%}). The {other} half is "
                "being paid for and is barely reaching the context.",
                action="Compare the two halves' own recall separately before changing the "
                       "merge: one of them may simply have nothing to add on this corpus.",
            )
    return None


# A merge needs a few observations before its balance means anything; below
# this a single question's top-k decides the verdict.
_MIN_MERGE_OBSERVATIONS = 20
# Measured on this machine: a working rank fusion puts 11% to 47% of the final
# context on the sparse half. The threshold sits well clear of that band, so
# the check reports a half that has effectively stopped contributing rather
# than one that contributes less than the other.
_MERGE_DOMINANCE_SHARE = 0.8


def detect_empty_answers(answers: list[str]) -> DiagnosticItem | None:
    """High fraction of empty / 'not found' answers ⇒ coverage or generation issue.

    Coarse signal: a high refusal rate is sometimes the CORRECT behavior
    (e.g. a control set with deliberate out-of-scope/uncovered questions —
    see core/eval/answerability.py). This detector flags the raw rate as a
    starting point; detect_incorrect_refusals below is the precise version
    that knows which refusals were actually wrong.

    Not called at all for a run that stopped before the generator: see the
    gate in run_detectors, and the reason there.
    """
    if not answers:
        return None
    bad = [a for a in answers if not a.strip() or _NOT_FOUND.search(a)]
    frac = len(bad) / len(answers)
    if frac >= 0.3:
        return DiagnosticItem(
            id="empty_answers",
            severity="error" if frac >= 0.5 else "warn",
            title="Many empty or \"not found\" answers",
            detail=f"{len(bad)}/{len(answers)} answers are empty or say nothing was found ({frac:.0%}).",
            action="Check corpus coverage, and whether the model's thinking mode is consuming the num_predict budget.",
        )
    return None


def detect_incorrect_refusals(run: dict[str, Any]) -> DiagnosticItem | None:
    """Among questions scored with correct_refusal (Eval Measurement
    Trustworthiness, Phase 0 — see _CompositeEvaluator), flag the fraction
    that got the refuse/answer decision WRONG: refusing on an answerable
    question (a real retrieval failure), or answering confidently on a
    question the corpus can't actually support (a hallucination).

    Only present on runs evaluated with the Phase 0 composite evaluator —
    older runs (or non-control-question runs) silently have nothing to
    score here, so this detector returns None rather than a false alarm.
    """
    scores = [
        qr.get("metrics", {}).get("correct_refusal")
        for qr in run.get("question_results", [])
    ]
    scored = [s for s in scores if s is not None]
    if not scored:
        return None
    wrong = sum(1 for s in scored if s < 0.5)
    frac = wrong / len(scored)
    if frac >= 0.2:
        return DiagnosticItem(
            id="incorrect_refusals",
            severity="error" if frac >= 0.4 else "warn",
            title="Refusal decisions are going the wrong way",
            detail=f"{wrong}/{len(scored)} questions got the refusal decision wrong "
            f"({frac:.0%}): either a refusal on an answerable question, which is a "
            "retrieval miss, or a confident answer with no source in the corpus, "
            "which is a hallucination.",
            action="Read each question's answerability class (core.eval.answerability). "
            "For answerable ones, ask why retrieval missed the source; for "
            "uncovered or out_of_scope ones, ask why the model did not refuse.",
        )
    return None


_LAYER_LABELS = {
    "retrieval": "retrieval",
    "rerank": "rerank",
    "generation": "generation",
    "suspected_ungrounded_answer": "suspected ungrounded answer",
}


def detect_layer_bottleneck(run: dict[str, Any]) -> DiagnosticItem | None:
    """Aggregates core/eval/funnel.py per-question verdicts across the run,
    surfacing which layer (retrieval/rerank/generation) explains the most
    failures — instead of requiring the engineer to open every question
    individually to find the pattern (Eval Measurement Trustworthiness,
    Phase 1).

    Pure aggregation, no filesystem/corpus access: answerability is
    inferred from which metric keys are present on each question
    (retrieval_recall_at_k present => answerable) — exactly how
    _CompositeEvaluator already gates them, so no need to re-derive it from
    article_refs/corpus here.
    """
    from collections import Counter

    from core.eval.funnel import diagnose_question

    counts: Counter[str] = Counter()
    total_applicable = 0
    for qr in run.get("question_results", []):
        metrics = qr.get("metrics") or {}
        answerability = "answerable" if "retrieval_recall_at_k" in metrics else "not_applicable"
        verdict = diagnose_question(answerability, metrics, metrics.get("pre_rerank_recall_at_k"))
        if verdict.layer == "not_applicable":
            continue
        total_applicable += 1
        counts[verdict.layer] += 1

    if total_applicable == 0:
        return None

    failing = {layer: n for layer, n in counts.items() if layer != "ok"}
    if not failing:
        return None

    worst_layer, worst_count = max(failing.items(), key=lambda kv: kv[1])
    frac = worst_count / total_applicable
    if frac < 0.2:
        return None

    label = _LAYER_LABELS.get(worst_layer, worst_layer)
    return DiagnosticItem(
        id="layer_bottleneck",
        severity="error" if frac >= 0.4 else "warn",
        title=f"Funnel bottleneck: {label}",
        detail=f"{worst_count}/{total_applicable} answerable questions failed at the "
        f"\"{label}\" layer ({frac:.0%}).",
        action="Open the individual questions carrying this verdict on the run page to see "
        "the specific failures and decide what to fix.",
    )


# ── orchestrator ───────────────────────────────────────────────────────────────

def detect_unverified_coverage(run: dict[str, Any]) -> DiagnosticItem | None:
    """Coverage was not verified against the index, so answerability classes
    (and therefore which questions count toward retrieval metrics) rest on the
    dataset's own refs rather than on what is actually retrievable.

    This exists because the un-verified state used to be invisible:
    an unreadable corpus classified every question "uncovered", which drops it
    out of retrieval metrics entirely, so the aggregate silently became
    meaningless while every check still passed. Surfacing it is the whole
    point — a reader must be able to tell a verified zero from an unverified
    one.

    Silent for runs stored before the field existed (empty dict) and for runs
    where verification did happen: nothing to report is not a warning.
    """
    coverage = run.get("coverage_check") or {}
    if not coverage or coverage.get("checked"):
        return None
    reason = coverage.get("reason") or "no reason given"
    return DiagnosticItem(
        id="unverified_coverage",
        severity="warn",
        title="Corpus coverage was not verified",
        detail=(
            f"Answerability classes were taken from the dataset's own refs without checking them "
            f"against the index ({reason}). A question whose source is absent from the index counts "
            "as answerable in this run, so the uncovered count may be lower than the truth."
        ),
        action="Check that the index for this corpus is reachable, then re-run to get verified classes.",
    )


def detect_undeclared_metric(run: dict[str, Any]) -> DiagnosticItem | None:
    """A number reaches the reader with a name and nothing else.

    A metric whose definition drifts from its name reads perfectly well: the
    name is the only thing on the screen, and the name still says what it
    always said. Nothing here decides that a name is right. What it reports
    is the state in which the question cannot be asked at all, which is the
    state every such drift hides in.

    Silent for a run with no aggregate at all: a run that measured nothing
    is not a run whose measurements are undeclared.
    """
    from core.eval.metric_definitions import definition_of

    aggregate = run.get("aggregate_metrics") or {}
    undeclared = sorted(name for name in aggregate if definition_of(name) is None)
    if not undeclared:
        return None
    return DiagnosticItem(
        id="undeclared_metric",
        severity="warn",
        title="A metric carries no definition",
        detail=(
            f"{len(undeclared)} of {len(aggregate)} metrics in this run declare nothing about "
            f"what they compute or over which questions: {', '.join(undeclared)}. Their names "
            "are all a reader has, so a name that stopped matching its definition would look "
            "exactly like this run does."
        ),
        action="Declare each metric beside its name, saying what it computes and over which "
               "questions, so the name can be checked against it.",
    )


def detect_aggregate_disagrees_with_questions(run: dict[str, Any]) -> DiagnosticItem | None:
    """The run-level number is not the number the questions carry.

    Every aggregate here is the mean of the per-question values, and nothing
    ever compared the two, so a question lost between writing the run and
    reading it moves the number on screen and moves nothing else: the reader
    blames what was measured and the fault is in the measuring.

    Measured across the forty-one runs stored when this was written, every
    aggregate equalled the mean of its per-question values to within a
    millionth, so the tolerance below is an observation. A metric declared
    as something other than a mean is skipped instead of guessed at, which
    is why the declaration carries the operation and not only the name.
    """
    from core.eval.metric_definitions import definition_of

    aggregate = run.get("aggregate_metrics") or {}
    questions = run.get("question_results") or []
    if not aggregate or not questions:
        return None

    disagreements: list[str] = []
    for name, recorded in sorted(aggregate.items()):
        definition = definition_of(name)
        if definition is not None and definition.aggregated_by != "mean":
            continue
        if not isinstance(recorded, int | float):
            continue
        values = [
            value for qr in questions
            if isinstance(value := (qr.get("metrics") or {}).get(name), int | float)
        ]
        if not values:
            disagreements.append(f"{name}: {recorded:.4f} in the run and on no question at all")
            continue
        mean = sum(values) / len(values)
        if abs(mean - recorded) > 1e-6:
            disagreements.append(
                f"{name}: {recorded:.4f} in the run and {mean:.4f} across the "
                f"{len(values)} questions carrying it"
            )

    if not disagreements:
        return None
    return DiagnosticItem(
        id="aggregate_disagrees",
        severity="error",
        title="The run's numbers are not its questions' numbers",
        detail=(
            f"{len(disagreements)} of {len(aggregate)} metrics do not survive being recomputed "
            f"from the questions of this same run: {'; '.join(disagreements)}. Whatever these "
            "numbers describe, it is not what the questions recorded."
        ),
        action="Compare what the run wrote with what a read of it returns; a question lost on "
               "the way out moves the aggregate and leaves everything else looking correct.",
    )


def detect_metric_without_grounds(run: dict[str, Any]) -> DiagnosticItem | None:
    """A number computed where it has nothing to be about.

    Not a wrong number, which could be argued with: a confident number about
    something nobody measured. A retrieval-only run scored for the quality of
    an answer it never generated, a recall computed for a question the corpus
    does not cover, a citation judged correct against sources that were never
    found.

    The evaluator enforces one of these by hand, inside itself, where a
    reader cannot see it and nothing can check it. The preconditions are
    declared beside each metric now, and this reads them.
    """
    from core.eval.metric_definitions import definition_of

    questions = run.get("question_results") or []
    ungrounded: dict[str, tuple[str, int]] = {}
    for question in questions:
        for name in (question.get("metrics") or {}):
            definition = definition_of(name)
            if definition is None:
                continue
            for precondition in definition.requires:
                if precondition.holds(run, question):
                    continue
                says, count = ungrounded.get(name, (precondition.says, 0))
                ungrounded[name] = (says, count + 1)
                break

    if not ungrounded:
        return None
    told = "; ".join(
        f"{name} on {count} question(s) where {says} does not hold"
        for name, (says, count) in sorted(ungrounded.items())
    )
    return DiagnosticItem(
        id="metric_without_grounds",
        severity="error",
        title="A metric was computed where it has no grounds",
        detail=(
            f"{len(ungrounded)} metric(s) of this run were recorded against questions that "
            f"cannot support them: {told}. Averaged into the run's numbers, these are "
            "confident values about something nobody measured."
        ),
        action="Read each metric's declared preconditions and stop recording it where they "
               "do not hold, so the average is over the questions the number is about.",
    )


def detect_chunk_id_collision(run: dict[str, Any]) -> DiagnosticItem | None:
    """One identifier standing for two different fragments.

    An identifier is derived from where a fragment came from, and a lost
    source path collapses two derivations onto one value. Retrieval then
    returns the wrong text under the right identifier, deduplication drops a
    fragment that was never a duplicate, and every metric keyed on the
    identifier agrees with itself while describing the wrong fragment.

    Read off the fragments a run returned, so no index access is needed: two
    fragments carrying one identifier and different text is the collision
    itself and not a symptom of it. Silent across forty-six thousand
    identifiers in the runs stored when this was written.
    """
    texts: dict[str, set[str]] = {}
    for qr in run.get("question_results", []):
        for key in ("source_refs", "pre_rerank_source_refs", "candidate_source_refs"):
            for ref in qr.get(key) or []:
                chunk_id = ref.get("chunk_id")
                if not chunk_id:
                    continue
                # A prefix is enough to tell two fragments apart and keeps a
                # long run from holding every fragment's full text at once.
                texts.setdefault(chunk_id, set()).add((ref.get("chunk_text") or "")[:400])

    colliding = sorted(chunk_id for chunk_id, seen in texts.items() if len(seen) > 1)
    if not colliding:
        return None
    return DiagnosticItem(
        id="chunk_id_collision",
        severity="error",
        title="One fragment identifier, two fragments",
        detail=(
            f"{len(colliding)} of {len(texts)} fragment identifiers in this run stand for more "
            f"than one text, the first being {colliding[0]}. Anything keyed on the identifier "
            "is describing whichever fragment it saw last."
        ),
        action="Check how the identifier is derived at load time: a source path missing from "
               "the derivation collapses fragments of different documents onto one value.",
    )


#: A stage of the pipeline, the evidence that it ran, and the field that
#: would say what it cost. Only stages the platform times itself are listed:
#: a system reporting no trace at all is a different finding, and
#: core/eval/trace_completeness.py already makes it.
_TIMED_STAGES: tuple[tuple[str, str, str], ...] = (
    ("reranking", "n_reranked", "rerank_ms"),
    ("generation", "output_tokens", "generate_ms"),
    # The sparse half having returned anything is what says a merge
    # happened. `n_merged` is not: it is set to the size of whatever
    # retrieval produced, on every pipeline, merge or no merge, so reading
    # it as evidence made this detector report every run ever stored.
    ("merging the two halves", "n_sparse", "merge_ms"),
)


def detect_unmeasured_stage_cost(run: dict[str, Any]) -> DiagnosticItem | None:
    """A stage that ran and reported no time.

    What a stage costs is the whole of the argument for keeping it, and a
    stage whose cost is not recorded is defended by nobody and questioned by
    nobody. The reranker is the usual one: it improves the ranking, it is
    kept, and how many milliseconds it adds to every question is a number
    nobody has.

    Deliberately silent when the run carries no stage trace whatsoever.
    That is a system reporting nothing about itself, which is a wider
    finding, already named by the trace-completeness gaps; repeating it here
    would put two sentences about one absence on the same screen.
    """
    traces = [qr.get("stage_trace") or {} for qr in run.get("question_results", [])]
    traces = [trace for trace in traces if trace]
    if not traces:
        return None

    unmeasured: list[str] = []
    for label, evidence, timing in _TIMED_STAGES:
        ran = sum(1 for trace in traces if float(trace.get(evidence) or 0) > 0)
        if not ran:
            continue
        timed = sum(1 for trace in traces if float(trace.get(timing) or 0) > 0)
        if timed == 0:
            unmeasured.append(f"{label}, which ran on {ran} of {len(traces)} questions")

    if not unmeasured:
        return None
    return DiagnosticItem(
        id="unmeasured_stage_cost",
        severity="warn",
        title="A stage ran and reported no time",
        detail=(
            f"{len(unmeasured)} stage(s) of this run left no duration behind: "
            f"{'; '.join(unmeasured)}. What the stage costs cannot be weighed against what it "
            "buys, so keeping it is a decision nobody can check."
        ),
        action="Record a duration for every stage the pipeline runs, so its price can be "
               "compared with the gain it is kept for.",
    )


def run_detectors(run: dict[str, Any]) -> list[DiagnosticItem]:
    """Run all silent-degradation detectors over a finished run dict."""
    refs = _all_source_refs(run)
    answers = _generated_answers(run)
    # dense_score/sparse_score are populated only by the platform's own
    # in-process retrieval (core/retrieval/hybrid.py) — an external RAG
    # (pipeline_source="http") never reports a per-signal dense/sparse
    # split at all, so those fields land on SourceRef's plain-float defaults
    # (0.0), indistinguishable from a genuinely-collapsed dense score. Found
    # live: detect_stub_embedder fired "5/5 chunks dense_score≈0" against a
    # real, working external-RAG answer whose actual (fused) score was a healthy
    # 6.29 — the platform simply has no visibility into an external RAG's
    # own internal retrieval split. detect_bm25_dominance reads the same two
    # fields the same way, so it carries the identical false-positive risk.
    # Both skipped for http-sourced runs.
    is_external = (run.get("config") or {}).get("pipeline_source") == "http"

    # A run that deliberately stopped before the generator answers nothing, and
    # that is the documented intent of the mode and not a fault of it. The
    # evaluator already refuses to score an empty answer for exactly this
    # reason; this detector did not, and reported "100% of answers are empty"
    # at severity error on every one of the thirty-six stored retrieval-only
    # runs. The gate sits here and not inside the detector because the
    # detector is given answers alone and has no way to know why they are empty.
    retrieval_only = bool((run.get("config") or {}).get("retrieval_only"))

    candidates = [
        # No external special case any more: detect_stub_embedder now decides
        # from whether the run recorded a per-signal split at all, which is the
        # same question the special case was standing in for, asked of the data
        # instead of the configuration.
        detect_stub_embedder(refs),
        detect_duplicates(run),
        detect_header_only(refs),
        None if is_external else detect_bm25_dominance(refs),
        None if retrieval_only else detect_empty_answers(answers),
        detect_incorrect_refusals(run),
        detect_layer_bottleneck(run),
        detect_unverified_coverage(run),
        detect_undeclared_metric(run),
        detect_aggregate_disagrees_with_questions(run),
        detect_chunk_id_collision(run),
        detect_metric_without_grounds(run),
        detect_unmeasured_stage_cost(run),
    ]
    return [c for c in candidates if c is not None]
