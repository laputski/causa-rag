"""Silent-degradation detectors.

Deterministic, rule-based checks over a finished run. Each detector encodes a
real debugging incident (see docs/DEBUGGING.md) so the platform flags the failure
automatically instead of the engineer finding it by hand after a complaint.

Pure functions — no LLM, no I/O. Mirrored in ui/src/lib/diagnostics.ts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
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

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "action": self.action,
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
    """All dense scores ≈ 0 ⇒ corpus indexed with stub vectors (USE_REAL_BGE_M3)."""
    dense = [r.get("dense_score", 0.0) for r in refs if r.get("dense_score") is not None]
    nonzero = [d for d in dense if d > 0.01]
    if dense and len(nonzero) / len(dense) < 0.1:
        return DiagnosticItem(
            id="stub_embedder",
            severity="error",
            title="Looks like a stub embedder",
            detail=f"{len(dense) - len(nonzero)}/{len(dense)} chunks have dense_score near zero, "
            "so the corpus was probably indexed with random vectors.",
            action="Re-index the corpus with USE_REAL_BGE_M3=true.",
        )
    return None


def detect_duplicates(refs: list[dict[str, Any]]) -> DiagnosticItem | None:
    """Same chunk_text appears multiple times in retrieved context (re-ingest dupes)."""
    texts = [r.get("chunk_text", "") for r in refs if r.get("chunk_text")]
    if not texts:
        return None
    dupes = len(texts) - len(set(texts))
    if dupes > 0:
        return DiagnosticItem(
            id="duplicates",
            severity="warn",
            title="Duplicates in the context",
            detail=f"{dupes} repeated chunks in the retrieved context.",
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
    """Sparse score dominates dense everywhere ⇒ keyword bias."""
    pairs = [
        (r.get("dense_score", 0.0), r.get("sparse_score", 0.0))
        for r in refs
        if r.get("sparse_score")
    ]
    if len(pairs) < 3:
        return None
    dominated = [1 for d, s in pairs if s > (d * 3 + 1e-6)]
    if len(dominated) / len(pairs) > 0.7:
        return DiagnosticItem(
            id="bm25_dominance",
            severity="info",
            title="BM25 dominates the ranking",
            detail="Sparse scores sit well above dense ones for most chunks, which points to a keyword bias.",
            action="Switch to hybrid_rrf, or raise the dense weight in weighted merging.",
        )
    return None


def detect_empty_answers(answers: list[str]) -> DiagnosticItem | None:
    """High fraction of empty / 'not found' answers ⇒ coverage or generation issue.

    Coarse signal: a high refusal rate is sometimes the CORRECT behavior
    (e.g. a control set with deliberate out-of-scope/uncovered questions —
    see core/eval/answerability.py). This detector flags the raw rate as a
    starting point; detect_incorrect_refusals below is the precise version
    that knows which refusals were actually wrong.
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
    candidates = [
        None if is_external else detect_stub_embedder(refs),
        detect_duplicates(refs),
        detect_header_only(refs),
        None if is_external else detect_bm25_dominance(refs),
        detect_empty_answers(answers),
        detect_incorrect_refusals(run),
        detect_layer_bottleneck(run),
        detect_unverified_coverage(run),
    ]
    return [c for c in candidates if c is not None]
