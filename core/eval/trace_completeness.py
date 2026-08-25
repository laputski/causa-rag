"""What a system's trace does not say, and what that costs.

The platform's diagnostics are only as good as what a served system reports
about itself. For its own pipeline that is everything; for someone else's it
is whatever their responses happen to carry.

Silence about that is the failure mode this closes. A diagnostic that cannot
run and a diagnostic that ran and found nothing look identical in a report:
both are absent. An owner then reads "no rerank problems" from a platform
that structurally cannot detect one, which is worse than reading nothing.

So each gap is named together with the specific verdict it makes
unreachable, in the vocabulary of the phase that would have produced it.
Pure: reads stored question results, returns findings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceGap:
    """One thing the system does not report, and the diagnosis lost with it."""

    field: str
    # What cannot be established at all without it — not "would be nicer",
    # but "this verdict is structurally unreachable".
    unavailable: str
    # What the owner would change to close the gap. Stated because a gap
    # nobody can act on is a complaint rather than a finding.
    remedy: str

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "unavailable": self.unavailable, "remedy": self.remedy}


def _any_question_has(question_results: list[dict[str, Any]], key: str) -> bool:
    return any(qr.get(key) for qr in question_results)


def assess_trace_completeness(question_results: list[dict[str, Any]]) -> list[TraceGap]:
    """Gaps in what a run's questions actually carried.

    Judged from the responses rather than from a registration declaration,
    because a declaration states intent and a response states fact, and the
    two disagree often enough that only one of them is worth reporting.

    A run with no questions returns no gaps: nothing was observed, so nothing
    is missing — reporting every gap for an empty run would drown the real
    ones the first time somebody looked.
    """
    if not question_results:
        return []

    gaps: list[TraceGap] = []

    if not _any_question_has(question_results, "source_refs"):
        gaps.append(TraceGap(
            field="sources",
            unavailable=(
                "No retrieval metric can be computed, so recall, precision and whether "
                "the answer is grounded in a source all stay unknown."
            ),
            remedy="Return a list of sources in the answer to every question.",
        ))

    if not _any_question_has(question_results, "stage_trace"):
        gaps.append(TraceGap(
            field="stage_trace",
            unavailable=(
                "The per-stage latency breakdown is unavailable, so which stage spends "
                "the time cannot be said."
            ),
            remedy="Return a stage_trace field carrying each stage's duration.",
        ))

    if not _any_question_has(question_results, "pre_rerank_source_refs"):
        gaps.append(TraceGap(
            field="pre_rerank_source_refs",
            unavailable=(
                "A rerank failure is indistinguishable from a retrieval failure: whether "
                "the required source was found and then discarded cannot be established."
            ),
            remedy=(
                "Return the candidate list as it stood before reranking, whenever "
                "reranking runs."
            ),
        ))

    if not _any_question_has(question_results, "candidate_source_refs"):
        gaps.append(TraceGap(
            field="candidate_source_refs",
            unavailable=(
                "The gain from a different context size cannot be estimated: nothing "
                "records where the reference source ranked beyond the selection boundary."
            ),
            remedy=(
                "Return a candidate window wider than the final context, meaning fetch "
                "more than is passed into the answer."
            ),
        ))

    return gaps


def diagnosis_depth(question_results: list[dict[str, Any]]) -> str:
    """A one-word summary of how far diagnosis can go.

    Named rather than scored on purpose. A percentage would invite comparing
    two systems by it, and these are not degrees of the same thing: a system
    reporting sources but no stage trace and one reporting a stage trace but
    no sources are both partial, and neither is "more complete".
    """
    gaps = {g.field for g in assess_trace_completeness(question_results)}
    if not question_results:
        return "unknown"
    if "sources" in gaps:
        return "answer_only"
    if not gaps:
        return "full"
    return "partial"
