"""A document the owner of a system can work from.

The earlier stages produce a cause per failure, an ordered list of work, and a
measured payoff for one proposed change. All of it is readable on a page
inside the platform, which is useless to the one person who most needs it:
whoever owns a system the platform does not.

The exit criterion of phase 4 is that such an owner receives a document to
work from rather than a score. A score says the system is worse than it
should be, which its owner already knew. A prescription says which document
to fix, how many questions that closes, and which questions to re-run to
check — and every one of those is already computed by the time this module
assembles them.

Three constraints shaped it.

**Nothing here is new evidence.** Assembly only. A prescription that could
assert something the run did not measure would be a recommendation dressed
as a finding, which is exactly the failure the funnel was cleared of.

**What cannot be diagnosed is named.** An owner reading a short prescription
must be able to tell "little is wrong" from "little is visible", and only
core/eval/trace_completeness.py's gaps distinguish them.

**It renders to text.** The recipient has no access to the platform, so the
artefact has to survive being pasted into an email.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.eval.trace_completeness import TraceGap, assess_trace_completeness, diagnosis_depth

# How many example questions accompany one task. Enough to show the pattern,
# few enough that the document stays readable; an owner who wants all of them
# has the full id list in the structured form.
_EXAMPLES_PER_TASK = 3


@dataclass(frozen=True)
class PrescribedFix:
    """One piece of work, with what it costs to check and what it buys."""

    cause: str
    lever: str
    entity: str
    questions: int
    examples: tuple[str, ...] = ()
    verification_question_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "lever": self.lever,
            "entity": self.entity,
            "questions": self.questions,
            "examples": list(self.examples),
            "verification_question_ids": list(self.verification_question_ids),
        }


@dataclass(frozen=True)
class Prescription:
    run_id: str
    dataset_name: str
    n_questions: int
    diagnosis_depth: str
    fixes: tuple[PrescribedFix, ...] = ()
    gaps: tuple[TraceGap, ...] = ()
    # The measured counterfactual from phase 3, when the run recorded enough
    # to compute one. Carried verbatim rather than re-derived.
    context_size_advice: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "n_questions": self.n_questions,
            "diagnosis_depth": self.diagnosis_depth,
            "fixes": [f.to_dict() for f in self.fixes],
            "gaps": [g.to_dict() for g in self.gaps],
            "context_size_advice": self.context_size_advice,
            "metrics": self.metrics,
        }


def build_prescription(payload: dict[str, Any]) -> Prescription:
    """Assembles one run's already-computed findings into a prescription.

    ``payload`` is the `GET /experiments/{run_id}` response, so every input
    is a value the platform already stands behind. Nothing is recomputed
    here, which is what keeps the document and the page from disagreeing.
    """
    question_results = payload.get("question_results") or []
    by_id = {str(qr.get("question_id") or ""): qr for qr in question_results}

    fixes = []
    for task in payload.get("fix_tasks") or []:
        ids = [str(i) for i in task.get("question_ids") or []]
        examples = [
            (by_id.get(i, {}).get("question") or i)
            for i in ids[:_EXAMPLES_PER_TASK]
        ]
        fixes.append(PrescribedFix(
            cause=str(task.get("cause") or ""),
            lever=str(task.get("lever") or ""),
            entity=str(task.get("entity") or ""),
            questions=int(task.get("questions") or 0),
            examples=tuple(examples),
            # Every question of the task, not only the examples: these are
            # what the owner re-runs to check, and checking a sample would
            # let a partial fix pass as a whole one.
            verification_question_ids=tuple(ids),
        ))

    return Prescription(
        run_id=str(payload.get("run_id") or ""),
        dataset_name=str(payload.get("dataset_name") or ""),
        n_questions=len(question_results),
        diagnosis_depth=diagnosis_depth(question_results),
        fixes=tuple(fixes),
        gaps=tuple(assess_trace_completeness(question_results)),
        context_size_advice=payload.get("context_size_advice"),
        metrics=dict(payload.get("aggregate_metrics") or {}),
    )


_CAUSE_RU = {
    "data_missing": "the source is absent from the index",
    "ranking": "the source is found but does not reach the selection",
    "chunking": "the source is split so that no piece sits close to the question",
    "not_retrievable": "the source is not found; the wording diverges",
    "unknown": "no cause established",
}

_LEVER_RU = {
    "ingest": "load or re-index the document",
    "ranking": "widen the selection, or tune the ranking",
    "chunking": "align the chunk boundaries with the reference unit",
    "vocabulary": "bring question wording and document terminology closer together",
    "chunking_or_vocabulary": "check both the chunk boundaries and the wording gap",
    "verify_index": "establish whether the source is in the index at all",
}


def render_markdown(prescription: Prescription) -> str:
    """The document itself.

    Written for someone with no access to the platform, so it repeats what a
    page would leave implicit: which run it came from, what was measured, and
    what was not observable at all.
    """
    p = prescription
    lines = [
        f"# Prescription for run {p.run_id}",
        "",
        f"Golden set: {p.dataset_name or 'unnamed'}. Questions: {p.n_questions}.",
        "",
    ]

    if p.metrics:
        recall = p.metrics.get("retrieval_recall_at_k")
        if recall is not None:
            lines += [f"Retrieval recall at the current configuration: {recall:.2f}.", ""]

    # Placed before the findings, not after: a reader who does not know what
    # was invisible will read a short list as good news.
    if p.gaps:
        lines += [
            "## What the platform could not check",
            "",
            "Nothing below states that these are free of problems. It states that "
            "what the system reports about itself does not allow them to be checked.",
            "",
        ]
        for gap in p.gaps:
            lines += [f"- **{gap.field}**. {gap.unavailable} To change that: {gap.remedy}"]
        lines += [""]

    if not p.fixes:
        lines += [
            "## No work assigned",
            "",
            "No failure resolved to a cause, so there is nothing to prescribe.",
            "",
        ]
    else:
        lines += ["## What to fix, in descending order of gain", ""]
        for i, fix in enumerate(p.fixes, start=1):
            cause = _CAUSE_RU.get(fix.cause, fix.cause)
            lever = _LEVER_RU.get(fix.lever, fix.lever)
            lines += [
                f"### {i}. {fix.entity or 'source not named'}",
                "",
                f"Cause: {cause}.",
                f"Action: {lever}.",
                f"Questions it would close: {fix.questions}.",
                "",
            ]
            if fix.examples:
                lines += ["Example questions:", ""]
                lines += [f"- {e}" for e in fix.examples]
                lines += [""]
            lines += [
                "How to verify: re-run and confirm that the listed questions "
                f"({len(fix.verification_question_ids)} of them) stop failing.",
                "",
            ]

    advice = p.context_size_advice
    if advice:
        lines += [
            "## A configuration change whose gain is already measured",
            "",
            f"Raise the context size from {advice.get('current_k')} to "
            f"{advice.get('recommended_k')}. Questions it would close: "
            f"{advice.get('questions_gained')}.",
            "",
            "That figure comes from the stored run rather than from an assumption: the "
            "run records the rank at which each required source actually sat.",
            "",
        ]
        unreachable = advice.get("unreachable") or 0
        if unreachable:
            lines += [
                f"Questions no context size would close: {unreachable}. "
                "Their source was not found at all.",
                "",
            ]

    return "\n".join(lines).rstrip() + "\n"


# ── The acceptance set, and the verdict over two runs ────


@dataclass(frozen=True)
class AcceptanceVerdict:
    """Whether a prescription was carried out, judged over the questions it
    itself named.

    The acceptance set is not a new artefact to maintain: it is
    the set of question ids the prescription listed as verification. That
    makes it fixed at the moment the prescription is written, which is the
    property acceptance needs — a criterion the recipient could still widen
    or narrow after the fact is not one.
    """

    accepted: bool
    fixed: tuple[str, ...] = ()
    still_failing: tuple[str, ...] = ()
    regressed: tuple[str, ...] = ()
    # Named in the prescription but absent from one of the runs, so the
    # comparison cannot speak to them. Kept separate from failures: an
    # unchecked question is not a failed one, and merging the two would let
    # a shrunken dataset read as a passed acceptance.
    unchecked: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        return (
            f"fixed {len(self.fixed)}, "
            f"still failing {len(self.still_failing)}, "
            f"regressed {len(self.regressed)}, "
            f"unchecked {len(self.unchecked)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "fixed": list(self.fixed),
            "still_failing": list(self.still_failing),
            "regressed": list(self.regressed),
            "unchecked": list(self.unchecked),
            "summary": self.summary,
        }


def _ok(question_result: dict[str, Any]) -> bool:
    """Same definition of "this question is fine" the regression machinery
    uses: the funnel verdict is exactly `ok`. Reused rather than restated so
    an acceptance verdict cannot disagree with a regression report about the
    same pair of runs."""
    return ((question_result.get("funnel") or {}).get("layer")) == "ok"


def judge_acceptance(
    acceptance_question_ids: list[str],
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> AcceptanceVerdict:
    """The verdict, over exactly the questions the prescription
    named.

    Acceptance requires two things, and the second is the one usually
    forgotten: every named question now passes, **and** none of them got
    worse. A change that closes four questions and breaks two is not a
    partial success at this boundary, because the owner was asked for a
    specific outcome and did not deliver it.

    A question the prescription named but neither run contains counts as
    unchecked and blocks acceptance. Treating it as passed would let a
    shrunken dataset read as a completed prescription.
    """
    before_by_id = {str(q.get("question_id") or ""): q for q in before}
    after_by_id = {str(q.get("question_id") or ""): q for q in after}

    fixed, still_failing, regressed, unchecked = [], [], [], []
    for qid in acceptance_question_ids:
        b, a = before_by_id.get(qid), after_by_id.get(qid)
        if b is None or a is None:
            unchecked.append(qid)
            continue
        was_ok, is_ok = _ok(b), _ok(a)
        if is_ok and not was_ok:
            fixed.append(qid)
        elif is_ok:
            continue  # was fine, still fine — nothing to report
        elif was_ok:
            regressed.append(qid)
        else:
            still_failing.append(qid)

    accepted = not (still_failing or regressed or unchecked)
    return AcceptanceVerdict(
        accepted=accepted,
        fixed=tuple(fixed),
        still_failing=tuple(still_failing),
        regressed=tuple(regressed),
        unchecked=tuple(unchecked),
    )
