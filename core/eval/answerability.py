"""Answerability classification for golden control questions.

A question's `article_refs` are the ground truth for what should be
retrieved. Whether the corpus can actually answer the question is a
property of (refs, corpus), not of any particular RAG run, so it is
classified once and every run is scored against the same three buckets:

- "answerable"   — the refs resolve against what is indexed. Score
                   retrieval/answer quality here.
- "uncovered"    — at least one ref does NOT resolve (a real source gap:
                   the referenced unit was never ingested). A refusal here
                   is correct, but it reflects corpus incompleteness, not
                   RAG quality.
- "out_of_scope" — no article_refs at all (e.g. "What's today's USD rate?").
                   A refusal here is correct by design.

Both "uncovered" and "out_of_scope" feed correct_refusal; only "answerable"
feeds retrieval_recall@k / answer_similarity. That gating is why a wrong
classification is not a cosmetic problem: it silently removes a question
from every retrieval metric.

**What changed, and why.** Classification used to take a
filesystem path (one corpus's own directory) and check for files under it. That path
belonged to one Realm's own corpus but was applied to every Realm's run,
described the platform's own disk rather than the index the served system
searches, and was excluded from version control — so on a fresh checkout
every ref failed to resolve and every question of that Realm silently
became "uncovered", zeroing retrieval metrics with no error raised
anywhere.
the design notes records four earlier incidents from the same
coupling. Classification now takes a `core/eval/ref_resolution.py`
`RefResolver`, which answers against the index and can say "unknown" when
it cannot answer at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from core.eval.ref_resolution import RefResolver, UnknownRefResolver

Answerability = Literal["answerable", "uncovered", "out_of_scope"]

_VALID_CLASSES: frozenset[str] = frozenset({"answerable", "uncovered", "out_of_scope"})


@dataclass(frozen=True)
class AnswerabilityVerdict:
    """Classification plus how it was reached.

    `coverage_checked` is False when no ref could actually be verified —
    either because no resolver was available or because the resolver
    returned "unknown". The class is still usable (it degrades to trusting
    the refs), but a caller that presents it as verified coverage would be
    lying, which is exactly the failure this rework removes.

    `missing_refs` lists the refs that resolved to "absent", so a reviewer
    reading an "uncovered" verdict can see which source unit is missing
    instead of only that something is.
    """

    answerability: Answerability
    coverage_checked: bool
    missing_refs: tuple[str, ...] = field(default=())


def classify_answerability(
    article_refs: list[str], resolver: RefResolver | None
) -> AnswerabilityVerdict:
    """Classify one question's refs against what the resolver can see.

    Aggregation over several refs, in priority order:

    1. No refs at all means "out_of_scope" — no ground truth was ever
       recorded for this question. This is a dataset-authoring gap, not a
       corpus gap, and the two must not be conflated (an earlier incident
       displayed identical wording for both, see the design notes).
    2. Any ref "absent" means "uncovered": the question demands a source
       unit the index does not contain, so no amount of ranking work can
       answer it.
    3. Otherwise, any ref "unknown" means the refs are trusted
       ("answerable") but `coverage_checked` is False.
    4. Otherwise every ref is "present": "answerable", verified.

    Rule 2 outranks rule 3 deliberately: a definite miss is information,
    an unverifiable ref is not, so one confirmed gap decides the verdict
    even when other refs could not be checked.
    """
    if not article_refs:
        return AnswerabilityVerdict(answerability="out_of_scope", coverage_checked=True)

    active = resolver if resolver is not None else UnknownRefResolver()

    missing: list[str] = []
    any_unknown = False
    for ref in article_refs:
        presence = active.presence(ref)
        if presence == "absent":
            missing.append(ref)
        elif presence == "unknown":
            any_unknown = True

    if missing:
        return AnswerabilityVerdict(
            answerability="uncovered", coverage_checked=True, missing_refs=tuple(missing)
        )
    return AnswerabilityVerdict(answerability="answerable", coverage_checked=not any_unknown)


def resolve_answerability_verdict(
    question: dict, resolver: RefResolver | None
) -> AnswerabilityVerdict:
    """Full verdict for one dataset row.

    An explicit per-row `answerability` field wins outright and is reported
    as checked, because its author computed it against whatever corpus that
    dataset actually belongs to — this is the only way an external RAG's
    dataset can be classified at all, since the platform never sees that
    RAG's corpus. Existing datasets carrying the field
    therefore classify exactly as before, with zero migration.
    """
    explicit = question.get("answerability")
    if explicit in _VALID_CLASSES:
        return AnswerabilityVerdict(answerability=explicit, coverage_checked=True)  # type: ignore[arg-type]
    return classify_answerability(question.get("article_refs") or [], resolver)


def resolve_answerability(question: dict, resolver: RefResolver | None) -> Answerability:
    """Class only, for the many callers that need nothing else."""
    return resolve_answerability_verdict(question, resolver).answerability


def classify_dataset(
    rows: list[dict], resolver: RefResolver | None
) -> dict[str, Answerability]:
    """Classify every question in a loaded golden set. Returns {id: class}."""
    return {row["id"]: resolve_answerability(row, resolver) for row in rows}
