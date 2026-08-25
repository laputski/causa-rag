"""Splitting a retrieval failure into its actual cause.

`core/eval/funnel.py` answers "on which layer did this question go wrong"
and, for a large share of failures, says `retrieval`. That verdict names a
layer, not a cause, and the three causes hiding behind it call for opposite
work:

- the expected source is not in the index at all, so no amount of ranking
  work can help and the fix is in ingestion;
- it is in the index and retrieval does surface it, just below the
  configured cut-off, so the fix is ranking or top_k;
- it is in the index but retrieval does not surface it even at a much wider
  k, so the query and the text are not close to each other at all.

Told only "retrieval", an engineer tunes ranking. When the document is
absent, that work is spent where no data exists. Making the distinction is
the whole point of this module.

Pure: no I/O, no LLM. The caller gathers the two pieces of evidence — what
`core/eval/ref_resolution.py` says about the expected refs, and what a
widened re-query found — and this decides what they mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Cause = Literal[
    "data_missing",
    "ranking",
    "chunking",
    "not_retrievable",
    "unknown",
]

# `chunking` was carved out of `not_retrievable`, which had deliberately been
# one bucket because the evidence available at the time could not tell the two
# apart.
#
# Both mean "indexed, yet not surfaced even at a wide k". What separates them
# is how many chunks the source unit occupies. A unit split across several
# chunks gives each fragment only part of the unit's content, so none of them
# need be close to a question the whole unit answers — that is a chunking
# problem. A unit that is a single chunk has all its text in one place and
# still is not close, which leaves the wording, not the splitting.
#
# `not_retrievable` therefore survives with a narrower meaning, and stays the
# verdict whenever the count is unavailable: not knowing how many chunks a
# unit occupies must not be reported as knowing it occupies one.


# The lever a cause points at. A stable identifier rather than
# prose, so the interface can name it in the reader's own language while the
# mapping itself stays here, beside the reasoning that produced the cause.
#
# The mapping follows one order of preference: fix the data first, tune
# ranking only once the data is right, and treat a miss at a wide k as a
# question about chunking or vocabulary rather than about the ranker.
Lever = Literal[
    "ingest",
    "ranking",
    "chunking",
    "vocabulary",
    # Kept for a window of older runs, stored when the two above could not be
    # told apart. Never produced now, still understood.
    "chunking_or_vocabulary",
    "verify_index",
]

_LEVER_BY_CAUSE: dict[str, Lever] = {
    "data_missing": "ingest",
    "ranking": "ranking",
    "chunking": "chunking",
    # The default for `not_retrievable` stays the undecided lever, because
    # the cause is produced both when a unit is known to be a single chunk
    # (wording, then) and when the count is unavailable (nothing is known).
    # `classify_retrieval_cause` passes the sharper lever when it has the
    # evidence for it; this map only supplies the safe answer.
    "not_retrievable": "chunking_or_vocabulary",
    "unknown": "verify_index",
}


def lever_for(cause: str) -> Lever:
    """Which lever a cause points at.

    An unrecognised cause falls back to verifying the index rather than to
    recommending a change: not knowing what went wrong is a reason to
    establish the ground truth, never a reason to start editing something.
    """
    return _LEVER_BY_CAUSE.get(cause, "verify_index")


@dataclass
class RootCause:
    cause: Cause
    detail: str
    # Left None by most construction sites and derived from the cause in
    # __post_init__, so a cause added later cannot ship without a lever. Set
    # explicitly only where the evidence supports something sharper than the
    # cause alone implies (see `not_retrievable` in the map above).
    lever: Lever | None = None
    # What the verdict was derived from, kept so a reader can disagree with
    # the conclusion without re-running anything.
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lever is None:
            self.lever = lever_for(self.cause)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "detail": self.detail,
            "lever": self.lever,
            "evidence": self.evidence,
        }


def classify_retrieval_cause(
    presences: dict[str, str],
    miss: dict[str, Any] | None,
    top_k: int,
    chunk_counts: dict[str, int | None] | None = None,
) -> RootCause:
    """Turns evidence about one failed question into a cause.

    ``presences`` maps each expected ref id to `present`/`absent`/`unknown`
    as reported by a `core/eval/ref_resolution.py` resolver. ``miss`` is the
    `core/eval/miss_diagnosis.py#diagnose_retrieval_miss` result for the same
    question, or None when the widened re-query could not be run. ``top_k``
    is what the run actually used, so a rank can be read against the cut-off
    the question was really judged by. ``chunk_counts`` maps each
    expected ref to how many chunks that source unit occupies, or to None
    where the resolver could not count.
    """
    if not presences:
        return RootCause(
            cause="unknown",
            detail="The question carries no reference refs, so no cause can be established.",
            evidence={},
        )

    absent = sorted(ref for ref, state in presences.items() if state == "absent")
    unknown = sorted(ref for ref, state in presences.items() if state == "unknown")

    # Checked before anything else: if the expected source is not indexed,
    # every retrieval observation about it is a consequence, not a cause.
    # Reporting a ranking problem here would send an engineer to tune a
    # ranker over data that does not exist.
    if absent:
        return RootCause(
            cause="data_missing",
            detail=(
                f"The reference source is absent from the index ({', '.join(absent)}). "
                "No ranking change can reach it; the fix belongs to ingestion."
            ),
            evidence={"refs": sorted(presences), "absent_refs": absent},
        )

    if unknown:
        # A distinction carried through from answerability: "could not check"
        # must not be reported as "checked and fine". Claiming a ranking cause
        # on an unverified index is exactly that false confidence.
        return RootCause(
            cause="unknown",
            detail=(
                "Whether the reference source is in the index could not be established "
                f"({', '.join(unknown)}), so the cause of the failure stays undetermined."
            ),
            evidence={"refs": sorted(presences), "unknown_refs": unknown},
        )

    if miss is None:
        return RootCause(
            cause="unknown",
            detail=(
                "The source is in the index, but the widened re-search could not be run, "
                "so the cause cannot be separated."
            ),
            evidence={"refs": sorted(presences), "presences": presences},
        )

    widened_k = miss.get("widened_k")
    if miss.get("found"):
        rank = miss.get("rank")
        return RootCause(
            cause="ranking",
            detail=(
                f"The source is in the index and the widened search finds it at rank {rank} "
                f"of {widened_k}, yet it does not reach the selected {top_k}. The cause lies in "
                "the ranking or in the selection size."
            ),
            evidence={
                "refs": sorted(presences), "rank": rank, "score": miss.get("score"),
                "widened_k": widened_k, "top_k": top_k,
            },
        )

    # The source is indexed and still not surfaced at a wide k.
    # How many chunks it occupies is what separates a splitting problem from
    # a wording one, and only the maximum matters: one split unit among the
    # expected refs is enough to explain the miss.
    counted = [c for c in (chunk_counts or {}).values() if isinstance(c, int)]
    split_into = max(counted, default=None)

    if split_into is not None and split_into > 1:
        return RootCause(
            cause="chunking",
            detail=(
                f"The source occupies {split_into} chunks in the index and is not found even "
                f"among {widened_k} candidates. Each piece carries only part of the unit, so no "
                "single one sits close to a question the whole unit answers. Two explanations "
                "fit: the split is too fine, or the reference ref is too broad and names a "
                "section no single chunk can retrieve."
            ),
            evidence={"refs": sorted(presences), "widened_k": widened_k, "split_into": split_into},
        )

    if split_into == 1:
        return RootCause(
            cause="not_retrievable",
            detail=(
                f"The source is in the index, occupies a single chunk, and is still not found "
                f"among {widened_k} candidates. Chunking plays no part here: the unit's whole "
                "text sits in one place. The question wording and the document terminology diverge."
            ),
            lever="vocabulary",
            evidence={"refs": sorted(presences), "widened_k": widened_k, "split_into": 1},
        )

    return RootCause(
        cause="not_retrievable",
        detail=(
            f"The source is in the index but is not found even among {widened_k} candidates. "
            "How many chunks the unit occupies is unknown, so the cause lies either in the "
            "chunking or in the gap between the question wording and the document terminology."
        ),
        evidence={"refs": sorted(presences), "widened_k": widened_k},
    )


def count_causes(question_results: list[dict[str, Any]]) -> dict[str, int]:
    """How many questions each cause accounts for, over a whole run.

    The aggregate is the reason this analysis runs for every failure rather
    than on demand for one question: a single verdict tells an engineer what
    went wrong once, whereas the counts tell them which kind of work would
    pay off most. That comparison is what phase 2 of the plan builds on.
    """
    counts: dict[str, int] = {}
    for qr in question_results:
        cause = (qr.get("root_cause") or {}).get("cause")
        if cause:
            counts[cause] = counts.get(cause, 0) + 1
    return counts
