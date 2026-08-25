"""What the run would have shown under a different setting.

Prioritisation says which work would pay off most. It cannot say by how much,
because "how much" needs the answer to a question no verdict contains: what
would have happened at a different context size. Running the experiment
again answers it, at the cost of a full run per candidate value, which is
why nobody does it for more than one or two.

The observation this module rests on is that the answer is already in the
data. If a run recorded where the expected source actually sat in the ranked
list, then recall at any smaller-or-equal cut-off is arithmetic, not
retrieval. A curve over every plausible `top_k` costs one pass over stored
results.

Pure: takes ranks that were already measured, returns what they imply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.eval.retrieval_metrics import extract_ref_id


def first_hit_rank(expected_refs: list[str], ranked_refs: list[dict[str, Any]]) -> int | None:
    """1-based position of the first ranked entry matching an expected ref.

    None when no entry matches, which means no context size would have
    helped — the answer was not in the candidate window at all, and raising
    the cut-off cannot surface what was never fetched. Keeping that distinct
    from "rank beyond the current k" is the whole value of the curve: one is
    a tuning problem, the other is not.
    """
    if not expected_refs or not ranked_refs:
        return None
    expected = set(expected_refs)
    for rank, ref in enumerate(ranked_refs, start=1):
        if extract_ref_id(ref) in expected:
            return rank
    return None


@dataclass(frozen=True)
class ContextSizePoint:
    """One point of the curve: at this cut-off, this many questions would
    have had their expected source in context."""

    top_k: int
    questions_hit: int
    # Relative to the size the run actually used, so a reader sees the change
    # rather than having to subtract two numbers themselves.
    delta_vs_current: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "questions_hit": self.questions_hit,
            "delta_vs_current": self.delta_vs_current,
        }


def context_size_curve(
    ranks: list[int | None], current_k: int, max_k: int,
) -> list[ContextSizePoint]:
    """How many questions each candidate cut-off would have answered.

    ``ranks`` is one entry per *failed* question: the position at which its
    expected source was observed, or None where it was never observed at all.
    A question whose rank is None never contributes, at any cut-off.

    Evaluated at every integer from 1 to ``max_k`` rather than at a handful
    of round numbers, because the useful value is usually just past a cluster
    of ranks and rounding to 10/20/50 would step over it.
    """
    if max_k < 1:
        return []
    observed = sorted(r for r in ranks if r is not None)
    at_current = sum(1 for r in observed if r <= current_k)
    points = []
    for k in range(1, max_k + 1):
        hit = sum(1 for r in observed if r <= k)
        points.append(ContextSizePoint(
            top_k=k, questions_hit=hit, delta_vs_current=hit - at_current,
        ))
    return points


def payoff_of_context_size(ranks: list[int | None], current_k: int, proposed_k: int) -> int:
    """How many additional questions a proposed cut-off would close.

    Never negative: a larger window cannot lose a source a smaller one
    already contained. A *smaller* proposal legitimately returns a negative
    number, which is the point of asking.
    """
    observed = [r for r in ranks if r is not None]
    return sum(1 for r in observed if r <= proposed_k) - sum(1 for r in observed if r <= current_k)


@dataclass(frozen=True)
class ContextSizeAdvice:
    """The smallest cut-off that captures the whole available payoff."""

    current_k: int
    recommended_k: int
    questions_gained: int
    # Questions that no cut-off would have closed, because their expected
    # source never appeared in the candidate window. Stated alongside the
    # gain so a reader is not left thinking the rest are a tuning problem.
    unreachable: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_k": self.current_k,
            "recommended_k": self.recommended_k,
            "questions_gained": self.questions_gained,
            "unreachable": self.unreachable,
        }


def recommend_context_size(
    ranks: list[int | None], current_k: int, max_k: int,
) -> ContextSizeAdvice | None:
    """The smallest cut-off reaching the best payoff the data supports.

    Smallest rather than largest deliberately. Every extra chunk of context
    costs tokens, latency and an increased chance of the generator being
    distracted by an irrelevant passage, so a value that buys nothing beyond
    a smaller one is strictly worse. The largest useful rank is exactly that
    boundary.

    Returns None when nothing would be gained, so a caller shows advice only
    where there is advice to give rather than proposing the status quo.
    """
    observed = [r for r in ranks if r is not None]
    reachable = [r for r in observed if r <= max_k]
    unreachable = len(ranks) - len(reachable)
    gained = [r for r in reachable if r > current_k]
    if not gained:
        return None
    recommended = max(gained)
    return ContextSizeAdvice(
        current_k=current_k,
        recommended_k=recommended,
        questions_gained=len(gained),
        unreachable=unreachable,
    )
