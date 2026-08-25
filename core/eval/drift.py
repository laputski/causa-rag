"""What users actually ask, against what the golden set assumes.

Every failure the platform sees today comes from a run over a golden set,
which means it only ever sees questions somebody thought of in advance. A
question nobody anticipated cannot fail a test that does not exist, so the
most dangerous gap is precisely the one no metric moves for.

Production traces close that. This module compares the two distributions and
reports where the golden set has nothing to say.

Pure: the caller supplies vectors (it owns the embedder), this decides what
the numbers mean. Same division as `core/eval/frontier.py`'s calibration —
a core module that embedded text would drag a model dependency into the
layer that must not have one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Below this, a production question is treated as having no counterpart in
# the golden set. Stated in the response rather than hidden, because it is a
# judgement call and a reader who disagrees needs to see what was assumed.
DEFAULT_COVERAGE_THRESHOLD = 0.6


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def nearest_similarities(
    production_vecs: list[list[float]], golden_vecs: list[list[float]],
) -> list[float]:
    """For each production question, how close the closest golden one is.

    Maximum rather than mean: a question is covered when *some* golden
    question resembles it, and averaging over a large golden set would drive
    every value toward zero and report a well-covered corpus as uncovered.

    An empty golden set yields zeros — nothing covers anything — which is the
    honest reading rather than an error, since "no golden set yet" is a real
    state a new Realm passes through.
    """
    if not golden_vecs:
        return [0.0] * len(production_vecs)
    return [max((_cosine(p, g) for g in golden_vecs), default=0.0) for p in production_vecs]


@dataclass(frozen=True)
class CoverageReport:
    """How much of real traffic the golden set speaks to."""

    threshold: float
    n_production: int
    n_golden: int
    uncovered: tuple[str, ...] = ()
    # Deciles of the nearest-similarity distribution. Reported because a
    # single "uncovered share" hides the shape: a set that covers most
    # questions poorly and a set that covers most well but misses a few
    # entirely produce similar shares and call for opposite work.
    deciles: tuple[float, ...] = ()

    @property
    def uncovered_share(self) -> float:
        return len(self.uncovered) / self.n_production if self.n_production else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_production": self.n_production,
            "n_golden": self.n_golden,
            "uncovered_share": round(self.uncovered_share, 4),
            "uncovered": list(self.uncovered),
            "deciles": [round(d, 4) for d in self.deciles],
        }


def coverage_report(
    questions: list[str],
    nearest: list[float],
    n_golden: int,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    max_examples: int = 20,
) -> CoverageReport:
    """Which production questions the golden set does not speak to.

    Uncovered questions are returned worst-first, because the one furthest
    from anything tested is the one most worth turning into a test — and a
    reader who only reads the first few should be reading those.

    Capped at ``max_examples``: a list of every uncovered question is a data
    dump rather than a finding, and the share already says how many there
    are.
    """
    paired = sorted(
        ((q, s) for q, s in zip(questions, nearest, strict=True)),
        key=lambda pair: pair[1],
    )
    uncovered = tuple(q for q, s in paired if s < threshold)[:max_examples]
    ordered = sorted(nearest)
    deciles = tuple(
        ordered[min(int(len(ordered) * i / 10), len(ordered) - 1)] for i in range(11)
    ) if ordered else ()
    return CoverageReport(
        threshold=threshold,
        n_production=len(questions),
        n_golden=n_golden,
        # The share must count *all* uncovered questions, not the capped
        # example list, or a cap would silently improve the number it is
        # reported beside.
        uncovered=uncovered,
        deciles=deciles,
    )


def uncovered_count(nearest: list[float], threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> int:
    """How many production questions fall below the threshold, uncapped."""
    return sum(1 for s in nearest if s < threshold)
