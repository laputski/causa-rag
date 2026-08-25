"""Deriving a similarity threshold instead of asserting one.

The retired pin mechanism used 0.93, a number nobody measured. It came from
one model's intuition about cosine similarity and was never checked against a
real corpus, so it could have been firing on unrelated questions or on none
at all, and no run would have shown either.

A threshold is not portable anyway. Cosine similarity has a different
distribution under every embedding model, so a number that is strict for one
is permissive for another. A bundle therefore ships the *procedure*: examples
that must match and examples that must not, and the recipient derives the
number with its own model.

Pure: takes similarities that a caller computed, returns the threshold they
imply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ThresholdCalibration:
    """A derived threshold, with the evidence that produced it."""

    threshold: float | None
    # The two distributions the threshold sits between. Reported so a reader
    # can see whether it sits in a comfortable gap or in a crowded overlap,
    # which one number cannot say.
    lowest_positive: float | None = None
    highest_negative: float | None = None
    separable: bool = False
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "lowest_positive": self.lowest_positive,
            "highest_negative": self.highest_negative,
            "separable": self.separable,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


def calibrate_threshold(
    positives: list[float], negatives: list[float], margin: float = 0.01,
) -> ThresholdCalibration:
    """The threshold implied by examples that must and must not match.

    ``positives`` are similarities between a question and its own
    paraphrases; ``negatives`` between it and questions it has nothing to do
    with.

    When the two do not overlap, the threshold sits just below the weakest
    paraphrase — the loosest value that still admits every intended match.
    Placing it mid-gap would be arbitrary in the other direction and would
    reject paraphrases the calibration set says are legitimate.

    **When they overlap, no threshold is returned.** An overlap means some
    unrelated question scores higher than some genuine paraphrase, so *any*
    threshold either fires where it must not or fails to fire where it must.
    Returning the best compromise would hide that, and hiding it is how a
    mechanism ends up firing on questions nobody intended — the exact failure
    the retired 0.93 could have had all along without anyone noticing.
    """
    if not positives or not negatives:
        return ThresholdCalibration(
            threshold=None, n_positive=len(positives), n_negative=len(negatives),
        )
    lowest_positive = min(positives)
    highest_negative = max(negatives)
    separable = lowest_positive > highest_negative
    return ThresholdCalibration(
        threshold=round(lowest_positive - margin, 4) if separable else None,
        lowest_positive=round(lowest_positive, 4),
        highest_negative=round(highest_negative, 4),
        separable=separable,
        n_positive=len(positives),
        n_negative=len(negatives),
    )


def calibration_material(entries: list[Any], max_pairs: int = 200) -> dict[str, Any]:
    """The examples a bundle ships so a recipient can calibrate.

    Negatives are *pairs* of distinct entries rather than a flat list of
    questions. Found live: a flat list put a one-entry bundle's own question
    into its "must not match" set, telling a recipient that a question must
    not match itself — which no threshold can satisfy, so calibration would
    have concluded the mechanism was unusable on a bundle that was fine.

    Drawn from the bundle's own questions rather than invented, because two
    corrections in one bundle are by construction about different things, so
    each is a genuine "must not match" for the other. They are already in the
    document, so the material costs nothing to carry and cannot drift from
    what the bundle contains.

    Paraphrases have no such free source. `must_match` stays empty until a
    reviewer supplies them, which is honest: without positives a recipient
    can establish an upper bound for the threshold and nothing more, and
    `calibrate_threshold` says so by returning None.

    A bundle with fewer than two entries carries no negatives at all. One
    question cannot be a negative for itself, and inventing an unrelated one
    would put the publisher's guess about the recipient's domain into an
    artefact that is supposed to carry only observations.
    """
    questions = [getattr(e, "question", "") for e in entries if getattr(e, "question", "")]
    pairs = [
        {"a": a, "b": b}
        for i, a in enumerate(questions)
        for b in questions[i + 1:]
    ][:max_pairs]
    return {
        "must_match": [],
        "must_not_match_pairs": pairs,
        "procedure": (
            "Compute the vectors with your own embedding model. Take the threshold "
            "slightly below the lowest similarity among the required matches and above "
            "the highest among the required non-matches. If those two sets overlap, no "
            "threshold exists and the mechanism cannot be used. With no required "
            "matches, only the threshold's upper bound is derived."
        ),
    }
