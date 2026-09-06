"""What each metric computes, declared beside its name.

Two entries of the failure catalogue were uncatchable for want of this, and
both for the same reason: a number reaches the reader with nothing but a
name attached, so nothing can ask whether the name is right or whether the
number survived the trip.

A metric whose definition drifts from its name is the first. `precision`
that counts recall reads perfectly well on a screen and is wrong in a way no
amount of staring finds, because the name is the only thing on the screen
and the name still says precision. A declaration cannot decide that the name
is right, and it does make the definition visible next to it, which is the
difference between a reader who can check and one who cannot.

A number that is not the number the questions carry is the second. The
aggregate is a mean of the per-question values and nothing ever compared the
two, so a question dropped between writing the run and reading it moves the
number on screen and moves nothing else. Measured across the forty-one runs
stored at the time this was written, every aggregate equalled the mean of
its per-question values to within a millionth, so the check is silent on
healthy data by observation and not by hope.

The population matters as much as the operation. `correct_refusal` is
written for every scored question and the retrieval metrics only for
answerable ones, so "the mean over the questions that carry it" is the only
statement true of all of them, and it is the one declared here.

Pure data and one function over it. No I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: How an aggregate is built out of the per-question values. Only one
#: operation exists today, and the field exists so that the day a percentile
#: or a count arrives, it is declared, and not discovered by a detector
#: reporting every run as broken.
Aggregation = Literal["mean"]


@dataclass(frozen=True)
class MetricDefinition:
    """One metric, in the words a reader would need to check its name."""

    name: str
    #: The questions this is written for. Stated because a mean over a
    #: subset and a mean over everything are different numbers with the
    #: same name.
    over: str
    #: How the run-level number is built from the per-question ones.
    aggregated_by: Aggregation
    #: What the number means, in one sentence, without naming a function.
    says: str


DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="correct_refusal",
        over="every question that was scored at all",
        aggregated_by="mean",
        says="the share of questions answered when the corpus covers them and refused "
             "when it does not",
    ),
    MetricDefinition(
        name="retrieval_recall_at_k",
        over="answerable questions",
        aggregated_by="mean",
        says="the share of the sources a question needs that the final context holds",
    ),
    MetricDefinition(
        name="retrieval_precision_at_k",
        over="answerable questions",
        aggregated_by="mean",
        says="the share of the final context that a question actually needed",
    ),
    MetricDefinition(
        name="retrieval_average_precision",
        over="answerable questions",
        aggregated_by="mean",
        says="the same share, counted so that a needed source ranked first is worth "
             "more than the same source ranked last",
    ),
    MetricDefinition(
        name="pre_rerank_recall_at_k",
        over="answerable questions where the candidate list before reranking was reported",
        aggregated_by="mean",
        says="the share of the sources a question needs that retrieval found before "
             "the reranker chose among them",
    ),
    MetricDefinition(
        name="answer_similarity",
        over="answerable questions",
        aggregated_by="mean",
        says="how close the answer is to the reference answer, in meaning",
    ),
    MetricDefinition(
        name="answer_relevance",
        over="answerable questions",
        aggregated_by="mean",
        says="how close the answer is to the question, in meaning",
    ),
    MetricDefinition(
        name="context_support",
        over="answerable questions",
        aggregated_by="mean",
        says="how much of the answer is carried by the sources returned with it",
    ),
    MetricDefinition(
        name="grounded_in_correct_source",
        over="answerable questions where a retrieved source matches the ground truth",
        aggregated_by="mean",
        says="how much of the answer is carried by the source the question names, "
             "and not merely by some source",
    ),
    MetricDefinition(
        name="citation_number_coverage",
        over="answerable questions whose matching sources carry a structural number",
        aggregated_by="mean",
        says="the share of those numbers that occur in the answer text",
    ),
)

_BY_NAME: dict[str, MetricDefinition] = {d.name: d for d in DEFINITIONS}


def definition_of(name: str) -> MetricDefinition | None:
    """The declaration for a metric, or None when it has none.

    None is the answer a detector acts on, so it is returned and not
    raised: an undeclared metric is a finding about the platform and not an
    error in the code asking about it.
    """
    return _BY_NAME.get(name)
