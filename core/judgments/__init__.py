"""Relevance judgments — a reviewer's statement about what is correct.

A judgment records an
observation ("for question Q, chunk C is relevant, chunk D is not"), never an
instruction ("inject C, drop D"). That distinction is the whole point of this
package: the same content used as a runtime rule generalises to nothing, while
used as a labelled example it becomes a test case, a preference pair for
ranking work, and evidence when looking for a common cause.
"""
from core.judgments.model import (
    JudgedChunk,
    RelevanceJudgment,
    judgment_from_dict,
    judgment_to_dict,
    preference_pairs,
    to_golden_question,
)

__all__ = [
    "JudgedChunk",
    "RelevanceJudgment",
    "judgment_from_dict",
    "judgment_to_dict",
    "preference_pairs",
    "to_golden_question",
]
