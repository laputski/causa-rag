"""Whether an answer refuses is decided in one place.

The rule lived in three. `core/eval/detectors.py` holds the canonical pattern;
this router kept a weaker copy and now imports the original; the interface kept
a third, of eleven fixed substrings against the pattern's fifteen alternatives.
So the same answer counted as a refusal in one panel of a screen and not in the
one beside it.

The comment beside the canonical pattern records that this exact drift had
already been found and removed once, between two copies on the server side. It
survived between the server and the interface.
"""
from __future__ import annotations

import pytest

from services.api_gateway.routers.experiments import _attach_refusal_verdict

# Wordings the canonical pattern matches and the interface's own list did not.
# Each one is an answer that refuses and used to be counted as an answer.
UNSEEN_BY_THE_INTERFACE = [
    "В предоставленных документах это не указано.",
    "Нет информации по этому вопросу.",
    "Документ не содержит такого положения.",
    "Сведения отсутствуют.",
    "I could not find this in the provided context.",
    "The handbook does not mention this.",
    "There is no relevant information here.",
]


@pytest.mark.parametrize("answer", UNSEEN_BY_THE_INTERFACE)
def test_a_refusal_the_interface_used_to_miss_is_now_named_by_the_server(answer: str) -> None:
    rows = [{"generated_answer": answer}]
    _attach_refusal_verdict(rows)
    assert rows[0]["is_refusal"] is True, (
        f"{answer!r} refuses and is not reported as refusing, so the count on the "
        "run page is still lower than the truth"
    )


def test_an_answer_that_answers_is_not_called_a_refusal() -> None:
    """The half a rule that fires on everything would fail."""
    rows = [
        {"generated_answer": "Section 12 requires the finance director to approve it."},
        {"generated_answer": "The daily allowance is 85 EUR outside Europe."},
    ]
    _attach_refusal_verdict(rows)
    assert [r["is_refusal"] for r in rows] == [False, False]


def test_an_empty_answer_counts_as_a_refusal() -> None:
    rows = [{"generated_answer": ""}, {"generated_answer": "   "}, {}]
    _attach_refusal_verdict(rows)
    assert all(r["is_refusal"] for r in rows)


def test_the_verdict_uses_the_same_pattern_the_detectors_use() -> None:
    """Not a second implementation that happens to agree today."""
    from core.eval.detectors import NOT_FOUND_RE
    from services.api_gateway.routers import experiments

    assert experiments._NOT_FOUND is NOT_FOUND_RE
