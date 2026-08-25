"""A technical manual's refusal.

A refusal names what was missing instead of apologising: somebody reading a
manual came for an action, and "try rephrasing" is no use to them.

The strings below are Russian, and stay Russian, for the reason the pack's
`__init__` gives: refusal wording is answer text, not interface text, and a
pack carries one subject area's vocabulary in one language.
"""
from __future__ import annotations

from core.models import Answer, GroundingResult

REFUSAL_TEMPLATES: dict[str, str] = {
    "no_grounding": (
        "В загруженной документации нет раздела, на который можно опереться "
        "для этого ответа. Проверьте, входит ли нужное руководство в корпус."
    ),
    "out_of_scope": (
        "Вопрос выходит за пределы загруженной документации. "
        "Если руководство должно быть в корпусе — загрузите его и повторите."
    ),
    "empty_input": "Запрос пуст: введите вопрос.",
}


class ManualsRefusalPolicy:
    """A core.refusal.RefusalPolicy for technical documentation."""

    policy_id = "manuals"

    def __init__(self, min_confidence: float = 0.5) -> None:
        self._min_confidence = min_confidence

    def should_refuse(self, grounding: GroundingResult) -> bool:
        return not grounding.is_grounded or grounding.confidence < self._min_confidence

    def build_refusal(self, reason: str = "no_grounding") -> Answer:
        return Answer(
            text=REFUSAL_TEMPLATES.get(reason, REFUSAL_TEMPLATES["out_of_scope"]),
            source_refs=[],
        )
