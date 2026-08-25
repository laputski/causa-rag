"""The type of a question put to technical documentation.

The classification is not wanted for itself: the type selects the answer mask.
A question about a procedure needs a section reference and a note about which
revision of the manual it came from; a safety question needs its warning before
the source and not after it; a closed question needs neither.

Keywords, and not a model, because a pack has to be readable and editable by
somebody who knows the subject area, not by somebody who can train a
classifier.

The keywords carry both languages. This pack is the worked example the demo
realm can switch on, and that realm's corpus is the English handbook, so a
Russian-only list classified all fifteen of its questions as `open`, the
fallback, and the example demonstrated nothing. A pack aimed at one language
alone is a legitimate thing to write; this one is not, because it ships as the
example.
"""
from __future__ import annotations

from core.models import QueryRequest
from core.routing import RouteDecision

_PROCEDURE = (
    "как ", "порядок", "инструкц", "шаг", "настро", "установ", "замен", "запуст",
    "how do i", "how long", "how many", "how much", "what should i", "what do i",
    "procedure", "step", "install", "configure", "replace", "submit", "request",
)
_SAFETY = (
    "безопасн", "опасн", "предупрежд", "запрещ", "нельзя", "риск", "травм",
    "safety", "danger", "hazard", "warning", "forbidden", "prohibit", "risk", "injur",
)
_CLOSED = (
    "есть ли", "нужно ли", "можно ли", "поддерживает", "входит ли",
    "is it ", "are there", "do i need", "may i", "am i allowed", "is there",
    "does the", "can i ",
)


def classify_question_type(text: str) -> str:
    lower = text.lower()
    # Safety outranks procedure: "how to replace a part while it is live" is a
    # safety question even though it opens like a procedure.
    if any(s in lower for s in _SAFETY):
        return "safety"
    if any(s in lower for s in _CLOSED):
        return "closed"
    if any(s in lower for s in _PROCEDURE):
        return "procedure"
    return "open"


class ManualsRoutePolicy:
    """A core.routing.RoutePolicy for technical documentation."""

    policy_id = "manual_question_type"

    def classify(self, request: QueryRequest) -> RouteDecision:
        return RouteDecision(
            mode=request.mode or "default",
            question_type=request.question_type or classify_question_type(request.text),
            pipeline_id="naive",
            confidence=0.7,
        )
