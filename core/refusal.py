"""RefusalPolicy — honest refusal when grounding fails."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from core.models import Answer, GroundingResult


@runtime_checkable
class RefusalPolicy(Protocol):
    policy_id: str

    def should_refuse(self, grounding: GroundingResult) -> bool:
        ...

    def build_refusal(self, reason: str, original_query: str) -> Answer:
        ...


class ThresholdRefusalPolicy:
    """Refuses when grounding confidence is below ``min_confidence``."""

    policy_id = "threshold_refusal"

    def __init__(
        self,
        min_confidence: float = 0.5,
        refusal_text_fn: Callable[[str, str], str] | None = None,
    ) -> None:
        self._min_confidence = min_confidence
        self._refusal_text_fn = refusal_text_fn or self._default_text

    def should_refuse(self, grounding: GroundingResult) -> bool:
        return not grounding.is_grounded or grounding.confidence < self._min_confidence

    def build_refusal(self, reason: str, original_query: str) -> Answer:
        text = self._refusal_text_fn(reason, original_query)
        return Answer(
            text=text,
            refused=True,
            refusal_reason=reason,
        )

    @staticmethod
    def _default_text(reason: str, query: str) -> str:
        # English, because this is `core/`: the platform's own last-resort
        # wording, used when no domain pack supplied a refusal policy. A pack
        # that wants its subject area's language provides `refusal_text_fn`,
        # which is what `domain_packs/manuals/refusal.py` does.
        return (
            f"The answer could not be supported by the available sources "
            f"(reason: {reason}). Please refine the question."
        )


def apply_grounding_guard(
    answer_text: str,
    grounding: GroundingResult,
    policy: Any,
    original_query: str,
) -> Answer:
    """Return a refusal Answer if policy triggers, otherwise a grounded Answer."""
    if policy.should_refuse(grounding):
        return policy.build_refusal("low_confidence", original_query)
    return Answer(text=answer_text, grounding=grounding)
