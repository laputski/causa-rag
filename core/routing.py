from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.models import QueryRequest


@dataclass
class RouteDecision:
    mode: str
    question_type: str
    pipeline_id: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RoutePolicy(Protocol):
    policy_id: str

    def classify(self, request: QueryRequest) -> RouteDecision:
        ...


class NaiveRoutePolicy:
    """Passthrough policy — uses mode/question_type already set on the request."""

    policy_id = "naive"

    def classify(self, request: QueryRequest) -> RouteDecision:
        return RouteDecision(
            mode=request.mode or "default",
            question_type=request.question_type or "open",
            pipeline_id="naive",
        )


def select_pipeline(decision: RouteDecision, registry: Any) -> Any:
    """Resolve a Pipeline from registry using pipeline_id from RouteDecision."""
    return registry.resolve("pipeline", decision.pipeline_id)
