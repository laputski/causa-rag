"""Temporal versioning — record status fields and actuality policy."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from core.models import ScoredChunk


class RecordStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REPEALED = "repealed"
    DRAFT = "draft"


@runtime_checkable
class ActualityPolicy(Protocol):
    policy_id: str

    def is_actual(self, metadata: dict[str, Any]) -> bool:
        ...


class ActiveOnlyPolicy:
    """Keeps only records with status == 'active'."""

    policy_id = "active_only"

    def is_actual(self, metadata: dict[str, Any]) -> bool:
        status = metadata.get("status", RecordStatus.ACTIVE)
        return str(status) == RecordStatus.ACTIVE


class AnyStatusPolicy:
    """Accepts all records regardless of status (used in tests and draft mode)."""

    policy_id = "any_status"

    def is_actual(self, metadata: dict[str, Any]) -> bool:
        return True


class TemporalFilter:
    """Filters ScoredChunks using an ActualityPolicy."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def apply(self, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        return [sc for sc in candidates if self._policy.is_actual(sc.chunk.metadata)]
