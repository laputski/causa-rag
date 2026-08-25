from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """Central registry for all pipeline components.

    Components register themselves by kind + id at startup.
    Core never imports concrete adapters — it only resolves through this registry.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def register(self, kind: str, component_id: str, instance: Any) -> None:
        self._store.setdefault(kind, {})[component_id] = instance

    def resolve(self, kind: str, component_id: str) -> Any:
        try:
            return self._store[kind][component_id]
        except KeyError:
            raise KeyError(f"Component not found: kind={kind!r} id={component_id!r}") from None

    def list_kind(self, kind: str) -> list[str]:
        return list(self._store.get(kind, {}).keys())

    def list_all(self) -> dict[str, list[str]]:
        return {kind: list(ids.keys()) for kind, ids in self._store.items()}


registry = ComponentRegistry()
