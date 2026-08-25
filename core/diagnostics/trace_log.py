"""A bounded, in-memory record of what recent queries actually did.

A real user's failure should be able to become a golden question, and that
needs the query and what retrieval returned for it. The platform cannot observe
either, because it deliberately does not sit in the request path.
A served system recording its own traces and letting them be fetched later is
how the platform learns without being in the way.

**Bounded on purpose.** A ring of fixed size, oldest dropped first. An
unbounded log inside a served system is a memory leak with a delayed fuse,
and the platform only ever needs the recent tail: a trace nobody exported
before it aged out was a trace nobody wanted.

**In memory on purpose.** Persisting it would make the template impose a
storage choice on every system grown from it, which is exactly the kind of
inherited dependency this design sheds. A deployment that wants durability
exports on a schedule.

**Privacy.** A trace holds a user's question and fragments of retrieved
documents. It is therefore recorded only when a deployment switches it on,
never by default, and the answer is kept only as a bounded preview rather
than in full.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

_ANSWER_PREVIEW_CHARS = 400
_DEFAULT_CAPACITY = 500


@dataclass(frozen=True)
class QueryTrace:
    """One handled query. Deliberately not the whole `Answer`: what a later
    reader needs is the question, which sources came back, and how the
    pipeline spent its time."""

    trace_id: str
    query: str
    created_at: str
    # (chunk_id, structural_path, score) per returned source, in rank order.
    # Enough to reconstruct which units were retrieved and where they sat,
    # without carrying whole documents out of the served system.
    sources: tuple[tuple[str, str, float], ...] = ()
    stage_trace: dict[str, Any] = field(default_factory=dict)
    answer_preview: str = ""
    top_k: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "created_at": self.created_at,
            "sources": [
                {"chunk_id": cid, "structural_path": path, "score": score}
                for cid, path, score in self.sources
            ],
            "stage_trace": self.stage_trace,
            "answer_preview": self.answer_preview,
            "top_k": self.top_k,
        }


class TraceLog:
    """Append-and-read-back, nothing else.

    No filtering, no search, no aggregation: a served system is not the place
    to analyse traces, only to hold them until the platform collects them.
    Every capability added here would be one more thing a derived system
    carries and one more thing to keep working.
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY, enabled: bool = False) -> None:
        self._entries: deque[QueryTrace] = deque(maxlen=max(1, capacity))
        self.enabled = enabled

    @property
    def capacity(self) -> int:
        return self._entries.maxlen or 0

    def record(self, trace_id: str, query: str, answer: Any, created_at: str, top_k: int = 0) -> None:
        """Records one handled query. A no-op while disabled, and never
        raises: a served system must not fail a user's request because its
        own diagnostics could not be written down."""
        if not self.enabled:
            return
        try:
            refs = getattr(answer, "source_refs", None) or []
            stage_trace = getattr(answer, "stage_trace", None)
            self._entries.append(QueryTrace(
                trace_id=trace_id,
                query=query,
                created_at=created_at,
                sources=tuple(
                    (
                        getattr(r, "chunk_id", "") or "",
                        getattr(r, "structural_path", "") or "",
                        float(getattr(r, "score", 0.0) or 0.0),
                    )
                    for r in refs
                ),
                stage_trace=stage_trace.model_dump() if hasattr(stage_trace, "model_dump") else {},
                answer_preview=(getattr(answer, "text", "") or "")[:_ANSWER_PREVIEW_CHARS],
                top_k=top_k,
            ))
        except Exception:
            return

    def export(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Newest first, because a caller collecting periodically wants what
        it has not seen yet, and that is always at the recent end."""
        entries = list(self._entries)[::-1]
        if limit is not None:
            entries = entries[:limit]
        return [e.to_dict() for e in entries]

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
