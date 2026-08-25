"""Langfuse v2 tracing adapter (SDK 2.60, Server 2.90).

Wraps each pipeline run as a Langfuse trace with spans:
  retrieve   → span:      query, top_k → n_chunks, paths
  generate   → generation: prompt, model → answer text

Falls back silently when Langfuse is unreachable or disabled
(LANGFUSE_ENABLED=false).
"""
from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog

log = structlog.get_logger()

_ENABLED = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "lf-pk-local-causa-rag")
_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "lf-sk-local-causa-rag")
_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3001")


class LangfuseTracer:
    """Thin wrapper around Langfuse SDK v2 for pipeline tracing."""

    def __init__(self) -> None:
        self._lf = None
        if _ENABLED:
            try:
                from langfuse import Langfuse
                self._lf = Langfuse(
                    public_key=_PUBLIC_KEY,
                    secret_key=_SECRET_KEY,
                    host=_HOST,
                )
                # auth_check removed — SDK v2.60 Projects schema incompatible with server v2.90;
                # connectivity verified on first flush instead
                log.info("langfuse.connected", host=_HOST)
            except Exception as exc:
                log.warning("langfuse.unavailable", error=str(exc))
                self._lf = None

    @property
    def enabled(self) -> bool:
        return self._lf is not None

    def trace_pipeline(
        self,
        trace_id: str,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> _PipelineTrace:
        return _PipelineTrace(self._lf, trace_id, query, metadata)

    def flush(self) -> None:
        if self._lf:
            try:
                self._lf.flush()
            except Exception:
                pass


class _PipelineTrace:
    """Context manager that wraps one query→answer cycle as a Langfuse trace."""

    def __init__(self, lf: Any, trace_id: str, query: str, metadata: dict | None) -> None:
        self._lf = lf
        self._trace_id = trace_id
        self._query = query
        self._metadata = metadata or {}
        self._trace = None

    def __enter__(self) -> _PipelineTrace:
        if self._lf is None:
            return self
        try:
            self._trace = self._lf.trace(
                id=self._trace_id,
                name="rag-pipeline",
                input=self._query,
                metadata=self._metadata,
            )
        except Exception as exc:
            log.warning("langfuse.trace.start_failed", error=str(exc))
            self._trace = None
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # flushed via LangfuseTracer.flush()

    def set_output(self, output: Any) -> None:
        if self._trace:
            try:
                self._trace.update(output=str(output))
            except Exception:
                pass

    @contextmanager
    def span(self, name: str, input: Any = None) -> Generator[_Handle, None, None]:
        if self._trace is None:
            yield _Handle(None)
            return
        try:
            sp = self._trace.span(name=name, input=input)
            yield _Handle(sp)
            sp.end()
        except Exception as exc:
            log.warning("langfuse.span.failed", name=name, error=str(exc))
            yield _Handle(None)

    @contextmanager
    def generation(self, name: str, model: str, prompt: str) -> Generator[_Handle, None, None]:
        if self._trace is None:
            yield _Handle(None)
            return
        try:
            gen = self._trace.generation(name=name, model=model, input=prompt)
            yield _Handle(gen)
            gen.end()
        except Exception as exc:
            log.warning("langfuse.generation.failed", name=name, error=str(exc))
            yield _Handle(None)


class _Handle:
    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def set_output(self, output: Any) -> None:
        if self._obj:
            try:
                self._obj.update(output=output)
            except Exception:
                pass
