"""Instrumentation SDK — wrap an external RAG's plain
retrieve()/generate() functions into objects that satisfy core.interfaces
(Retriever/Generator/Pipeline), so the platform's full diagnostics
(pipeline-trace, eval, regression-guard) apply to a system that was never
written against this platform's interfaces.

The wrapped function's contract is intentionally loose — it does not need to
return ScoredChunk/Answer objects — so that wrapping a real external system
stays a handful of lines (see examples/instrument_external/).
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from core.models import Answer, Chunk, QueryRequest, ScoredChunk, SourceRef, StageTrace

# What a wrapped retriever function may return per result: a ready-made
# ScoredChunk, a (text, score) pair, or a dict with at least a "text" key.
RawRetrieveResult = ScoredChunk | tuple[str, float] | dict[str, Any]
RetrieveFn = Callable[..., list[RawRetrieveResult]]
GenerateFn = Callable[[str], str]


def _normalize_chunk(raw: RawRetrieveResult, fallback_doc_id: str) -> ScoredChunk:
    if isinstance(raw, ScoredChunk):
        return raw
    if isinstance(raw, tuple):
        text, score = raw
        chunk = Chunk(doc_id=fallback_doc_id, text=text)
        return ScoredChunk(chunk=chunk, score=score)
    if isinstance(raw, dict):
        chunk = Chunk(
            doc_id=raw.get("doc_id", fallback_doc_id),
            text=raw.get("text", ""),
            structural_path=raw.get("structural_path", ""),
        )
        return ScoredChunk(chunk=chunk, score=float(raw.get("score", 0.0)))
    raise TypeError(f"Unsupported retrieve() result item: {raw!r}")


def wrap_retriever(fn: RetrieveFn, retriever_id: str = "external_retriever") -> Any:
    """Wrap a plain ``fn(query, k, filters=None) -> list[...]`` into a Retriever.

    ``fn`` may return ``ScoredChunk`` objects, ``(text, score)`` tuples, or
    dicts with ``text``/``score``/``doc_id``/``structural_path`` keys — pick
    whichever shape requires the least change to the wrapped system.
    """

    class _Wrapped:
        def __init__(self) -> None:
            self.retriever_id = retriever_id

        def retrieve(self, query: str, k: int = 5, filters: dict[str, Any] | None = None, **_: Any) -> list[ScoredChunk]:
            raw_results = fn(query, k, filters) if _accepts_filters(fn) else fn(query, k)
            fallback_doc_id = str(uuid.uuid4())
            return [_normalize_chunk(r, fallback_doc_id) for r in raw_results]

    return _Wrapped()


def _accepts_filters(fn: Callable[..., Any]) -> bool:
    try:
        import inspect
        params = inspect.signature(fn).parameters
        return len(params) >= 3 or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    except (TypeError, ValueError):
        return False


def wrap_generator(fn: GenerateFn, generator_id: str = "external_generator") -> Any:
    """Wrap a plain ``fn(prompt) -> str`` into a Generator."""

    class _Wrapped:
        def __init__(self) -> None:
            self.generator_id = generator_id

        def generate(self, prompt: str, **_: Any) -> str:
            return fn(prompt)

    return _Wrapped()


def build_pipeline(
    retriever: Any,
    generator: Any,
    reranker: Any | None = None,
    top_k: int = 5,
    pipeline_id: str = "external",
) -> Any:
    """Assemble a Pipeline from an (already-wrapped) retriever + generator.

    Unlike core.pipeline.NaivePipeline, this does not require an Embedder —
    the wrapped retriever is expected to do its own embedding internally,
    exactly as the external system already does. Still produces a real
    StageTrace, so the resulting pipeline-trace looks the same as an
    in-process one (diagnostics depth, not provenance, is what
    distinguishes white-box from black-box).
    """

    class _ExternalPipeline:
        def __init__(self) -> None:
            self.pipeline_id = pipeline_id
            self._retriever = retriever
            self._generator = generator
            self._reranker = reranker

        def run(self, request: QueryRequest) -> Answer:
            from core.pipeline import _build_prompt  # reuse the shared prompt template

            trace = StageTrace()
            t_total = time.perf_counter()

            t0 = time.perf_counter()
            k = request.top_k or top_k
            scored = self._retriever.retrieve(request.text, k, request.filters or None)
            trace.dense_retrieve_ms = round((time.perf_counter() - t0) * 1000, 1)
            trace.n_dense = len(scored)
            trace.n_merged = len(scored)

            if self._reranker is not None:
                t0 = time.perf_counter()
                scored = self._reranker.rerank(request.text, scored)[:k]
                trace.rerank_ms = round((time.perf_counter() - t0) * 1000, 1)
                trace.n_reranked = len(scored)

            trace.n_deduped = len(scored)
            context_chunks = [sc.chunk.text for sc in scored]
            trace.context_chars = sum(len(c) for c in context_chunks)

            prompt, prompt_id, prompt_version = _build_prompt(request.text, context_chunks)
            t0 = time.perf_counter()
            raw_text = self._generator.generate(prompt)
            trace.generate_ms = round((time.perf_counter() - t0) * 1000, 1)

            trace.total_ms = round((time.perf_counter() - t_total) * 1000, 1)

            source_refs = [
                SourceRef(
                    doc_id=sc.chunk.doc_id,
                    chunk_id=sc.chunk.chunk_id,
                    structural_path=sc.chunk.structural_path,
                    score=sc.score,
                    chunk_text=sc.chunk.text,
                )
                for sc in scored
            ]

            return Answer(
                text=raw_text,
                source_refs=source_refs,
                stage_trace=trace,
                rendered_prompt_preview=prompt[:500],
                metadata={
                    "pipeline": self.pipeline_id,
                    "trace_id": request.trace_id,
                    "prompt_id": prompt_id,
                    "prompt_version": prompt_version,
                },
            )

    return _ExternalPipeline()
