"""RAG pipelines.

``NaivePipeline`` is the plain path: retrieve → generate → format. It also
accepts optional reranker / grounder / route_policy components and runs them
when provided, driven by ``ExperimentConfig``. When none are given, behaviour is
identical to the pipeline without them.
``ConfigurablePipeline`` is a thin alias used when components come from config.
trace_id flows through every stage via structlog context.
"""
from __future__ import annotations

import time
from typing import Any

import structlog

from core.interfaces import Embedder, Generator, Retriever
from core.models import Answer, QueryRequest, ScoredChunk, SourceRef, StageTrace

log = structlog.get_logger()

try:
    from adapters.langfuse_tracer import LangfuseTracer
    _tracer = LangfuseTracer()
except Exception:
    _tracer = None  # type: ignore[assignment]


def _to_source_refs(scored: list[ScoredChunk]) -> list[SourceRef]:
    """Shared by the final (post-rerank) and pre-rerank snapshots — same
    shape, same metadata extraction, so the two are directly comparable
    (core/eval/funnel.py, Phase 1)."""
    return [
        SourceRef(
            doc_id=sc.chunk.doc_id,
            chunk_id=sc.chunk.chunk_id,
            structural_path=sc.chunk.structural_path,
            score=sc.score,
            chunk_text=sc.chunk.text,
            dense_score=sc.chunk.metadata.get("dense_score", 0.0),
            sparse_score=sc.chunk.metadata.get("sparse_score", 0.0),
            rrf_rank=sc.chunk.metadata.get("rrf_rank", 0),
            source_code=sc.chunk.metadata.get("source_code"),
            article_no=sc.chunk.metadata.get("article_no"),
            pinned=sc.chunk.metadata.get("pinned", False),
        )
        for sc in scored
    ]


def _build_prompt(
    query: str, context_chunks: list[str], realm_id: str | None = None,
) -> tuple[str, str, int]:
    """Returns (prompt, prompt_id, prompt_version)."""
    try:
        from core.prompt_store import prompt_store
        tpl = prompt_store.get_active(realm_id)
        if tpl:
            return tpl.render(query, context_chunks), tpl.id, tpl.version
    except Exception:
        pass
    # The fallback hardcoded prompt, with the same "Fragment N" labelling as
    # PromptTemplate.render (core/prompt_store.py), not "[N]", to avoid the
    # a bracketed index being confused with an article citation number.
    context = "\n\n".join(f"Fragment {i + 1}:\n{c}" for i, c in enumerate(context_chunks))
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context provided above."
    )
    return prompt, "fallback", 0


class NaivePipeline:
    """Simplest working RAG pipeline.

    Cache boundary tracking:
      - embed_cache_hits / embed_cache_misses — from embedder.cache_stats()
      - retrieve_calls — incremented every query (retrieval is never cached)
    These counters are included in Answer.metadata so ExperimentRunner can log them.
    """

    pipeline_id = "naive"

    def __init__(
        self,
        retriever: Retriever,
        embedder: Embedder,
        generator: Generator,
        top_k: int = 5,
        # The candidate window, separate from the final
        # context. None keeps the old behaviour exactly: fetch as many as go
        # into the answer, which is what made a wide pre-cut snapshot
        # impossible and with it every counterfactual about context size.
        fetch_k: int | None = None,
        pipeline_id: str | None = None,
        reranker: Any | None = None,
        grounder: Any | None = None,
        route_policy: Any | None = None,
        scorer: Any | None = None,
        mask_engine: Any | None = None,
        refusal_policy: Any | None = None,
        realm_id: str | None = None,
        retrieval_pins: list[Any] | None = None,
    ) -> None:
        if pipeline_id is not None:
            self.pipeline_id = pipeline_id
        self._retriever = retriever
        self._embedder = embedder
        self._generator = generator
        self._top_k = top_k
        self._fetch_k = fetch_k
        # Optional, config-driven steps. None ⇒ step disabled.
        self._reranker = reranker
        self._grounder = grounder
        self._route_policy = route_policy
        # domain-pack-provided steps. Same None-disables pattern;
        # zero domain knowledge here, only generic hook points.
        self._scorer = scorer
        self._mask_engine = mask_engine
        self._refusal_policy = refusal_policy
        # retrieval_pins overlay (core/pins/overlay.py). A
        # plain list of core.pins.overlay.Pin, not a single component
        # (unlike reranker/grounder/etc. above) — every active pin for this
        # pipeline's (realm_id, corpus_id) is fetched once by the caller
        # (services/api_gateway/routers/experiments.py) and passed in here;
        # this pipeline never queries MongoDB itself. Empty/None ⇒ the
        # overlay step is a no-op, byte-identical to before this existed.
        self._retrieval_pins = retrieval_pins or []
        self._retrieve_calls: int = 0
        # Bound once at pipeline-construction time (mirrors how the
        # retriever/generator are already rebuilt per-Realm at that same
        # point — see core/experiment/runner.py#_build_pipeline and
        # services/api_gateway/main.py#_build_chat_pipeline) rather than
        # threaded per-request: which prompt is active is now Realm-scoped
        # (core/prompt_store.py), so this pipeline instance needs to know
        # which Realm it's answering for. None = no Realm context (falls
        # back to the old global-active resolution, see
        # PromptStore.get_active).
        self._realm_id = realm_id

    def run(self, request: QueryRequest, *, _generate: bool = True) -> Answer:
        # _generate=False is retrieve() below, not a public knob. It is a
        # flag rather than a second method because the worth of a
        # retrieval-only measurement rests entirely on it measuring the
        # retrieval this pipeline actually performs; a parallel copy of
        # the search half would drift from this one, and the report would
        # then describe code no query ever takes.
        bound = log.bind(trace_id=request.trace_id, pipeline=self.pipeline_id)
        k = request.top_k or self._top_k
        # Never below k: a candidate window narrower than the context it
        # feeds would silently shrink the answer instead of widening the
        # search, which is the opposite of what asking for one means.
        fetch = max(self._fetch_k or k, k)
        self._retrieve_calls += 1

        lf_trace = (
            _tracer.trace_pipeline(
                trace_id=request.trace_id or "",
                query=request.text,
                metadata={"pipeline": self.pipeline_id, "top_k": k},
            )
            if _tracer and _tracer.enabled
            else _NoopTrace()
        )

        t_total_start = time.perf_counter()
        trace = StageTrace()

        # Optional routing (config-driven). Classifies the request;
        # observable in answer metadata. Routing does not yet affect behaviour.
        route_meta: dict[str, Any] = {}
        if self._route_policy is not None:
            decision = self._route_policy.classify(request)
            route_meta = {
                "route_mode": decision.mode,
                "route_question_type": decision.question_type,
                "route_pipeline_id": decision.pipeline_id,
                "route_confidence": decision.confidence,
            }
            bound.info("pipeline.route", **route_meta)

        with lf_trace:
            bound.info("pipeline.retrieve.start", query=request.text[:100])
            with lf_trace.span("retrieve", input={"query": request.text, "top_k": k}) as sp:
                t0 = time.perf_counter()
                query_vec = self._embedder.embed([request.text])[0]
                trace.embed_ms = round((time.perf_counter() - t0) * 1000, 1)

                t0 = time.perf_counter()
                scored = self._retriever.retrieve(
                    query=request.text,
                    k=fetch,
                    filters=request.filters or None,
                    query_vector=query_vec,  # type: ignore[call-arg]
                )
                retrieve_ms = round((time.perf_counter() - t0) * 1000, 1)
                # detect hybrid vs dense — hybrid sets retriever_id on sub-results
                has_dense = any(getattr(sc, "retriever_id", "") == "qdrant_dense" for sc in scored)
                has_sparse = any(getattr(sc, "retriever_id", "") in ("opensearch", "sparse") for sc in scored)
                if has_dense and has_sparse:
                    trace.dense_retrieve_ms = retrieve_ms / 2
                    trace.sparse_retrieve_ms = retrieve_ms / 2
                    trace.merge_ms = retrieve_ms / 4
                else:
                    trace.dense_retrieve_ms = retrieve_ms
                trace.n_merged = len(scored)
                sp.set_output({"n_chunks": len(scored), "paths": [s.chunk.structural_path for s in scored]})
            bound.info("pipeline.retrieve.done", n_results=len(scored))

            seen_texts: set[str] = set()
            deduped = []
            for sc in scored:
                if sc.chunk.text not in seen_texts:
                    seen_texts.add(sc.chunk.text)
                    deduped.append(sc)
            scored = deduped
            trace.n_deduped = len(scored)

            # Optional domain-pack scorer (config-driven). Re-scores
            # the same ScoredChunk shape the reranker also consumes.
            if self._scorer is not None:
                scored = self._scorer.score(request.text, scored)

            # Graph retrieval is detected by retriever id; record graph stats so the
            # trace shows graph participation.
            if getattr(self._retriever, "retriever_id", "") == "graph_hybrid":
                trace.graph_ms = trace.dense_retrieve_ms
                trace.n_graph = len(scored)

            # Funnel diagnosis, Phase 1 — snapshot the ranked list BEFORE the
            # reranker reorders/cuts it, so recall@k can be computed both
            # before and after reranking (core/eval/funnel.py). Without this,
            # "the reranker dropped the right chunk" and "retrieval never
            # found it" look identical from stored data alone.
            pre_rerank_source_refs = _to_source_refs(scored) if self._reranker is not None else []

            # The candidate window before the final cut, kept
            # separate from `pre_rerank_source_refs` on purpose. That field
            # means "what the reranker was given" and core/eval/funnel.py
            # blames the reranker whenever it holds the answer and the final
            # list does not. Reusing it for a window widened by fetch_k, with
            # no reranker in the pipeline at all, would make funnel report a
            # rerank failure where nothing reranked anything.
            #
            # Recorded only when the window is genuinely wider than the
            # context, since an identical copy of the final list answers no
            # counterfactual and doubles what every run stores.
            candidate_source_refs = _to_source_refs(scored) if fetch > k else []

            # Optional reranking (config-driven). Observable in trace.
            if self._reranker is not None:
                bound.info("pipeline.rerank.start", reranker=getattr(self._reranker, "reranker_id", "?"))
                t0 = time.perf_counter()
                scored = self._reranker.rerank(request.text, scored)[:k]
                trace.rerank_ms = round((time.perf_counter() - t0) * 1000, 1)
                trace.n_reranked = len(scored)
                bound.info("pipeline.rerank.done", n=len(scored), ms=trace.rerank_ms)

            # retrieval_pins overlay (core/pins/overlay.py).
            # Injection point: AFTER any rerank step, not between merge and
            # rerank. The original reason was that rerank-then-cut left
            # exactly `k` candidates, so a pin injected earlier would very
            # likely be discarded by that same cut.
            #
            # A later change separated fetch_k from top_k, so the reason no
            # longer holds and an earlier injection is possible in principle.
            # It is deliberately not moved: this overlay is off by default now
            # and reduced to an opt-in what-if on the stand, so
            # changing where it applies would be work on a mechanism nothing
            # currently runs. See the design notes
            if self._retrieval_pins:
                from core.pins.overlay import apply_overlay, select_matching_pins
                matching_pins = select_matching_pins(query_vec, self._retrieval_pins)
                if matching_pins:
                    rewrite_results: dict[str, list[ScoredChunk]] = {}
                    for pin in matching_pins:
                        if not pin.rewrite:
                            continue
                        try:
                            rewrite_vec = self._embedder.embed([pin.rewrite])[0]
                            rewrite_results[pin.id] = self._retriever.retrieve(
                                query=pin.rewrite, k=k, filters=request.filters or None,
                                query_vector=rewrite_vec,
                            )
                        except Exception:
                            bound.warning("pipeline.pin_overlay.rewrite_failed", pin_id=pin.id)
                    scored = apply_overlay(scored, matching_pins, rewrite_results)
                    bound.info(
                        "pipeline.pin_overlay.applied",
                        n_pins=len(matching_pins), n_chunks=len(scored),
                        pin_ids=[p.id for p in matching_pins],
                    )

            # The final cut. A reranker already truncated to k
            # above; without one, `scored` is the whole fetched window and
            # must be cut here, or widening the search would silently widen
            # the answer's context too.
            scored = scored[:k]

            context_chunks = [sc.chunk.text for sc in scored]
            source_refs = _to_source_refs(scored)
            trace.context_chars = sum(len(c) for c in context_chunks)

            if not _generate:
                # Everything the search produced is already in hand.
                # Asking the model now would spend a generation nothing
                # reads, and would let a generator failure fail a run that
                # never depended on a generator.
                trace.total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)
                lf_trace.set_output("")
                embed_stats: dict[str, object] = {}
                if hasattr(self._embedder, "cache_stats"):
                    embed_stats = self._embedder.cache_stats()  # type: ignore[assignment]
                return Answer(
                    text="",
                    source_refs=source_refs,
                    pre_rerank_source_refs=pre_rerank_source_refs,
                    candidate_source_refs=candidate_source_refs,
                    stage_trace=trace,
                    metadata={
                        "pipeline": self.pipeline_id,
                        "trace_id": request.trace_id,
                        "retrieve_calls": self._retrieve_calls,
                        # Named in the answer so a stored run cannot be
                        # mistaken later for a generation run whose model
                        # happened to return nothing.
                        "retrieval_only": True,
                        **{f"embed_{key}": val for key, val in embed_stats.items()},
                        **route_meta,
                    },
                )

            prompt, prompt_id, prompt_version = _build_prompt(request.text, context_chunks, self._realm_id)
            bound.info("pipeline.generate.start", prompt_len=len(prompt), prompt_id=prompt_id, prompt_version=prompt_version)
            gen_model = getattr(self._generator, "_model", self._generator.generator_id)
            with lf_trace.generation("generate", model=gen_model, prompt=prompt) as sp:
                t0 = time.perf_counter()
                raw_text = self._generator.generate(prompt)
                trace.generate_ms = round((time.perf_counter() - t0) * 1000, 1)
                sp.set_output(raw_text)
            bound.info("pipeline.generate.done", answer_len=len(raw_text))

            # Optional grounding (config-driven). Flags unsupported claims.
            grounding_meta: dict[str, Any] = {}
            gr = None
            if self._grounder is not None:
                t0 = time.perf_counter()
                gr = self._grounder.check(raw_text, scored)
                trace.grounding_ms = round((time.perf_counter() - t0) * 1000, 1)
                trace.n_unsupported = len(gr.unsupported_claims)
                grounding_meta = {
                    "grounding_is_grounded": gr.is_grounded,
                    "grounding_confidence": gr.confidence,
                    "grounding_unsupported": gr.unsupported_claims,
                }
                bound.info(
                    "pipeline.grounding.done",
                    grounded=gr.is_grounded,
                    unsupported=trace.n_unsupported,
                )

            # Optional domain-pack refusal (config-driven). Reuses
            # the existing (previously unwired) core/refusal.py RefusalPolicy
            # contract: only meaningful when grounding was actually computed.
            refused = False
            refusal_reason = ""
            if self._refusal_policy is not None and gr is not None and self._refusal_policy.should_refuse(gr):
                refusal_answer = self._refusal_policy.build_refusal("low_confidence", request.text)
                raw_text = refusal_answer.text
                refused = True
                refusal_reason = refusal_answer.refusal_reason or "low_confidence"
                bound.info("pipeline.refusal", reason=refusal_reason)
            elif self._mask_engine is not None:
                # Optional domain-pack answer mask. Skipped on
                # refusal: a refusal template is already a complete answer.
                qtype = route_meta.get("route_question_type", "open")
                raw_text = self._mask_engine.render(
                    qtype, raw_text, source_paths=[s.chunk.structural_path for s in scored],
                )

            lf_trace.set_output(raw_text)
            trace.total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)

        cache_stats: dict[str, object] = {}
        if hasattr(self._embedder, "cache_stats"):
            cache_stats = self._embedder.cache_stats()  # type: ignore[assignment]

        # extract token counts if generator supports it
        if hasattr(self._generator, "_last_token_counts"):
            counts = self._generator._last_token_counts
            trace.input_tokens = counts.get("prompt_eval_count", 0)
            trace.output_tokens = counts.get("eval_count", 0)

        from core.citation import compute_citation_labels, substitute_fragment_markers
        computed_citations = [] if refused else compute_citation_labels(raw_text, source_refs)
        # Resolve any "Fragment N" marker actually visible to the user
        # in-place. See core/citation.py:substitute_fragment_markers for
        # why (the model is asked to cite by position only now; this is
        # what turns that position into the real, human-readable label).
        # The Russian "Фрагмент N" spelling is matched there too, because
        # a Realm's own prompt template may be written in any language.
        if not refused:
            raw_text = substitute_fragment_markers(raw_text, source_refs)

        return Answer(
            text=raw_text,
            source_refs=source_refs,
            pre_rerank_source_refs=pre_rerank_source_refs,
            candidate_source_refs=candidate_source_refs,
            computed_citations=computed_citations,
            refused=refused,
            refusal_reason=refusal_reason,
            stage_trace=trace,
            rendered_prompt_preview=prompt[:500],
            metadata={
                "pipeline": self.pipeline_id,
                "trace_id": request.trace_id,
                "retrieve_calls": self._retrieve_calls,
                "prompt_id": prompt_id,
                "prompt_version": prompt_version,
                # Found live: a run's configuration table had no way to show
                # which model actually generated its answers — only the
                # decorative config.generator.component_id ("ollama", the
                # adapter kind, never the model) — even though this exact
                # value was already computed above for the Langfuse trace
                # label and just never surfaced anywhere else.
                "generator_model": gen_model,
                **{f"embed_{k}": v for k, v in cache_stats.items()},
                **route_meta,
                **grounding_meta,
            },
        )

    def retrieve(self, request: QueryRequest) -> Answer:
        """Run the search and stop, returning sources with an empty text.

        core/experiment/runner.py switches a run to retrieval-only by asking
        whether its pipeline has this method, so its absence is what used to
        make `retrieval_only: true` an accepted and silently ignored setting
        on an in-process run: the config parsed, and every question still
        went to the model. That cost most where the Configuration Report
        needs the flag, because MIRACL ships relevance judgements and no
        reference answers, leaving nothing for a generated answer to be
        scored against.
        """
        return self.run(request, _generate=False)


class ConfigurablePipeline(NaivePipeline):
    """Pipeline whose steps (reranker / grounder / route_policy) come from config.

    Identical to ``NaivePipeline`` but defaults its ``pipeline_id`` to
    ``"configurable"`` and is the class the ExperimentRunner instantiates when
    building a pipeline from ``ExperimentConfig``.
    """

    pipeline_id = "configurable"


class _NoopTrace:
    """Silent no-op when Langfuse is disabled or unavailable."""

    def __enter__(self): return self
    def __exit__(self, *a): pass
    def set_output(self, *a): pass

    def span(self, *a, **kw):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield _NoopSpan()
        return _noop()

    def generation(self, *a, **kw):
        return self.span()


class _NoopSpan:
    def set_output(self, *a): pass
