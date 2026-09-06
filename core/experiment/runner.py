"""ExperimentRunner — orchestrates a pipeline run over an EvalDataset.

Builds the pipeline from ExperimentConfig (resolving components from registry),
runs each question, collects per-question results.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from core.experiment.config import ExperimentConfig
from core.models import Answer, QueryRequest
from core.registry import ComponentRegistry

if TYPE_CHECKING:
    # Under TYPE_CHECKING alone: the annotation below was a bare string that
    # resolved to nothing, so no type checker could read it and ruff reported an
    # undefined name. This makes it resolvable without adding a runtime import,
    # which is what the "imported at call site" note was protecting.
    from eval.dataset import EvalDataset

log = structlog.get_logger()


def _what_actually_ran(pipeline: Any) -> dict[str, Any]:
    """The embedder that queries, and the one whose vectors are searched.

    Two different things, and the whole point of recording them. The index a
    retriever reads is namespaced by the embedder that built it, so the
    retriever knows which model's vectors it is searching; the pipeline holds
    the model the query is embedded with. When those disagree, a query vector
    is being compared with vectors from another model, and every score is
    meaningless while every number still arrives.

    Read off the objects that will do the work, never off the configuration:
    four of its fields are accepted and never applied, so a record taken from
    it would state an intention. Empty for what cannot be known, an external
    system in particular, which embeds nothing here.
    """
    embedder = getattr(pipeline, "_embedder", None)
    retriever = getattr(pipeline, "_retriever", None)
    # A hybrid wraps two retrievers; the dense half is the one namespaced by
    # an embedder, and the sparse half never is.
    dense = getattr(retriever, "_dense", retriever)
    applied: dict[str, Any] = {}
    if embedder is not None:
        applied["query_embedder_id"] = getattr(embedder, "embedder_id", "")
        applied["query_embedder_version"] = getattr(embedder, "version", "")
        real = getattr(embedder, "is_real_model", None)
        if real is not None:
            applied["query_embedder_is_real_model"] = bool(real)
    if dense is not None:
        applied["index_embedder_id"] = getattr(dense, "_embedder_id", "")
        applied["index_chunking_strategy"] = getattr(dense, "_strategy_id", "")
    return {k: v for k, v in applied.items() if v != ""}


def _rebind_corpus_id(
    retriever: Any, corpus_id: str,
    realm_id: str | None = None, qdrant_cfg: dict[str, Any] | None = None,
    opensearch_cfg: dict[str, Any] | None = None,
) -> Any:
    """Rebuild `retriever` bound to `corpus_id`, `realm_id` and a Realm's own
    host:port if it isn't already.

    Registry-resolved pipelines (naive/hybrid_rrf/hybrid_weighted/graph) are
    constructed exactly ONCE at gateway startup, always with
    corpus_id="default" and env-var host/port (services/api_gateway/main.py
    never passes a corpus_id or Realm resource). Without this, both
    ExperimentConfig.corpus_id AND which Realm's Qdrant/OpenSearch instance
    an in_process run actually hit were decorative — every run silently
    queried the startup-time default collection/index on the gateway's own
    env-var instance, regardless of config.corpus_id or which Realm launched
    it (the exact same class of leak already found and fixed for the
    Content/Health/Graph tabs — see services/api_gateway/routers/
    corpus.py#_resolve_qdrant — just on the experiment-run path instead).

    `qdrant_cfg`/`opensearch_cfg` are the Realm's own resource dicts (same
    shape as corpus.py#_get_realm_resource's return value: `{"host":...,
    "port":...}`) — resolved async by the caller (core/ has no Mongo access
    of its own, mirrors the `external_rag_resolver` injection pattern on
    this same class) and passed in as plain data, not a callback. Absent
    ⇒ falls back to the retriever's own existing host/port (the behaviour before Realm-scoped resources,
    behavior, still exactly what registry-resolved pipelines without a
    Realm context get).

    Recurses through Hybrid/GraphHybrid wrappers, rebuilding only the
    dense/sparse leaves. The graph component is left untouched — it has no
    corpus_id (or realm_id) partitioning at all (one shared graph regardless
    of corpus_id), so there is nothing to rebind there.

    No-ops (returns the same instance) for anything without a `_corpus_id`
    attribute — stubs (QdrantRetrieverStub/OpenSearchRetrieverStub used in
    tests) and any retriever type this function doesn't know how to rebuild,
    rather than raising on an unrecognized type.
    """
    cls_name = type(retriever).__name__

    # Wrapper types carry no _corpus_id of their own (only their dense/sparse
    # leaves do) — must recurse unconditionally, BEFORE the leaf-only
    # already-matches check below, or this would always look like a no-op.
    if cls_name == "HybridRetriever":
        from core.retrieval.hybrid import HybridRetriever
        return HybridRetriever(
            dense_retriever=_rebind_corpus_id(retriever._dense, corpus_id, realm_id, qdrant_cfg, opensearch_cfg),
            sparse_retriever=_rebind_corpus_id(retriever._sparse, corpus_id, realm_id, qdrant_cfg, opensearch_cfg),
            embedder=retriever._embedder, merge=retriever._merge,
            alpha=retriever._alpha, rrf_k=retriever._rrf_k,
        )
    if cls_name == "GraphHybridRetriever":
        from core.retrieval.graph_hybrid import GraphHybridRetriever
        return GraphHybridRetriever(
            graph_retriever=retriever._graph,  # shared graph — not rebindable
            base_retriever=_rebind_corpus_id(retriever._base, corpus_id, realm_id, qdrant_cfg, opensearch_cfg),
            graph_weight=retriever._graph_weight, hops=retriever._hops,
        )

    current = getattr(retriever, "_corpus_id", None)
    current_realm = getattr(retriever, "_realm_id", None)
    if current is None:
        return retriever
    if current == corpus_id and current_realm == realm_id and qdrant_cfg is None and opensearch_cfg is None:
        return retriever

    if cls_name == "QdrantRetriever":
        from adapters.qdrant import QdrantRetriever
        return QdrantRetriever(
            host=(qdrant_cfg or {}).get("host", retriever._host),
            port=int((qdrant_cfg or {}).get("port", retriever._port)),
            strategy_id=retriever._strategy_id, embedder_id=retriever._embedder_id,
            corpus_id=corpus_id, realm_id=realm_id,
        )
    if cls_name == "OpenSearchRetriever":
        from adapters.opensearch import OpenSearchRetriever
        return OpenSearchRetriever(
            host=(opensearch_cfg or {}).get("host", retriever._host),
            port=int((opensearch_cfg or {}).get("port", retriever._port)),
            strategy_id=retriever._strategy_id, corpus_id=corpus_id, realm_id=realm_id,
            # Carried over like host/port/strategy_id above. Dropping it
            # reverts the rebound copy to the default analyser, so an
            # Arabic corpus ingested as Arabic would be queried through an
            # index this copy insists is Russian: either the wrong stemmer
            # or a refusal, and both arrive long after the choice was made.
            language=retriever._language,
        )
    return retriever



# Config fields that reach an external RAG through `params`, the contract
# carrying no field of its own for them. Written as a list because it grew: `fetch_k`
# was forwarded alone, and `rrf_k` arriving later would have repeated, one
# release apart, the exact defect the docstring below records.
_FORWARDED_TO_EXTERNAL = ("fetch_k", "rrf_k")


def _params_with_declared_fields(
    params: dict[str, Any] | None, config: Any, supported: list[str] | None,
) -> dict[str, Any] | None:
    """Forward the knobs of `_FORWARDED_TO_EXTERNAL` that a RAG declares reading.

    Found by re-verification: `fetch_k` reached only the two in-process
    branches, so a run against an external RAG accepted the field, recorded
    it in the config and changed nothing. That is the same shape
    `merge_strategy` once had: an accepted setting that does not exist.

    Forwarded through `params`, the established channel for knobs without a
    dedicated contract field, and **only for a knob the RAG named in its
    `supported_params`**. Sending one regardless would break the platform's
    own rule and would let a wide-window run look applied against a system
    that ignores the key.

    An explicit value already in `params` wins: a caller who typed one meant
    it, and silently overwriting it with the config field would make the
    raw-params escape hatch unreliable.
    """
    declared = set(supported or ())
    merged = dict(params or {})
    changed = False
    for field_name in _FORWARDED_TO_EXTERNAL:
        value = getattr(config, field_name, None)
        if value is None or field_name not in declared:
            continue
        merged.setdefault(field_name, value)
        changed = True
    return merged if changed else params


def _rebind_merge(retriever: Any, merge: str | None, alpha: float | None,
                  rrf_k: int | None = None) -> Any:
    """Apply a config's merge strategy and weight.

    `merge_strategy`/`merge_alpha` were decorative for their whole existence:
    `hybrid_rrf` and `hybrid_weighted` are two registry entries built at
    gateway startup with their merge baked in, so a config could name any
    strategy and any alpha and the run would use whatever the entry was
    constructed with. A preset promising a different weighting produced a
    byte-identical run, which is the same class of lie as the chunking preset
    that ran identical retrieval to every other.

    Applied by rebuilding the hybrid wrapper around the *same* dense and
    sparse retrievers rather than constructing new ones, so this composes
    with `_rebind_corpus_id` (which has already bound them to the right
    corpus by the time this runs) instead of undoing it.

    A no-op for anything that is not a hybrid retriever, and for values that
    already match — a dense-only pipeline has nothing to merge, and saying so
    by doing nothing is better than raising at a caller that merely passed
    its config along.
    """
    dense = getattr(retriever, "_dense", None)
    sparse = getattr(retriever, "_sparse", None)
    if dense is None or sparse is None:
        return retriever
    target_merge = merge or getattr(retriever, "_merge", "rrf")
    target_alpha = getattr(retriever, "_alpha", 0.5) if alpha is None else alpha
    # Carried, not defaulted. Rebuilding without it reset the fusion constant
    # to sixty, so asking for a different merge weight silently undid a
    # different rank-fusion constant set anywhere upstream. Measured: a
    # retriever built with rrf_k=10 came back with 60.
    target_rrf_k = getattr(retriever, "_rrf_k", 60) if rrf_k is None else rrf_k
    if (target_merge == getattr(retriever, "_merge", None)
            and target_alpha == getattr(retriever, "_alpha", None)
            and target_rrf_k == getattr(retriever, "_rrf_k", None)):
        return retriever
    try:
        return type(retriever)(
            dense_retriever=dense, sparse_retriever=sparse, embedder=retriever._embedder,
            merge=target_merge, alpha=target_alpha, rrf_k=target_rrf_k,
        )
    except Exception:
        # An unknown merge name raises in HybridRetriever's own constructor.
        # Degrading to the retriever as built beats failing a whole run over
        # one mistyped field, and the run records what it actually used.
        return retriever


def _rebind_generator(generator: Any, model: str | None) -> Any:
    """Rebuild `generator` bound to `model` if a per-run override was
    requested and it isn't already using it.

    Registry-resolved pipelines are constructed exactly ONCE at gateway
    startup with whatever OLLAMA_MODEL/settings.active_model was current at
    that time (services/api_gateway/main.py) — config.params["model"] used
    to be a complete no-op for an in_process run, the same way
    config.corpus_id used to be before _rebind_corpus_id existed: every run
    silently reused the startup-time model regardless of what was
    requested, and the only way to change it was PUT /settings/model —
    process-wide, affecting every Realm's chat too, not scoped to one run.
    An external RAG's own generation model was already selectable this same
    way (params["model"], read RAG-side by its own generation
    step) — this closes the matching gap for
    in_process runs, reusing the identical config key rather than a
    second, differently-shaped field.

    No-op (returns the same instance) when model is falsy, or for any
    generator type other than OllamaGenerator (a stub/vllm generator has no
    equivalent "just swap the model string" constructor shape) — same
    "graceful no-op for an unrecognized type" convention as
    _rebind_corpus_id above.
    """
    if not model or type(generator).__name__ != "OllamaGenerator":
        return generator
    if getattr(generator, "_model", None) == model:
        return generator
    from adapters.ollama_generator import OllamaGenerator
    return OllamaGenerator(base_url=generator._base_url, model=model, timeout=generator._timeout)


def _rebind_reranker(reranker: Any, params: dict[str, Any] | None) -> Any:
    """Rebuild `reranker` on the model a run asked for.

    `ComponentRef.params` reached nothing: the pipeline builder resolves a
    reranker out of the registry by its `component_id` and drops the params
    beside it, so a run could name any model and get the one the gateway
    started with. Measured: a reference asking for "BAAI/bge-reranker-v2-m3"
    produced a component that had never heard of the request.

    That absence is not cosmetic. Which model reranks decides which languages
    the rerank step understands, and a reranker that does not speak the
    corpus's language reorders by noise. The local reranker defaults to
    `cross-encoder/ms-marco-MiniLM-L-6-v2`, which is English only, while the
    default embedder is multilingual, so on a Russian corpus that mismatch was
    the platform's own default and a run had no way to say otherwise. The
    MIRACL sweep names the multilingual model explicitly for exactly this
    reason, and could do so only by building its own component.

    No-op for a reranker type with no model to swap, on the same convention as
    `_rebind_generator` above: a caller merely passing its config along should
    not have a run fail over a field that cannot apply to what it resolved.
    """
    model = (params or {}).get("model_name")
    if not model or not hasattr(reranker, "_model_name"):
        return reranker
    if reranker._model_name == model:
        return reranker
    try:
        return type(reranker)(model_name=model)
    except Exception:
        # A reranker whose constructor does not take the keyword. Degrading
        # beats failing a whole run, and the run records what it asked for.
        return reranker


@dataclass
class QuestionResult:
    question_id: str
    question: str
    reference_answer: str
    generated_answer: str
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    # Funnel diagnosis, Phase 1 — ranked list BEFORE the reranker cut it,
    # so retrieval_recall_at_k can be computed pre- and post-rerank
    # (core/eval/funnel.py). Empty when no reranker ran.
    pre_rerank_source_refs: list[dict[str, Any]] = field(default_factory=list)
    # The candidate window before the final cut, present only when
    # fetch_k widened it past top_k. What core/eval/counterfactual.py reads to
    # answer "what would a different context size have shown" without running
    # the experiment again.
    candidate_source_refs: list[dict[str, Any]] = field(default_factory=list)
    # The golden refs this question was measured against. Stored
    # on the result because a counterfactual read later needs to know what
    # counted as correct, and the dataset may have changed since. Without it
    # the only refs available at read time are the ones a root cause happens
    # to carry, which exist for a fraction of failures and skew every
    # aggregate computed from them.
    expected_refs: list[str] = field(default_factory=list)
    # core/citation.py:compute_citation_labels() — citation derived from
    # retrieved chunk metadata (structural_path), not from whatever number
    # the LLM wrote in the answer text. Already computed on every Answer
    # (core/pipeline.py) and shown in the live chat (ChatPage.tsx), but
    # never carried through to experiment runs before now — found live: a
    # 22% literal-citation-hallucination rate (always article 1, regardless of the
    # actually-retrieved article) was invisible on RunPage because nothing
    # there read this field, even though the data existed on every Answer.
    computed_citations: list[str] = field(default_factory=list)
    # Per-stage latency/counters (core/models.py:StageTrace) captured on every
    # Answer by core/pipeline.py but previously dropped when building the
    # per-question result — so RunPage could show aggregate metrics but never
    # a per-stage latency breakdown. None when the pipeline produced no trace
    # (e.g. an external RAG that returns no ExternalTrace.stage_trace), which
    # the UI renders as "trace unavailable" rather than a fake zero-latency
    # diagram (honest degradation).
    stage_trace: dict[str, Any] | None = None
    # Set when this question's pipeline.run()/retrieve() call raised —
    # found live: an external RAG timing out on question 47 of 143 (a slow
    # or intermittently-hanging service, not a platform bug) aborted the
    # whole ExperimentRunner.run() loop, discarding every already-answered
    # question's results and reporting the entire run as failed. A per-
    # question error is now recorded here (empty answer, metrics stay {} —
    # same "absent ⇒ not counted" convention the aggregation below already
    # relies on) and the loop continues, so a flaky external RAG costs you
    # that one question's metrics, not the whole run's progress.
    error: str | None = None
    # core/eval/answerability.py's Answerability class ("answerable"/
    # "uncovered"/"out_of_scope"), as actually resolved at eval time by
    # whatever evaluator ran this question (see _CompositeEvaluator.
    # resolve_answerability in services/api_gateway/routers/experiments.py).
    # Found live: without this, GET /experiments/{run_id} could only guess
    # "answerable" vs "not" from whether `retrieval_recall_at_k` was present
    # in `metrics` — collapsing "uncovered" and "out_of_scope" into one
    # indistinguishable bucket, so a question with NO golden article_refs at
    # all (out_of_scope) displayed the same "reference source missing from
    # corpus (uncovered)" wording as a question whose refs genuinely don't
    # resolve. None for evaluators that don't expose this (or no evaluator
    # at all) — the display falls back to the old best-effort inference.
    answerability: str | None = None
    # Why this question's retrieval failed, not just that it did.
    # Shape: core/eval/root_cause.py#RootCause.to_dict(). None when the
    # question did not fail on retrieval, or when the analysis could not run
    # (no index access), which is deliberately different from "analysed and
    # found nothing": the latter is recorded as cause "unknown".
    #
    # Filled by the services layer after the run rather than during it,
    # because establishing the cause needs a widened re-query against the
    # index and a ref resolver, neither of which core/ may reach for itself.
    root_cause: dict[str, Any] | None = None


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    question_results: list[QuestionResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    n_questions: int = 0
    dataset_name: str = ""
    prompt_id: str = ""
    prompt_version: int = 0
    # Which Realm this run belongs to. "" = no Realm (legacy/global).
    realm_id: str = ""
    # Found live: the run configuration table had no way to show
    # which model actually generated the answers — config.generator.
    # component_id is decorative (always "ollama", the adapter kind, see
    # the design notes "Decorative"). Captured from the first
    # answer's metadata the same way prompt_id/prompt_version already are —
    # both in_process (core/pipeline.py) and http (adapters/http_pipeline.py,
    # reading the external RAG's own reported model) populate the same
    # "generator_model" metadata key.
    generator_model: str = ""
    # True when a user stopped this run before it reached every question (see
    # ExperimentRunner.run's should_stop) — distinguishes "stopped on
    # purpose, partial results below n_questions is expected" from a run
    # that quietly answered fewer questions than promised for some other
    # reason. len(question_results) vs n_questions already carries the
    # count; this just makes the reason explicit for the UI.
    stopped: bool = False
    # Whether this run's answerability classes were verified
    # against what is actually indexed, or merely trusted from the dataset's
    # own refs. Shape: {"checked": bool, "reason": str}.
    #
    # Recorded because "not verified" used to be invisible: an unreadable
    # corpus silently classified every question "uncovered", which excludes
    # it from retrieval metrics entirely, so the numbers became meaningless
    # with nothing anywhere saying so. Carrying the flag onto the run is what
    # lets core/eval/detectors.py surface it (see detect_unverified_coverage)
    # instead of the reader having to guess whether coverage was real.
    #
    # Set by the services layer after the run (it owns index access); an
    # empty dict means a run stored before this field existed, which the
    # detector treats as "nothing to say" rather than as a problem.
    coverage_check: dict[str, Any] = field(default_factory=dict)
    # What actually ran, as opposed to what the configuration asked for. The
    # two differ today: `config.embedder` is accepted and never applied, both
    # branches of the build take the registry's own embedder, so a check
    # reading the configuration would compare an intention with a record and
    # neither would be what happened.
    applied: dict[str, Any] = field(default_factory=dict)
    # What the corpus this run queried was built from and built by. Filled by
    # the services layer, which can reach the registry; core may not.
    corpus_manifest: dict[str, Any] = field(default_factory=dict)
    # Components this run's configuration named and the registry could not
    # produce. The pipeline builder degrades to running without them, on
    # purpose: an uninstalled reranker extra should not fail a whole run. What
    # it did not do was say so, so a run whose configuration named a reranker
    # and that reranked nothing was indistinguishable, on every screen and in
    # every stored document, from one that did.
    #
    # That distinction decides whether a proving-ground pair means anything: a
    # bait pairing two rerankers proves nothing at all when neither ran.
    # Empty on a run stored before this field existed, which reads as "nothing
    # to say" and never as "everything was available".
    unavailable_components: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config.config_hash,
            "config_name": self.config.name,
            "config": self.config.model_dump(),
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "n_questions": self.n_questions,
            "dataset_name": self.dataset_name,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "realm_id": self.realm_id,
            "stopped": self.stopped,
            "generator_model": self.generator_model,
            "coverage_check": self.coverage_check,
            "applied": self.applied,
            "corpus_manifest": self.corpus_manifest,
            "unavailable_components": self.unavailable_components,
            "aggregate_metrics": self.aggregate_metrics,
            "question_results": [
                {
                    "question_id": qr.question_id,
                    "question": qr.question,
                    "reference_answer": qr.reference_answer,
                    "generated_answer": qr.generated_answer,
                    "metrics": qr.metrics,
                    # Needed by the detectors and retrieval diagnostics.
                    "source_refs": qr.source_refs,
                    "pre_rerank_source_refs": qr.pre_rerank_source_refs,
                    "candidate_source_refs": qr.candidate_source_refs,
                    "expected_refs": qr.expected_refs,
                    "computed_citations": qr.computed_citations,
                    "stage_trace": qr.stage_trace,
                    "error": qr.error,
                    "answerability": qr.answerability,
                    "root_cause": qr.root_cause,
                }
                for qr in self.question_results
            ],
            # Dataset-averaged per-stage latency, so RunPage can show one
            # pipeline diagram for the whole run without re-averaging on the
            # client. Empty when no question produced a stage_trace.
            "avg_stage_trace": self.avg_stage_trace(),
            "n_errors": self.n_errors(),
        }

    def n_errors(self) -> int:
        return sum(1 for qr in self.question_results if qr.error)

    # Stage/latency keys averaged for the dataset-level diagram. Counters
    # (n_*, *_tokens, *_chars) are averaged too — a mean chunk/token count per
    # question is a meaningful funnel signal, same as mean latency.
    def avg_stage_trace(self) -> dict[str, float] | None:
        traces = [qr.stage_trace for qr in self.question_results if qr.stage_trace]
        if not traces:
            return None
        keys = {k for tr in traces for k in tr}
        return {
            k: round(sum(float(tr.get(k, 0) or 0) for tr in traces) / len(traces), 2)
            for k in keys
        }


class ExperimentRunner:
    """Runs a full experiment: config → pipeline → dataset → metrics."""

    def __init__(
        self,
        registry: ComponentRegistry,
        external_rag_resolver: Any | None = None,
    ) -> None:
        self._registry = registry
        # Resolves config.external_rag_id -> a dict
        # of the fields HttpPipeline needs (url, headers, retrieve_endpoint,
        # request_template, response_mapping) from the registered ExternalRag
        # record. Injected by the gateway (which owns the Mongo layer) so
        # core/ stays free of persistence deps.
        # Signature: external_rag_resolver(rag_id: str) -> dict[str, Any].
        self._external_rag_resolver = external_rag_resolver

    def _build_pipeline(
        self, config: ExperimentConfig, realm_id: str = "",
        qdrant_cfg: dict[str, Any] | None = None, opensearch_cfg: dict[str, Any] | None = None,
        retrieval_pins: list[Any] | None = None,
        unavailable: list[str] | None = None,
    ) -> Any:
        """Build the pipeline the config describes.

        `unavailable`, when given, collects every component the config named
        and the registry could not produce. Collected, and never raised, since
        an uninstalled reranker extra should not fail a whole run. Reported,
        and no longer dropped: a run that named a reranker and reranked nothing
        used to look identical to one that reranked.

        `realm_id`/`qdrant_cfg`/`opensearch_cfg` — see
        _rebind_corpus_id's docstring. Only affects the in_process branch
        below; the http branch already carries corpus_id through the HTTP
        contract itself and has no Realm-owned in-process infra
        to rebind.

        Always rebinds the retriever to config.corpus_id (see
        _rebind_corpus_id — registry-resolved pipelines are frozen at
        gateway startup on corpus_id="default") and applies config.top_k.
        A config without reranker/grounding/route_policy/etc. gets a fresh
        instance of the SAME pipeline class as the registered one (not the
        literal registered instance — that one is corpus_id="default"-bound
        and shared across every run). When any optional step is set, the
        components are wrapped in a ConfigurablePipeline with those steps
        resolved from the registry. Missing optional components (e.g. an
        uninstalled reranker extra) degrade gracefully — the step is
        skipped with a warning, never a hard failure.
        """
        if config.pipeline_source == "http":
            # An external RAG by URL, not a registered
            # component. Bypasses the registry entirely; grounding/
            # route_policy/scorer/mask_engine/refusal_policy don't apply (the
            # external service owns its pipeline and has no declared
            # equivalent for those). pipeline_id and reranker DO travel
            # through, mirroring corpus_id: a
            # dog-fooding external RAG like reference_rag_server can honor
            # them, and a third-party RAG that doesn't recognize the extra
            # JSON fields simply ignores them.
            from adapters.http_pipeline import HttpPipeline
            # Prefer a registered external_rag_id (resolved
            # via the injected resolver) over an inline endpoint.
            # retrieve_endpoint enables retrieval_only runs, and
            # request_template/response_mapping (both may be None) enable the
            # tier-2 declarative mapping.
            reranker_id = config.reranker.component_id if config.reranker else None
            if config.external_rag_id and self._external_rag_resolver is not None:
                resolved = self._external_rag_resolver(config.external_rag_id)
                return HttpPipeline(
                    url=resolved["url"],
                    headers=resolved.get("headers") or {},
                    retrieve_endpoint=resolved.get("retrieve_endpoint"),
                    request_template=resolved.get("request_template"),
                    response_mapping=resolved.get("response_mapping"),
                    # Without this, an external RAG built
                    # against multiple corpora (e.g. reference_rag_server) had
                    # no way to know which one this run targets and silently
                    # answered from whichever it started up against.
                    corpus_id=config.corpus_id,
                    realm_id=realm_id or None,
                    external_pipeline_id=config.pipeline_id,
                    reranker_id=reranker_id,
                    # Extensible, capabilities-gated knobs.
                    params=_params_with_declared_fields(
                        config.params, config, resolved.get("supported_params"),
                    ) or None,
                    # Registered per-RAG override of HttpPipeline's 30s default —
                    # some RAGs (e.g. a multi-step agentic one) genuinely need
                    # longer. Absent/falsy ⇒ the 30s default, unchanged for
                    # every RAG that doesn't set it.
                    timeout=resolved.get("timeout_s") or 30.0,
                )
            if not config.http_endpoint:
                raise ValueError(
                    "pipeline_source='http' requires http_endpoint or a resolvable external_rag_id"
                )
            return HttpPipeline(
                url=config.http_endpoint, corpus_id=config.corpus_id, realm_id=realm_id or None,
                external_pipeline_id=config.pipeline_id, reranker_id=reranker_id,
                params=config.params or None,
            )

        base = self._registry.resolve("pipeline", config.pipeline_id)
        retriever = _rebind_corpus_id(
            base._retriever, config.corpus_id, realm_id or None, qdrant_cfg, opensearch_cfg,
        )
        # After the corpus rebind, so the merge wrapper is rebuilt
        # around retrievers already bound to the right corpus.
        retriever = _rebind_merge(retriever, config.merge_strategy, config.merge_alpha, config.rrf_k)
        generator = _rebind_generator(base._generator, (config.params or {}).get("model"))

        if (config.reranker is None and config.grounding is None and config.route_policy is None
                and config.scorer is None and config.mask_engine is None and config.refusal_policy is None):
            return type(base)(
                retriever=retriever,
                embedder=base._embedder,
                generator=generator,
                top_k=config.top_k,
                fetch_k=config.fetch_k,
                pipeline_id=base.pipeline_id,
                realm_id=realm_id or None,
                retrieval_pins=retrieval_pins,
            )

        from core.pipeline import ConfigurablePipeline

        def _maybe(kind: str, ref: Any) -> Any:
            if ref is None:
                return None
            try:
                return self._registry.resolve(kind, ref.component_id)
            except KeyError:
                log.warning("experiment.component.unavailable", kind=kind, id=ref.component_id)
                if unavailable is not None:
                    unavailable.append(f"{kind}:{ref.component_id}")
                return None

        return ConfigurablePipeline(
            retriever=retriever,
            embedder=base._embedder,
            generator=generator,
            top_k=config.top_k,
            fetch_k=config.fetch_k,
            reranker=_rebind_reranker(_maybe("reranker", config.reranker),
                                      config.reranker.params if config.reranker else None),
            grounder=_maybe("grounder", config.grounding),
            route_policy=_maybe("route_policy", config.route_policy),
            scorer=_maybe("scorer", config.scorer),
            mask_engine=_maybe("mask_engine", config.mask_engine),
            refusal_policy=_maybe("refusal", config.refusal_policy),
            realm_id=realm_id or None,
            retrieval_pins=retrieval_pins,
        )

    def run(
        self,
        config: ExperimentConfig,
        dataset: EvalDataset,
        evaluator: Any | None = None,
        # Called after each question with
        # (processed, total) so a caller running this on a background
        # thread/task can surface live progress (e.g. into the WebSocket
        # /{run_id}/progress stream) without this class knowing about HTTP,
        # threads, or asyncio. Optional: a caller with no progress UI passes
        # nothing and nothing changes (honest degradation).
        on_progress: Any | None = None,
        # Found live: a run picking a slow model (or a stuck/misbehaving
        # external RAG) had no way to be interrupted short of waiting out
        # every remaining question — checked once per question (same
        # granularity as on_progress, same "no callback ⇒ no behavior
        # change" honest-degradation convention). A plain
        # `() -> bool`, not an asyncio/threading primitive — the caller
        # (services/api_gateway/routers/experiments.py, running this via
        # asyncio.to_thread) owns how "stop requested" is actually signaled
        # across the thread boundary; this class only ever calls it.
        should_stop: Any | None = None,
        # pre-resolved by the caller (services/api_gateway/routers/
        # experiments.py, which owns the async Mongo lookup), not fetched
        # here: core/ has no persistence deps of its own, same reasoning as
        # external_rag_resolver on __init__. Without these, an in_process run
        # always hit the gateway's own startup-time env-var Qdrant/OpenSearch
        # instance regardless of which Realm launched it — config.corpus_id
        # alone wasn't enough once two Realms could register distinct
        # physical resources (see _rebind_corpus_id's docstring).
        realm_id: str = "",
        qdrant_cfg: dict[str, Any] | None = None,
        opensearch_cfg: dict[str, Any] | None = None,
        # Same "caller pre-resolves it, core/ has no Mongo
        # access" convention as qdrant_cfg/opensearch_cfg above: the active
        # retrieval_pins for this run's (realm_id, corpus_id) are fetched
        # once by services/api_gateway/routers/experiments.py before run()
        # is even called, then threaded straight through to the pipeline.
        retrieval_pins: list[Any] | None = None,
    ) -> ExperimentResult:
        bound = log.bind(config_hash=config.config_hash, experiment=config.name)
        bound.info("experiment.start", dataset=dataset.name, n_questions=len(dataset.questions))
        started_at = datetime.datetime.now(datetime.UTC).isoformat()

        # The run's own identity, independent of any external tracker: name
        # plus the config hash that defines what was actually run. The gateway
        # overwrites it with its own uuid for API-created runs
        # (services/api_gateway/routers/experiments.py), so this is what a
        # direct/scripted runner call gets.
        run_id = f"{config.name}_{config.config_hash}"

        unavailable: list[str] = []
        pipeline = self._build_pipeline(
            config, realm_id, qdrant_cfg, opensearch_cfg, retrieval_pins, unavailable,
        )
        result = ExperimentResult(
            config=config,
            run_id=run_id,
            started_at=started_at,
            n_questions=len(dataset.questions),
            dataset_name=dataset.name,
            unavailable_components=unavailable,
            applied=_what_actually_ran(pipeline),
        )

        # retrieval_only calls the cheaper retrieve() path
        # when the pipeline exposes one (HttpPipeline always does; it falls
        # back to run() itself when no retrieve_endpoint was declared, and
        # NaivePipeline stops before the generator — see core/pipeline.py).
        # A pipeline without the method still generates, so the flag stays
        # a silent no-op there; the hasattr is what makes that silent.
        use_retrieve_only = config.retrieval_only and hasattr(pipeline, "retrieve")
        total = len(dataset.questions)

        for i, q in enumerate(dataset.questions):
            if should_stop is not None and should_stop():
                bound.info("experiment.stopped", processed=i, total=total)
                result.stopped = True
                break
            req = QueryRequest(
                text=q["question"],
                top_k=config.top_k,
            )
            try:
                answer: Answer = pipeline.retrieve(req) if use_retrieve_only else pipeline.run(req)
            except Exception as exc:
                # One question's transient failure (a slow/intermittently-
                # hanging external RAG, most commonly) used to abort the
                # entire run and discard every already-answered question's
                # results — see QuestionResult.error. Logged at the same
                # level as the eventual full-run failure would have been,
                # so this is still visible without digging through
                # per-question rows.
                bound.warning(
                    "experiment.question.failed",
                    question_id=q.get("id", q["question"][:20]),
                    index=i, total=total, error=str(exc),
                )
                result.question_results.append(QuestionResult(
                    question_id=q.get("id", q["question"][:20]),
                    question=q["question"],
                    reference_answer=q.get("reference_answer") or q.get("ground_truth", ""),
                    generated_answer="",
                    error=str(exc),
                ))
                if on_progress is not None:
                    on_progress(i + 1, total)
                continue

            if not result.prompt_id and answer.metadata:
                result.prompt_id = str(answer.metadata.get("prompt_id", ""))
                result.prompt_version = int(answer.metadata.get("prompt_version", 0))
            if not result.generator_model and answer.metadata and answer.metadata.get("generator_model"):
                result.generator_model = str(answer.metadata["generator_model"])
            qr = QuestionResult(
                question_id=q.get("id", q["question"][:20]),
                question=q["question"],
                reference_answer=q.get("reference_answer") or q.get("ground_truth", ""),
                generated_answer=answer.text,
                source_refs=[sr.model_dump() for sr in answer.source_refs],
                pre_rerank_source_refs=[sr.model_dump() for sr in answer.pre_rerank_source_refs],
                candidate_source_refs=[sr.model_dump() for sr in getattr(answer, 'candidate_source_refs', []) or []],
                expected_refs=list(q.get('article_refs') or []),
                computed_citations=answer.computed_citations,
                stage_trace=answer.stage_trace.model_dump() if answer.stage_trace else None,
            )

            if evaluator is not None:
                qr.metrics = evaluator.evaluate(q, answer)
                # Optional — a generic `evaluator: Any` (e.g. eval/gate.py's
                # CI usage) isn't required to expose this; evaluate()'s own
                # return contract (dict[str, float]) stays unchanged so
                # nothing there needs to change.
                resolve = getattr(evaluator, "resolve_answerability", None)
                if resolve is not None:
                    qr.answerability = resolve(q)

            result.question_results.append(qr)

            if on_progress is not None:
                on_progress(i + 1, total)

        # Aggregate metrics
        if result.question_results:
            all_metrics: dict[str, list[float]] = {}
            for qr in result.question_results:
                for k, v in qr.metrics.items():
                    all_metrics.setdefault(k, []).append(v)
            result.aggregate_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

        # Observability: warn when retrieval is completely blind (recall=0 for
        # every question) — this almost always signals a corpus mismatch or a
        # broken source_code/article_no field in the external RAG response.
        if result.question_results:
            recalls = [qr.metrics.get("retrieval_recall_at_k", -1) for qr in result.question_results]
            n_zero = sum(1 for r in recalls if r == 0.0)
            n_measured = sum(1 for r in recalls if r >= 0)
            if n_measured > 0 and n_zero == n_measured:
                bound.warning(
                    "experiment.retrieval.total_failure",
                    corpus_id=config.corpus_id,
                    pipeline_source=config.pipeline_source,
                    external_rag_id=getattr(config, "external_rag_id", None),
                    hint="retrieval_recall_at_k=0 for every question — "
                        "check corpus_id forwarding, source_code/article_no "
                        "field names, and that the external RAG's corpus "
                        "contains the expected documents",
                )
            elif n_measured > 0 and n_zero / n_measured > 0.9:
                bound.warning(
                    "experiment.retrieval.high_miss_rate",
                    zero_recall_fraction=round(n_zero / n_measured, 3),
                    corpus_id=config.corpus_id,
                )

        result.finished_at = datetime.datetime.now(datetime.UTC).isoformat()

        bound.info("experiment.done", metrics=result.aggregate_metrics)
        return result
