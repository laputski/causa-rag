"""HttpPipeline — calls an external RAG service over HTTP.

Posts {query, top_k, filters, trace_id} to a configured URL and maps the
response into an Answer. If the response includes a "trace" object matching
the ExternalTrace contract (stage_trace/sources/rendered_prompt), the run
gets the same per-stage pipeline-trace as an in-process pipeline; otherwise
diagnostics degrade to outcome-only without raising an error.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from adapters.jsonpath_mapping import apply_response_mapping, render_request_template
from core.models import Answer, ExternalTrace, QueryRequest, SourceRef, StageTrace

_log = logging.getLogger(__name__)


class EgressNotAllowedError(PermissionError):
    """Raised when the configured URL is not on the egress allowlist."""


def _allowlist_from_env() -> list[str]:
    raw = os.getenv("RAG_HTTP_ALLOWLIST", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


# @lat: [[external-rag#Tier 2 — declarative JSONPath mapping]]
class HttpPipeline:
    """Pipeline backed by an external HTTP RAG service (model C)."""

    pipeline_id = "http"

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        allowlist: list[str] | None = None,
        transport: httpx.BaseTransport | None = None,
        # Optional retrieval-only endpoint declared at
        # registration time. When absent, retrieve() falls back to run() and
        # discards the answer text, so callers don't need to branch.
        retrieve_endpoint: str | None = None,
        # Tier 2 (declarative mapping). When set, run()
        # builds its request body from `request_template` (filling
        # {{query}}/{{top_k}}/{{filters}}/{{trace_id}} placeholders) instead
        # of the native {query, top_k, filters, trace_id} shape, and parses
        # the response via `response_mapping` (JSONPath) instead of expecting
        # the native {answer, sources} contract. Both None (the common case)
        # ⇒ tier 1 (native contract), unchanged behavior.
        request_template: dict[str, Any] | None = None,
        response_mapping: dict[str, str] | None = None,
        # Which corpus namespace the external RAG should
        # query, mirroring ExperimentConfig.corpus_id for in_process pipelines
        # (core/experiment/runner.py:_for_the_corpus). Without this, a
        # config's corpus_id silently had no effect on an external RAG — the
        # native contract had no field for it at all, so an external RAG
        # built against multiple corpora (e.g. services/reference_rag_server)
        # always answered from whatever corpus it happened to start up
        # against, regardless of which one the dataset/question set assumed.
        # None ⇒ omitted from the payload (back-compat with RAGs that don't
        # support multi-corpus and would reject/ignore an unknown field).
        corpus_id: str | None = None,
        # Which Realm this request belongs to. Without this, an
        # external RAG has no way to safely discover per-Realm facts about
        # corpus_id (e.g. GET /corpus/collections?realm_id=&corpus_id= for the
        # embedder(s) it was indexed with, see causa_rag_client's
        # get_corpus_embedders()) — corpus_id alone is not guaranteed unique
        # across Realms (the exact leak the realm-scoped collection
        # naming closed on the platform's own side; a RAG can't replicate that
        # safety without knowing realm_id too). None ⇒ omitted from the
        # payload, same back-compat posture as corpus_id.
        realm_id: str | None = None,
        # Which built-in strategy/reranker a dog-fooding
        # external RAG (services/reference_rag_server) should use, mirroring
        # ExperimentConfig.pipeline_id/.reranker for in_process pipelines.
        # Stored separately from the class-level `pipeline_id` attribute
        # (this pipeline's own identity, "http") to avoid confusing the two —
        # this is data sent TO the external RAG, not this object's own id.
        # A third-party RAG that doesn't recognize these extra JSON fields
        # simply ignores them; only services/reference_rag_server honors them.
        external_pipeline_id: str | None = None,
        reranker_id: str | None = None,
        # Extensible, capabilities-gated per-request
        # knobs (e.g. fetch_k, fusion alpha, generation temperature) that
        # don't warrant a dedicated contract field each. Generalizes the
        # pattern pipeline_id/corpus_id/reranker_id already established:
        # included verbatim under "params" only when non-empty, so a RAG
        # that doesn't recognize it just ignores the extra JSON key (same
        # honest-degradation contract as every other optional field here).
        params: dict[str, Any] | None = None,
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout
        self._external_pipeline_id = external_pipeline_id
        self._reranker_id = reranker_id
        self._params = params
        self._allowlist = allowlist if allowlist is not None else _allowlist_from_env()
        self._transport = transport
        self._retrieve_endpoint = retrieve_endpoint
        self._request_template = request_template
        self._response_mapping = response_mapping
        self._corpus_id = corpus_id
        self._realm_id = realm_id
        self._check_allowlist()
        if self._retrieve_endpoint:
            self._check_allowlist(self._retrieve_endpoint)

    def _check_allowlist(self, url: str | None = None) -> None:
        url = url or self._url
        if not self._allowlist:
            return
        if not any(url.startswith(prefix) for prefix in self._allowlist):
            raise EgressNotAllowedError(
                f"{url!r} is not on the egress allowlist {self._allowlist!r} "
                "(RAG_HTTP_ALLOWLIST) — refusing to call out."
            )

    @staticmethod
    def _raise_for_status_with_body(resp: httpx.Response) -> None:
        """`resp.raise_for_status()` alone only puts the status code and URL
        in the exception message — found live: a real 4xx from an external
        RAG (FastAPI-style services return 422s with a `detail` field naming
        exactly what's wrong, e.g. "Unknown corpus_id='handbook': no
        matching collection") was stored/surfaced as just "Client error '422
        Unprocessable Entity' for url '...'", with the one piece of
        information that actually explains the failure sitting unread in
        the response body. Same class of bug as
        qdrant-snapshot-restore's `UnexpectedResponse.__str__()` truncation
        (see the design notes) — re-raises with the body's `detail`
        (or the raw text if it isn't JSON/has no `detail` key) appended."""
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise httpx.HTTPStatusError(f"{e}: {detail}", request=e.request, response=e.response) from e

    def run(self, request: QueryRequest) -> Answer:
        headers = {"X-Trace-Id": request.trace_id, **self._headers}
        if self._request_template is not None:
            payload = render_request_template(
                self._request_template, request.text, request.top_k,
                request.filters or {}, request.trace_id,
                corpus_id=self._corpus_id,
                pipeline_id=self._external_pipeline_id,
                realm_id=self._realm_id,
            )
            # Inject corpus_id/pipeline_id/reranker_id as fallback top-level
            # fields when the template doesn't already include them. A template
            # can override placement via {{corpus_id}} / {{pipeline_id}}
            # placeholders; without them these fields are appended so the RAG
            # still receives the routing context it needs (e.g. corpus selection).
            if self._corpus_id is not None and "corpus_id" not in payload:
                payload["corpus_id"] = self._corpus_id
            if self._realm_id is not None and "realm_id" not in payload:
                payload["realm_id"] = self._realm_id
            if self._external_pipeline_id is not None and "pipeline_id" not in payload:
                payload["pipeline_id"] = self._external_pipeline_id
            if self._reranker_id is not None and "reranker_id" not in payload:
                payload["reranker_id"] = self._reranker_id
            if self._params and "params" not in payload:
                payload["params"] = self._params
        else:
            payload = {
                "query": request.text,
                "top_k": request.top_k,
                "filters": request.filters or {},
                "trace_id": request.trace_id,
            }
            if self._corpus_id is not None:
                payload["corpus_id"] = self._corpus_id
            if self._realm_id is not None:
                payload["realm_id"] = self._realm_id
            if self._external_pipeline_id is not None:
                payload["pipeline_id"] = self._external_pipeline_id
            if self._reranker_id is not None:
                payload["reranker_id"] = self._reranker_id
            if self._params:
                payload["params"] = self._params

        _log.debug(
            "http_pipeline.request",
            extra={"url": self._url, "corpus_id": self._corpus_id, "realm_id": self._realm_id,
                   "top_k": request.top_k, "trace_id": request.trace_id,
                   "payload_keys": list(payload.keys())},
        )
        t0 = time.monotonic()
        with httpx.Client(transport=self._transport, timeout=self._timeout) as client:
            resp = client.post(self._url, json=payload, headers=headers)
            self._raise_for_status_with_body(resp)
            body = resp.json()
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        _log.debug(
            "http_pipeline.response",
            extra={"url": self._url, "status": resp.status_code,
                   "elapsed_ms": elapsed_ms, "trace_id": request.trace_id,
                   "answer_len": len(body.get("answer", "")),
                   "sources_count": len(body.get("sources") or [])},
        )

        if self._response_mapping is not None:
            # Tier 2 — translate the RAG's own native response shape into
            # this platform's canonical {"answer", "sources"} before the
            # rest of this method (which only ever speaks tier 1's shape).
            body = apply_response_mapping(body, self._response_mapping)

        text = body.get("answer", "")
        raw_trace = body.get("trace")
        external_trace = ExternalTrace(**raw_trace) if raw_trace else ExternalTrace()

        stage_trace: StageTrace | None = external_trace.stage_trace
        # Sources are canonical at the top level of the response, and not
        # nested under trace; the older trace.sources shape is
        # read only as a backward-compat fallback when the top level omits it.
        raw_sources = body.get("sources")
        if raw_sources is not None:
            source_refs: list[SourceRef] = [SourceRef(**s) for s in raw_sources]
        else:
            source_refs = external_trace.sources or []
        rendered_prompt = external_trace.rendered_prompt or ""
        # Carry the external RAG's pre-rerank list through so
        # core/eval/funnel.py can separate retrieval-failure from rerank-failure
        # for an external RAG, exactly as for in-process. Absent ⇒ empty ⇒ funnel
        # degrades to coarse attribution (no false layer).
        pre_rerank = external_trace.pre_rerank_source_refs or []

        return Answer(
            text=text,
            source_refs=source_refs,
            pre_rerank_source_refs=pre_rerank,
            stage_trace=stage_trace,
            rendered_prompt_preview=rendered_prompt[:500],
            metadata={
                "pipeline": self.pipeline_id,
                "trace_id": request.trace_id,
                "external_url": self._url,
                "external_trace_available": stage_trace is not None,
                # Surfaced here (not a dedicated Answer field) the
                # same way external_url/trace_id already are; test_external_rag()
                # reads it back out to compare against the corpus's registered
                # embedder(s).
                "reported_embedders": external_trace.embedders,
                # Found live: a run's configuration table had no way to show
                # which model actually generated its answers for an external
                # RAG run either — a real external RAG's response body
                # already reports this, nested at `metadata.model` (its own
                # params.model override honored server-side — the response
                # dict is shaped {"metadata": {"model": <model used>, ...}}),
                # it was just never read here. First attempt at this fix
                # wrongly assumed a top-level `body["model"]`, which
                # live-verifying against a real response (curl'd directly)
                # showed was always None — the
                # field only ever exists under "metadata". Same Answer
                # metadata key ("generator_model") as core/pipeline.py's
                # in_process path, so ExperimentRunner.run() can capture it
                # identically regardless of pipeline_source. Absent for a RAG
                # that doesn't report it at all (honest degradation).
                #
                # Found live, again: for a tier-2 (response_mapping) RAG,
                # `body` was already replaced above with the mapped
                # {"answer", "sources", "generator_model"} shape — the native
                # response's "metadata" key no longer exists on it, so this
                # has to read the already-mapped `generator_model` instead of
                # re-reading a "metadata" key that tier 2 never produces. Also
                # guards against a RAG sending "metadata" as something other
                # than an object (a malformed/typo'd contract) — `.get()` on
                # a non-dict raised AttributeError and turned a good answer
                # into a per-question failure instead of degrading honestly.
                "generator_model": (
                    body.get("generator_model")
                    if self._response_mapping is not None
                    else (body.get("metadata").get("model") if isinstance(body.get("metadata"), dict) else None)
                ),
            },
        )

    def retrieve(self, request: QueryRequest) -> Answer:
        """retrieval-only path: POST to retrieve_endpoint
        and return only `sources` (no `answer`, no LLM call on the external
        side). Falls back to the full run() when no retrieve_endpoint was
        declared at registration time, so the runner can always call
        retrieve() uniformly.
        """
        if not self._retrieve_endpoint:
            return self.run(request)

        headers = {"X-Trace-Id": request.trace_id, **self._headers}
        if self._request_template is not None:
            payload = render_request_template(
                self._request_template, request.text, request.top_k,
                request.filters or {}, request.trace_id,
                corpus_id=self._corpus_id,
                pipeline_id=self._external_pipeline_id,
                realm_id=self._realm_id,
            )
            if self._corpus_id is not None and "corpus_id" not in payload:
                payload["corpus_id"] = self._corpus_id
            if self._realm_id is not None and "realm_id" not in payload:
                payload["realm_id"] = self._realm_id
            if self._external_pipeline_id is not None and "pipeline_id" not in payload:
                payload["pipeline_id"] = self._external_pipeline_id
            if self._reranker_id is not None and "reranker_id" not in payload:
                payload["reranker_id"] = self._reranker_id
            if self._params and "params" not in payload:
                payload["params"] = self._params
        else:
            payload = {
                "query": request.text,
                "top_k": request.top_k,
                "filters": request.filters or {},
                "trace_id": request.trace_id,
            }
            if self._corpus_id is not None:
                payload["corpus_id"] = self._corpus_id
            if self._realm_id is not None:
                payload["realm_id"] = self._realm_id
            if self._external_pipeline_id is not None:
                payload["pipeline_id"] = self._external_pipeline_id
            if self._reranker_id is not None:
                payload["reranker_id"] = self._reranker_id
            if self._params:
                payload["params"] = self._params

        _log.debug(
            "http_pipeline.retrieve.request",
            extra={"url": self._retrieve_endpoint, "corpus_id": self._corpus_id, "realm_id": self._realm_id,
                   "top_k": request.top_k, "trace_id": request.trace_id,
                   "payload_keys": list(payload.keys())},
        )
        t0 = time.monotonic()
        with httpx.Client(transport=self._transport, timeout=self._timeout) as client:
            resp = client.post(self._retrieve_endpoint, json=payload, headers=headers)
            self._raise_for_status_with_body(resp)
            body = resp.json()
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        _log.debug(
            "http_pipeline.retrieve.response",
            extra={"url": self._retrieve_endpoint, "status": resp.status_code,
                   "elapsed_ms": elapsed_ms, "trace_id": request.trace_id,
                   "sources_count": len(body.get("sources") or [])},
        )

        if self._response_mapping is not None:
            body = apply_response_mapping(body, self._response_mapping)

        raw_sources = body.get("sources") or []
        source_refs = [SourceRef(**s) for s in raw_sources]
        raw_trace = body.get("trace")
        external_trace = ExternalTrace(**raw_trace) if raw_trace else ExternalTrace()

        return Answer(
            text="",
            source_refs=source_refs,
            stage_trace=None,
            rendered_prompt_preview="",
            metadata={
                "pipeline": self.pipeline_id,
                "trace_id": request.trace_id,
                "external_url": self._retrieve_endpoint,
                "retrieval_only": True,
                "reported_embedders": external_trace.embedders,
            },
        )
