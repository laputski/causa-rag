"""Thin HTTP client for the RAG debugging/eval platform.

Wraps the existing gateway endpoints (services/api_gateway/routers/
external_rags.py, services/api_gateway/routers/experiments.py) with a small
typed surface: register_rag, upload_dataset, run_experiment, get_results,
wait_for_completion. Talks only HTTP (httpx) — no platform dependency (no
torch, no Qdrant/Mongo clients) so it installs cleanly inside a third-party
RAG project.

Eval (metrics, funnel attribution, regression) happens ON THE PLATFORM, not
in this client. This client only drives the
already-existing REST surface; it does not duplicate any scoring logic.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import httpx

# The contract version this client was built against
# (services/api_gateway/routers/external_rags.py:get_external_rag_spec
# returns {"version": ...} from the same Pydantic models that parse every
# response). Pinned to this package's own version (see pyproject.toml) so
# "which contract does this client speak" is never a separate, driftable
# question from "which client version is this".
CONTRACT_VERSION = "1.0.0"


class ContractVersionMismatch(RuntimeError):
    """Raised when the platform's published contract version doesn't match
    this client's CONTRACT_VERSION — an explicit "upgrade the library" error
    instead of a silent shape mismatch deep inside a request/response parse
    (the alternative this replaces: a config field this
    older/newer client doesn't know about gets silently dropped or
    misinterpreted, the way `corpus_id` and `pre_rerank_source_refs` each
    silently had no effect before the contract grew a field for them).
    """


class RagPlatformError(RuntimeError):
    """Raised for any non-2xx response from the platform, with the response
    body's detail (if any) included in the message."""


def _parse_semver(version: str) -> tuple[int, int, int] | None:
    """Best-effort X.Y.Z parse — None (not a ValueError) for anything that
    doesn't fit, so check_contract_version() can fall back to the old strict
    exact-match behavior for a malformed/unexpected version string instead of
    crashing on the parse itself."""
    parts = version.split(".")
    if len(parts) != 3:
        return None
    try:
        major, minor, patch = (int(p) for p in parts)
    except ValueError:
        return None
    return (major, minor, patch)


class RagPlatformClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        check_version: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport  # test-only hook (httpx.MockTransport)
        if check_version:
            self.check_contract_version()

    # ── Version check ─────────────────────────────────────────────

    def check_contract_version(self) -> str:
        """Fetch GET /external-rag-spec and compare its "version" against
        this client's CONTRACT_VERSION. Returns the platform's version on
        exact match or a same-major difference (warns in the latter case —
        see below); raises ContractVersionMismatch on a major-version
        difference (or an unparseable version on either side, preserving the
        old strict behavior as a safety net). Called automatically on
        __init__ unless check_version=False (e.g. for a platform that hasn't
        deployed /external-rag-spec yet — degrades to an explicit opt-out,
        not a silent skip).

        semver-aware, not exact-string-match: the platform's own
        contract additions (e.g. trace.embedders) bump only the minor version
        and are backward-compatible by construction (an older client that
        doesn't send/read the new optional field keeps working exactly as
        before). Hard-raising on every such bump would force every already-
        deployed RAG using this client to crash on its next restart purely
        because the platform shipped an additive feature — a forced outage,
        not a notification. A MAJOR version difference still raises: that's
        the platform's own signal that request/response shapes actually
        changed in a way an old client can't safely assume compatibility
        with.
        """
        spec = self._get("/external-rag-spec")
        platform_version = spec.get("version", "")
        if platform_version == CONTRACT_VERSION:
            return platform_version

        client_semver = _parse_semver(CONTRACT_VERSION)
        platform_semver = _parse_semver(platform_version)
        if client_semver is not None and platform_semver is not None and client_semver[0] == platform_semver[0]:
            warnings.warn(
                f"causa-rag-client {CONTRACT_VERSION} contract version differs from "
                f"the platform's {platform_version!r} at {self._base_url!r} (same major "
                f"version — additive/backward-compatible per this platform's own "
                f"versioning convention). Consider updating causa-rag-client to pick "
                f"up new optional fields — see the platform's GET /external-rag-spec "
                f"'changelog'.",
                stacklevel=2,
            )
            return platform_version

        raise ContractVersionMismatch(
            f"causa-rag-client {CONTRACT_VERSION} speaks contract version "
            f"{CONTRACT_VERSION!r}, but the platform at {self._base_url!r} "
            f"reports contract version {platform_version!r}. Upgrade the library: "
            f"(pip install -U causa-rag-client) or point at a matching "
            f"platform deployment — do not proceed, the request/response "
            f"shapes may have changed."
        )

    # ── HTTP helpers ─────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            resp = client.request(method, f"{self._base_url}{path}", **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RagPlatformError(f"{method} {path} -> {resp.status_code}: {detail}")
        return resp.json() if resp.content else None

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        return self._request("POST", path, json=body)

    # ── Registration ───────────────────────────────────────────────

    def register_rag(
        self,
        name: str,
        url: str,
        *,
        realm_id: str | None = None,
        description: str = "",
        headers: dict[str, str] | None = None,
        retrieve_endpoint: str | None = None,
        request_template: dict[str, Any] | None = None,
        response_mapping: dict[str, str] | None = None,
        supported_params: list[str] | None = None,
    ) -> dict[str, Any]:
        """Registers an external RAG (mirrors the "Resources and RAG endpoints"
        manual form) — POST /external-rags. Returns the saved record,
        including its generated `id` (pass as rag_id to run_experiment).

        `realm_id` scopes this registration to a Realm exactly
        like the manual form's Realm switcher does — omit it and the record
        gets `realm_id=None`, invisible in every Realm-scoped list (New
        Experiment's RAG picker, ChatPage's implementation selector,
        "Resources and RAG endpoints") even though it still exists and can be
        used by `rag_id` directly. There is no dedup: calling this twice —
        from the library, from the manual form, or both — always inserts a
        new record with a fresh generated `id`; nothing merges or conflicts
        by name/url. If you re-register the same RAG on every process start,
        either reuse the `id` this call returns (skip calling it again) or
        expect one new record per call.

        `supported_params` declares which keys of
        run_experiment's `params=` this RAG actually reads — e.g.
        `["fetch_k", "temperature"]`. Declared here, not probed, because
        whether a knob had any effect isn't observable from a single test()
        call; an un-declared key is simply never expected to do anything.
        """
        return self._post(
            "/external-rags",
            {
                "name": name,
                "url": url,
                "realm_id": realm_id,
                "description": description,
                "headers": headers or {},
                "retrieve_endpoint": retrieve_endpoint,
                "request_template": request_template,
                "response_mapping": response_mapping,
                "supported_params": supported_params or [],
            },
        )

    def list_rags(self, realm_id: str | None = None) -> list[dict[str, Any]]:
        """GET /external-rags?realm_id= — every RAG registered under this
        Realm (or every RAG at all, if realm_id is omitted).

        Meant for a RAG's OWN startup code to check "am I already
        registered?" before calling register_rag() again: that method has no
        dedup (calling it twice always inserts a new record, per its own
        docstring) — an unconditional register_rag() call in a service's
        lifespan/startup hook would insert a fresh duplicate on every
        process restart otherwise.
        """
        return self._get("/external-rags", params={"realm_id": realm_id} if realm_id else None)

    def get_corpus_embedders(self, realm_id: str, corpus_id: str) -> list[str]:
        """Which embedder(s) indexed this corpus — GET /corpus/collections?
        realm_id=. Replaces a hand-maintained local mapping
        (e.g. an `.env`-configured `corpus_collection_map` on the RAG side)
        with a live lookup against the platform's own registry, so a RAG
        doesn't silently assume its own default embedder applies to a corpus
        it never actually indexed itself — the bug that caused a hard Qdrant
        "Vector dimension error" the first time this was found live.

        `realm_id` is required, not optional: `corpus_id` alone is not
        guaranteed unique across Realms (the platform's own collection names
        are realm-scoped for exactly this reason — see
        the design notes).

        Returns every non-empty `embedder_id` across the corpus's registered
        backends (a corpus can legitimately have more than one — e.g. a
        GraphRAG-type corpus has a separate embedder for Neo4j community
        embeddings, distinct from the Qdrant chunk embedder). Empty list if
        the corpus isn't registered or has no `embedder_id` on any backend.
        """
        collections = self._get("/corpus/collections", params={"realm_id": realm_id})
        entry = next((c for c in collections or [] if c.get("corpus_id") == corpus_id), None)
        if entry is None:
            return []
        backends = entry.get("backends") or {}
        return [
            b["embedder_id"] for b in backends.values()
            if isinstance(b, dict) and b.get("embedder_id")
        ]

    def test_rag(self, rag_id: str) -> dict[str, Any]:
        """Probe the registered RAG once before a full run (POST
        /external-rags/{rag_id}/test) — surfaces connection_tier and a
        parsed_sources_preview so a bad mapping/doc_id scheme is caught
        before it's mistaken for a quality problem (see the design notes
        Tier 2 "limit of this tier").
        """
        return self._post(f"/external-rags/{rag_id}/test", {})

    def upload_dataset(
        self,
        rag_id: str,
        questions: list[dict[str, Any]],
        *,
        filename: str,
        realm_id: str,
    ) -> dict[str, Any]:
        """Uploads golden questions for this RAG — POST /datasets (previously POST /external-rags/{rag_id}/datasets, a separate
        second-class collection removed when RAG-authored and platform
        datasets were unified into one `datasets` collection — see
        the design notes). `rag_id` is now pure provenance
        (`source_rag_id`), not part of the URL. Each question may carry an
        explicit `answerability` field ("answerable"/"uncovered"/
        "out_of_scope"); without one, the platform's on-disk lookup for a
        numbered-unit corpus degrades to "uncovered" for everything, so for a corpus of any other shape you almost always want to
        set it explicitly.
        """
        return self._post(
            "/datasets",
            {"filename": filename, "realm_id": realm_id, "questions": questions, "source_rag_id": rag_id},
        )

    # ── Experiments ──────────────────────────────────────────────────────────

    def run_experiment(
        self,
        *,
        name: str,
        dataset_name: str,
        rag_id: str | None = None,
        http_endpoint: str | None = None,
        corpus_id: str = "default",
        pipeline_id: str = "naive",
        reranker_id: str | None = None,
        top_k: int = 5,
        metric_embedder_id: str = "bge_m3",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Starts an experiment run against a registered RAG (`rag_id`) or an
        inline endpoint (`http_endpoint`) — exactly one of the two. POST
        /experiments now returns immediately with a job handle; poll get_results()/wait_for_completion() for the outcome.

        `metric_embedder_id` selects the platform-side measurement embedder
        (semantic metrics), NOT a retrieval embedder inside the RAG — the
        RAG's own retrieval embedder is none of the platform's business, it
        only ever sees the RAG's `sources`/`answer` (see
        the design notes). v0 only
        supports the platform's fixed registered set (currently "bge_m3").

        `params` is the extensible, capabilities-gated block, omitted here in v0 pending that task; pass component-level
        knobs (reranker_id/top_k/pipeline_id/corpus_id) directly for now.
        """
        if bool(rag_id) == bool(http_endpoint):
            raise ValueError("run_experiment requires exactly one of rag_id or http_endpoint")

        config: dict[str, Any] = {
            "name": name,
            "pipeline_source": "http",
            "pipeline_id": pipeline_id,
            "corpus_id": corpus_id,
            "top_k": top_k,
            # Required ComponentRef fields on ExperimentConfig even though
            # the http branch of _build_pipeline never reads chunking_strategy/
            # generator — see the design notes "Decorative" section.
            "chunking_strategy": {"kind": "chunker", "component_id": "fixed"},
            "embedder": {"kind": "embedder", "component_id": metric_embedder_id},
            "generator": {"kind": "generator", "component_id": "ollama"},
        }
        if rag_id:
            config["external_rag_id"] = rag_id
        else:
            config["http_endpoint"] = http_endpoint
        if reranker_id:
            config["reranker"] = {"kind": "reranker", "component_id": reranker_id}
        if params:
            config["params"] = params

        return self._post("/experiments", {"config": config, "dataset_name": dataset_name})

    def get_results(self, run_id: str) -> dict[str, Any]:
        """GET /experiments/{run_id} — `status` is "running" while the
        backgrounded run is still in flight, "done" once aggregate_metrics/
        question_results are populated, or raises RagPlatformError (HTTP 500)
        if the background task failed.
        """
        return self._get(f"/experiments/{run_id}")

    def wait_for_completion(
        self, run_id: str, *, poll_interval: float = 1.0, timeout: float | None = None,
    ) -> dict[str, Any]:
        """Polls get_results() until status != "running". Raises TimeoutError
        if `timeout` (seconds) elapses first. No new transport — same
        GET /experiments/{run_id} a UI tab would poll, just synchronous.
        """
        start = time.monotonic()
        while True:
            result = self.get_results(run_id)
            if result.get("status") != "running":
                return result
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(f"run {run_id!r} still running after {timeout}s")
            time.sleep(poll_interval)
