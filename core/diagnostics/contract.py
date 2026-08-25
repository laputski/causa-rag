"""What a served system can report about its own behaviour.

The platform already asks this of every *external* RAG at registration time
(`services/api_gateway/routers/external_rags.py` documents the shape under
`capabilities_declaration`). The platform's own template once
declared nothing at all, so the one system whose internals are fully known
was the one that said least about itself.

Declaring it has a practical consequence beyond tidiness: a capability that
is absent and a capability that is broken look identical from outside, and
only a declaration tells them apart. Without one, a missing pre-rerank
snapshot reads as "the reranker verdict is unavailable, something is wrong"
rather than "no reranker ran, there is nothing to snapshot".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Granularity = Literal["chunk", "document"]


@dataclass(frozen=True)
class DiagnosticContract:
    """Deliberately the same field names the external-RAG registration uses,
    so the platform reads one shape whether a system is its own or someone
    else's. `supports_pre_rerank` and `supports_trace_export` extend it;
    both are additive, so an external RAG that omits them is simply read as
    not offering them."""

    supports_trace: bool = False
    supports_retrieval_only: bool = False
    # Whether the ranked list *before* reranking is reported. Without it
    # `core/eval/funnel.py` structurally cannot tell a rerank failure from a
    # retrieval failure — it is not a nicety, it is what makes one of the
    # funnel's verdicts reachable at all.
    supports_pre_rerank: bool = False
    # Whether past query traces can be fetched back out. What phase 7 needs
    # to turn a real user's failure into a golden question.
    supports_trace_export: bool = False
    retrieve_endpoint: str | None = None
    max_top_k: int | None = None
    source_ref_granularity: Granularity = "chunk"
    # Request knobs the system actually reads. The platform never assumes a
    # key absent from this list does anything, which is the point: silently
    # ignored parameters are how a config change looks applied and is not.
    supported_params: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "supports_trace": self.supports_trace,
            "supports_retrieval_only": self.supports_retrieval_only,
            "supports_pre_rerank": self.supports_pre_rerank,
            "supports_trace_export": self.supports_trace_export,
            "retrieve_endpoint": self.retrieve_endpoint,
            "max_top_k": self.max_top_k,
            "source_ref_granularity": self.source_ref_granularity,
            "supported_params": list(self.supported_params),
        }


def contract_for_pipeline(
    pipeline: Any,
    *,
    retrieve_endpoint: str | None = None,
    max_top_k: int | None = None,
    supported_params: tuple[str, ...] = (),
    supports_trace_export: bool = False,
) -> DiagnosticContract:
    """Derives what it can from the pipeline itself, so a declaration cannot
    drift from the object it describes.

    `supports_pre_rerank` follows from whether a reranker is actually
    configured, rather than being asserted by hand: a pipeline with no
    reranker has no pre-rerank list to report, and claiming otherwise would
    make the platform wait for a snapshot that can never come.

    The rest are properties of the deployment (which endpoints are exposed,
    which knobs are wired) that the object cannot know about itself, so the
    caller states them.
    """
    has_reranker = getattr(pipeline, "_reranker", None) is not None
    return DiagnosticContract(
        supports_trace=True,
        supports_retrieval_only=retrieve_endpoint is not None,
        supports_pre_rerank=has_reranker,
        supports_trace_export=supports_trace_export,
        retrieve_endpoint=retrieve_endpoint,
        max_top_k=max_top_k,
        source_ref_granularity="chunk",
        supported_params=supported_params,
    )
