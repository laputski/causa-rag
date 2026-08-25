"""Tier-2 declarative mapping — translate between this
platform's contract and an external RAG's own native request/response shape
using a config-only JSONPath mapping, no code on either side.

Two independent halves:
  - render_request_template(): fills `{{query}}`/`{{top_k}}`/`{{filters}}`/
    `{{trace_id}}` placeholders into the RAG's own native request body shape.
  - apply_response_mapping(): extracts `answer`/`sources[]` out of the RAG's
    own native response shape via JSONPath expressions, into this platform's
    canonical {"answer", "sources"} shape.

Tier 1 (native contract) bypasses this module entirely — HttpPipeline only
calls into here when an ExternalRag record carries a `response_mapping`.
"""
from __future__ import annotations

from typing import Any

from jsonpath_ng import parse as _parse_jsonpath

_PLACEHOLDERS = ("{{query}}", "{{top_k}}", "{{filters}}", "{{trace_id}}", "{{corpus_id}}", "{{pipeline_id}}")


# @lat: [[external-rag#Tier 2 — declarative JSONPath mapping]]
def render_request_template(
    template: dict[str, Any],
    query: str,
    top_k: int,
    filters: dict[str, Any],
    trace_id: str,
    corpus_id: str | None = None,
    pipeline_id: str | None = None,
    realm_id: str | None = None,
) -> dict[str, Any]:
    """Fill placeholders into `template` (the RAG's own native request body
    shape, as configured at registration time). A value that IS exactly one
    placeholder keeps the placeholder's native type (e.g. top_k stays an
    int); a placeholder embedded in a larger string is interpolated as text.
    """
    subs: dict[str, Any] = {
        "{{query}}": query,
        "{{top_k}}": top_k,
        "{{filters}}": filters,
        "{{trace_id}}": trace_id,
        "{{corpus_id}}": corpus_id or "",
        "{{pipeline_id}}": pipeline_id or "",
        "{{realm_id}}": realm_id or "",
    }

    def _render(node: Any) -> Any:
        if isinstance(node, str):
            if node in subs:
                return subs[node]
            rendered = node
            for placeholder, value in subs.items():
                if placeholder in rendered:
                    rendered = rendered.replace(placeholder, str(value))
            return rendered
        if isinstance(node, dict):
            return {k: _render(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_render(v) for v in node]
        return node

    return _render(template)  # type: ignore[no-any-return]


def _extract_one(expr: str | None, data: Any) -> Any:
    if not expr:
        return None
    try:
        matches = _parse_jsonpath(expr).find(data)
    except Exception:
        # A misconfigured JSONPath is a mapping bug (a risk noted at
        # registration), not a platform crash — degrade to "not found" so
        # the caller's existing default-value handling takes over.
        return None
    return matches[0].value if matches else None


# @lat: [[external-rag#Tier 2 — declarative JSONPath mapping]]
def apply_response_mapping(raw: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Extract this platform's canonical {"answer", "sources", "generator_model"}
    shape out of `raw` (the RAG's own native response body) using JSONPath
    expressions in `mapping`. Recognized keys: answer, sources (path to the
    list), generator_model (optional — the RAG's own model name, wherever it
    lives in the native shape), and source_doc_id/source_chunk_id/
    source_score/source_text/source_structural_path/source_code/
    source_article_no (each evaluated relative to one item of the `sources`
    list).
    """
    answer = _extract_one(mapping.get("answer"), raw) or ""
    sources_raw = _extract_one(mapping.get("sources"), raw) or []
    # Found live: a tier-2-mapped RAG's model name was unrecoverable — the
    # native response body is discarded once mapped into this canonical
    # shape, and there was no mapping key to carry it through. Same
    # honest-degradation posture as every other field here: absent mapping
    # key or no match ⇒ "" (adapters/http_pipeline.py treats "" like a RAG
    # that never reported a model at all).
    generator_model = _extract_one(mapping.get("generator_model"), raw) or ""

    sources: list[dict[str, Any]] = []
    for item in sources_raw:
        sources.append(
            {
                "doc_id": _extract_one(mapping.get("source_doc_id"), item) or "",
                "chunk_id": _extract_one(mapping.get("source_chunk_id"), item) or "",
                "score": _extract_one(mapping.get("source_score"), item) or 0.0,
                "chunk_text": _extract_one(mapping.get("source_text"), item) or "",
                "structural_path": _extract_one(mapping.get("source_structural_path"), item) or "",
                "source_code": _extract_one(mapping.get("source_code"), item),
                "article_no": _extract_one(mapping.get("source_article_no"), item),
            }
        )

    return {"answer": answer, "sources": sources, "generator_model": generator_model}
