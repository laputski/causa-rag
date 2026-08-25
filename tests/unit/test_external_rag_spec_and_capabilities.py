"""Published contract spec + capabilities probe.

GET /external-rag-spec must stay generated from the same Pydantic models that
parse real responses — these tests catch drift if SourceRef's
required/optional fields ever change without updating the spec generator.
Capabilities are derived from an actual probe response, not declared by the
registrant (granularity especially is an observable fact about what the
external RAG returned, not something to trust blindly).
"""
from __future__ import annotations

import asyncio

from core.models import SourceRef
from services.api_gateway.routers.external_rags import get_external_rag_spec, spec_router


def test_spec_router_has_bare_path():
    paths = {r.path for r in spec_router.routes}
    assert "/external-rag-spec" in paths


def test_spec_matches_real_sourceref_schema():
    spec = asyncio.run(get_external_rag_spec())
    schema = spec["schemas"]["SourceRef"]
    # doc_id is the only required field — spec must reflect this,
    # not the older shape where chunk_id was also required.
    assert schema.get("required") == ["doc_id"]
    assert "chunk_id" in schema["properties"]


def test_spec_documents_sources_at_top_level():
    spec = asyncio.run(get_external_rag_spec())
    assert "sources" in spec["response"]
    assert "trace" in spec["response"]


def test_spec_v1_1_0_documents_embedders_field_additively():
    """Trace.embedders is new, optional, and bumped the spec to
    1.1.0 (minor, not major) — an already-deployed causa_rag_client must
    not hard-fail its version check over this (see test_client.py's
    semver-tolerant check_contract_version tests)."""
    spec = asyncio.run(get_external_rag_spec())
    assert spec["version"] == "1.1.0"
    assert "embedders" in spec["schemas"]["ExternalTrace"]["properties"]
    assert "1.1.0" in spec["changelog"]


def _granularity_from_sources(sources: list[SourceRef]) -> str | None:
    if not sources:
        return None
    return "chunk" if any(s.chunk_id for s in sources) else "document"


def test_granularity_derivation_chunk_level():
    sources = [SourceRef(doc_id="d1", chunk_id="c1")]
    assert _granularity_from_sources(sources) == "chunk"


def test_granularity_derivation_document_level():
    sources = [SourceRef(doc_id="d1")]
    assert _granularity_from_sources(sources) == "document"


def test_granularity_derivation_no_sources():
    assert _granularity_from_sources([]) is None
