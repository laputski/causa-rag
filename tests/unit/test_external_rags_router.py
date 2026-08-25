"""External RAGs registry router — contract tests (no running Mongo needed)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from services.api_gateway.routers import external_rags as ext_rags
from services.api_gateway.routers.external_rags import (
    ExternalRagCreateRequest,
    ExternalRagUpdateRequest,
    delete_external_rag,
    router,
    update_external_rag,
)


def test_router_prefix():
    assert router.prefix == "/external-rags"


def test_routes_exist():
    paths = {r.path for r in router.routes}
    assert any(p.endswith("/external-rags") for p in paths)
    assert any("{rag_id}" in p for p in paths)
    assert any("{rag_id}/test" in p for p in paths)


def test_per_rag_dataset_crud_routes_no_longer_here():
    """per-rag dataset CRUD moved to the
    unified services/api_gateway/routers/datasets.py (POST/GET /datasets,
    DELETE /datasets/by-id/{id}, source_rag_id-scoped) — this router no
    longer owns a dedicated dataset sub-resource."""
    paths = {r.path for r in router.routes}
    assert not any("datasets" in p for p in paths)


@pytest.mark.asyncio
async def test_delete_external_rag_cascades_to_its_datasets():
    """Replaces the old dedicated-collection 'one query' cascade
    with an equivalent delete_many filtered by source_rag_id on the unified
    `datasets` collection."""
    with patch.object(ext_rags.mdb, "delete_one", AsyncMock(return_value=1)), \
         patch.object(ext_rags.mdb, "delete_many", AsyncMock(return_value=3)) as delete_many:
        await delete_external_rag("bdc63487")

    delete_many.assert_called_once_with("datasets", {"source_rag_id": "bdc63487"})


@pytest.mark.asyncio
async def test_delete_external_rag_404_skips_cascade_when_rag_not_found():
    with patch.object(ext_rags.mdb, "delete_one", AsyncMock(return_value=0)), \
         patch.object(ext_rags.mdb, "delete_many", AsyncMock()) as delete_many, \
         pytest.raises(HTTPException):
        await delete_external_rag("missing")

    delete_many.assert_not_called()


def test_create_request_defaults_uses_realm_resources_to_false():
    """Absence must mean 'no opinion, don't offer the corpus
    inspection shortcut', not silently opt every existing registration into
    it (which would be a claim about infra sharing nobody actually made)."""
    body = ExternalRagCreateRequest(name="x", url="https://example.com")
    assert body.uses_realm_resources is False


@pytest.mark.asyncio
async def test_create_external_rag_persists_uses_realm_resources():
    with patch.object(ext_rags.mdb, "insert_one", AsyncMock(return_value=None)) as insert_one:
        from services.api_gateway.routers.external_rags import create_external_rag
        await create_external_rag(ExternalRagCreateRequest(
            name="x", url="https://example.com", uses_realm_resources=True,
        ))
    assert insert_one.call_args[0][1]["uses_realm_resources"] is True


def test_create_request_defaults_supported_params_to_empty_list():
    """Declared at registration, defaults to "this RAG
    declares no extensible params", not an error, for back-compat with
    every registration request made before this field existed."""
    body = ExternalRagCreateRequest(name="x", url="https://example.com")
    assert body.supported_params == []


def test_create_request_accepts_supported_params():
    body = ExternalRagCreateRequest(
        name="x", url="https://example.com", supported_params=["fetch_k", "temperature"],
    )
    assert body.supported_params == ["fetch_k", "temperature"]


# ── per-(Realm, ExternalRag) default-config ────────────────────────

def test_create_request_defaults_have_no_config_out_of_the_box():
    """Backward compat: a registration made before default_* existed (or one
    that doesn't set them) must not accidentally imply a corpus_id/pipeline_id
    — absence means 'no opinion', not 'default corpus'."""
    body = ExternalRagCreateRequest(name="x", url="https://example.com")
    assert body.default_corpus_id is None
    assert body.default_pipeline_id is None
    assert body.default_reranker_id is None
    assert body.default_params == {}


def test_create_request_accepts_default_config():
    body = ExternalRagCreateRequest(
        name="x", url="https://example.com",
        default_corpus_id="handbook", default_pipeline_id="hybrid_rrf",
        default_reranker_id="cross_encoder", default_params={"alpha": 0.7},
    )
    assert body.default_corpus_id == "handbook"
    assert body.default_params == {"alpha": 0.7}


def test_patch_route_exists():
    paths_and_methods = {(r.path, m) for r in router.routes for m in r.methods}
    assert any(p.endswith("{rag_id}") and "PATCH" in m for p, m in paths_and_methods)


@pytest.mark.asyncio
async def test_update_external_rag_merges_only_provided_fields():
    existing = {"id": "abc123", "name": "old", "default_corpus_id": None, "_id": "mongoid"}
    body = ExternalRagUpdateRequest(default_corpus_id="handbook")
    with patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=existing)), \
         patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)) as update_one:
        result = await update_external_rag("abc123", body)

    update_one.assert_called_once_with(
        "external_rags", {"id": "abc123"}, {"$set": {"default_corpus_id": "handbook"}},
    )
    assert result["default_corpus_id"] == "handbook"
    assert result["name"] == "old"  # untouched field preserved
    assert "_id" not in result  # never leaks the raw Mongo id


@pytest.mark.asyncio
async def test_update_external_rag_404_when_not_found():
    with patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=None)), \
         pytest.raises(HTTPException) as exc_info:
        await update_external_rag("missing", ExternalRagUpdateRequest(default_corpus_id="x"))
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_external_rag_400_when_no_fields_given():
    with pytest.raises(HTTPException) as exc_info:
        await update_external_rag("abc123", ExternalRagUpdateRequest())
    assert exc_info.value.status_code == 400


# ── timeout_s — per-RAG override of HttpPipeline's 30s default ────────────────

def test_create_request_defaults_timeout_s_to_none():
    """None ⇒ every call site falls back to HttpPipeline's 30s default,
    unchanged for every RAG registered before this field existed."""
    body = ExternalRagCreateRequest(name="x", url="https://example.com")
    assert body.timeout_s is None


def test_create_request_accepts_timeout_s():
    body = ExternalRagCreateRequest(name="x", url="https://example.com", timeout_s=180.0)
    assert body.timeout_s == 180.0


@pytest.mark.asyncio
async def test_update_external_rag_sets_timeout_s():
    existing = {"id": "abc123", "name": "old", "timeout_s": None, "_id": "mongoid"}
    body = ExternalRagUpdateRequest(timeout_s=180.0)
    with patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=existing)), \
         patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)):
        result = await update_external_rag("abc123", body)
    assert result["timeout_s"] == 180.0


@pytest.mark.asyncio
async def test_create_external_rag_persists_timeout_s():
    body = ExternalRagCreateRequest(name="slow-agent", url="https://example.com", timeout_s=180.0)
    with patch.object(ext_rags.mdb, "insert_one", AsyncMock(return_value=None)):
        doc = await ext_rags.create_external_rag(body)
    assert doc["timeout_s"] == 180.0


@pytest.mark.asyncio
async def test_test_external_rag_passes_registered_timeout_to_http_pipeline():
    """The registration-time probe must honor timeout_s too — otherwise a
    slow-but-working RAG (found live: a multi-step agentic one) always fails
    test() even after the field is set, since HttpPipeline still defaults to
    30s."""
    from core.models import Answer

    doc = {"id": "abc123", "url": "http://slow-rag/platform/query", "timeout_s": 180.0}
    fake_answer = Answer(text="ok", source_refs=[], stage_trace=None)
    with (
        patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=doc)),
        patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)),
        patch("adapters.http_pipeline.HttpPipeline") as mock_pipeline_cls,
    ):
        mock_pipeline_cls.return_value.run.return_value = fake_answer
        await ext_rags.test_external_rag("abc123")

    assert mock_pipeline_cls.call_args.kwargs["timeout"] == 180.0


@pytest.mark.asyncio
async def test_test_external_rag_falls_back_to_30s_when_timeout_s_unset():
    from core.models import Answer

    doc = {"id": "abc123", "url": "http://fast-rag/platform/query"}
    fake_answer = Answer(text="ok", source_refs=[], stage_trace=None)
    with (
        patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=doc)),
        patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)),
        patch("adapters.http_pipeline.HttpPipeline") as mock_pipeline_cls,
    ):
        mock_pipeline_cls.return_value.run.return_value = fake_answer
        await ext_rags.test_external_rag("abc123")

    assert mock_pipeline_cls.call_args.kwargs["timeout"] == 30.0


# ── Embedder mismatch/hint detection ──────────────────────────────

def _rag_doc(**overrides):
    doc = {
        "id": "abc123", "url": "http://rag/platform/query",
        "realm_id": "demo", "default_corpus_id": "handbook",
    }
    doc.update(overrides)
    return doc


def _fake_answer(reported_embedders=None):
    from core.models import Answer
    return Answer(
        text="ok", source_refs=[], stage_trace=None,
        metadata={"reported_embedders": reported_embedders},
    )


@pytest.mark.asyncio
async def test_test_passes_corpus_and_realm_id_to_http_pipeline():
    """The prerequisite fix this phase depends on — without corpus_id/realm_id
    on the probe, there's no meaningful corpus to compare a reported embedder
    against at all."""
    with (
        patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=_rag_doc())),
        patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)),
        patch("adapters.http_pipeline.HttpPipeline") as mock_pipeline_cls,
        patch.object(ext_rags, "_known_embedders_for_corpus", AsyncMock(return_value=set())),
    ):
        mock_pipeline_cls.return_value.run.return_value = _fake_answer()
        await ext_rags.test_external_rag("abc123")

    assert mock_pipeline_cls.call_args.kwargs["corpus_id"] == "handbook"
    assert mock_pipeline_cls.call_args.kwargs["realm_id"] == "demo"


@pytest.mark.asyncio
async def test_no_warning_when_reported_embedder_matches_known():
    with (
        patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=_rag_doc())),
        patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)),
        patch("adapters.http_pipeline.HttpPipeline") as mock_pipeline_cls,
        patch.object(ext_rags, "_known_embedders_for_corpus", AsyncMock(return_value={"bge_m3"})),
    ):
        mock_pipeline_cls.return_value.run.return_value = _fake_answer(["bge_m3"])
        result = await ext_rags.test_external_rag("abc123")

    assert result["capabilities"]["embedder_mismatch_warning"] is None
    assert result["capabilities"]["embedder_hint"] is None
    assert result["capabilities"]["reported_embedders"] == ["bge_m3"]


@pytest.mark.asyncio
async def test_mismatch_warning_when_reported_and_known_disjoint():
    with (
        patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=_rag_doc())),
        patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)),
        patch("adapters.http_pipeline.HttpPipeline") as mock_pipeline_cls,
        patch.object(ext_rags, "_known_embedders_for_corpus", AsyncMock(return_value={"bge_m3"})),
    ):
        mock_pipeline_cls.return_value.run.return_value = _fake_answer(["paraphrase-multilingual-mpnet-base-v2"])
        result = await ext_rags.test_external_rag("abc123")

    assert result["capabilities"]["embedder_mismatch_warning"] is not None
    assert "handbook" in result["capabilities"]["embedder_mismatch_warning"]
    assert result["capabilities"]["embedder_hint"] is None


@pytest.mark.asyncio
async def test_no_mismatch_when_reported_partially_overlaps_known():
    """A RAG using only ONE of a multi-embedder corpus's embedders (e.g. only
    the Qdrant/BGE-M3 side of a GraphRAG-type corpus) is legitimate, not a
    mismatch — any intersection is enough."""
    with (
        patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=_rag_doc())),
        patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)),
        patch("adapters.http_pipeline.HttpPipeline") as mock_pipeline_cls,
        patch.object(ext_rags, "_known_embedders_for_corpus", AsyncMock(return_value={"bge_m3", "nomic-embed-text"})),
    ):
        mock_pipeline_cls.return_value.run.return_value = _fake_answer(["bge_m3"])
        result = await ext_rags.test_external_rag("abc123")

    assert result["capabilities"]["embedder_mismatch_warning"] is None


@pytest.mark.asyncio
async def test_hint_when_rag_reports_no_embedder_but_corpus_has_known_ones():
    with (
        patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=_rag_doc())),
        patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)),
        patch("adapters.http_pipeline.HttpPipeline") as mock_pipeline_cls,
        patch.object(ext_rags, "_known_embedders_for_corpus", AsyncMock(return_value={"bge_m3"})),
    ):
        mock_pipeline_cls.return_value.run.return_value = _fake_answer(None)
        result = await ext_rags.test_external_rag("abc123")

    assert result["capabilities"]["embedder_hint"] is not None
    assert result["capabilities"]["embedder_mismatch_warning"] is None


@pytest.mark.asyncio
async def test_no_hint_or_warning_when_corpus_not_registered():
    """Platform never guesses at an unregistered corpus's embedder — nothing
    to compare against means no warning AND no hint, not a default assumption."""
    with (
        patch.object(ext_rags.mdb, "find_one", AsyncMock(return_value=_rag_doc())),
        patch.object(ext_rags.mdb, "update_one", AsyncMock(return_value=1)),
        patch("adapters.http_pipeline.HttpPipeline") as mock_pipeline_cls,
        patch.object(ext_rags, "_known_embedders_for_corpus", AsyncMock(return_value=set())),
    ):
        mock_pipeline_cls.return_value.run.return_value = _fake_answer(None)
        result = await ext_rags.test_external_rag("abc123")

    assert result["capabilities"]["embedder_hint"] is None
    assert result["capabilities"]["embedder_mismatch_warning"] is None


@pytest.mark.asyncio
async def test_known_embedders_for_corpus_collects_across_backends():
    corpora = [{
        "realm_id": "demo", "corpus_id": "handbook",
        "backends": {
            "qdrant": {"embedder_id": "bge_m3"},
            "neo4j": {"embedder_id": "nomic-embed-text"},
            "opensearch": {"index": "no-embedder-here"},
        },
    }]
    with patch("services.api_gateway.routers.corpus._list_corpora", AsyncMock(return_value=corpora)):
        result = await ext_rags._known_embedders_for_corpus("demo", "handbook")

    assert result == {"bge_m3", "nomic-embed-text"}


@pytest.mark.asyncio
async def test_known_embedders_for_corpus_empty_without_realm_or_corpus_id():
    result = await ext_rags._known_embedders_for_corpus(None, "handbook")
    assert result == set()
    result = await ext_rags._known_embedders_for_corpus("demo", None)
    assert result == set()
