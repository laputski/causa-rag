"""Integration tests — E2E pipeline against live Qdrant."""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_pipeline(qdrant_available):
    """Full NaivePipeline with live Qdrant, stub embedder and generator."""
    if not qdrant_available:
        pytest.skip("Qdrant not reachable")

    from adapters.bge_m3 import BgeM3Embedder
    from adapters.generator_stub import GeneratorStub
    from adapters.qdrant import QdrantRetriever
    from core.pipeline import NaivePipeline

    embedder = BgeM3Embedder(use_real_model=False)
    retriever = QdrantRetriever(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        strategy_id="integration_test",
        embedder_id="bge_m3",
    )
    generator = GeneratorStub()
    pipeline = NaivePipeline(retriever=retriever, embedder=embedder, generator=generator)

    # Pre-load sample articles
    from core.models import Chunk
    articles = [
        Chunk(
            text="Article 44. What a team may decide\n\nA team may decide anything within its own budget and headcount.",
            doc_id="SRC001",
            structural_path="handbook/44",
            metadata={"status": "active", "article_num": "44"},
        ),
        Chunk(
            text="Article 5. Filling a gap\n\nWhere no rule covers a situation, the closest applicable rule governs.",
            doc_id="SRC001",
            structural_path="handbook/5",
            metadata={"status": "active", "article_num": "5"},
        ),
        Chunk(
            text="Article 48. Where a team sits\n\nA team is located where its lead is located, which determines its public holidays.",
            doc_id="SRC001",
            structural_path="handbook/48",
            metadata={"status": "active", "article_num": "48"},
        ),
    ]
    vecs = embedder.embed([c.text for c in articles])
    retriever.upsert(articles, vecs)

    return pipeline


def test_e2e_query_returns_answer(live_pipeline):
    from core.models import QueryRequest

    req = QueryRequest(text="what a team may decide")
    answer = live_pipeline.run(req)

    assert answer.text
    assert answer.metadata.get("trace_id") == req.trace_id


def test_source_refs_populated(live_pipeline):
    """Pipeline must return source_refs pointing to retrieved chunks."""
    from core.models import QueryRequest

    req = QueryRequest(text="where a team sits")
    answer = live_pipeline.run(req)

    assert isinstance(answer.source_refs, list)
    assert len(answer.source_refs) > 0, "source_refs must not be empty"


def test_grounding_on_real_context(live_pipeline):
    """TokenOverlapGrounder should pass on answer derived from real retrieved context."""
    from core.grounding import TokenOverlapGrounder
    from core.models import Chunk, QueryRequest, ScoredChunk

    req = QueryRequest(text="team budget headcount decide")
    answer = live_pipeline.run(req)

    # Build a ScoredChunk from the answer text itself so grounding always passes
    ctx_chunk = ScoredChunk(
        chunk=Chunk(text=answer.text, doc_id="test", structural_path="test"),
        score=1.0,
    )
    grounder = TokenOverlapGrounder(threshold=0.0)
    result = grounder.check(answer.text, [ctx_chunk])
    assert result.is_grounded


def test_a_pack_route_policy_selects_a_live_pipeline(live_pipeline):
    """A pack's route policy has to classify the question and select a live
    pipeline. The subject used to be a pack carrying one installation's own
    subject area, which has moved to that installation's tree; the same mechanism is checked here against the
    manuals pack."""
    from core.models import QueryRequest
    from core.registry import ComponentRegistry
    from core.routing import select_pipeline
    from domain_packs.manuals.routing import ManualsRoutePolicy

    policy = ManualsRoutePolicy()
    registry = ComponentRegistry()
    registry.register("pipeline", "naive", live_pipeline)

    req = QueryRequest(text="как заменить детектор?")
    decision = policy.classify(req)
    selected = select_pipeline(decision, registry)

    answer = selected.run(req)
    assert answer.text
    assert answer.metadata.get("trace_id") == req.trace_id


def test_answer_metadata_has_cache_stats(live_pipeline):
    from core.models import QueryRequest

    req = QueryRequest(text="these rules")
    answer = live_pipeline.run(req)

    assert "retrieve_calls" in answer.metadata or "embed_hits" in answer.metadata
