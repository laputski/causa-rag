"""Integration tests — Qdrant dense retrieval against a live instance."""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.integration


def _make_vector(seed_text: str) -> list[float]:
    """Deterministic 1024-dim vector for test seeding."""
    import hashlib
    seed = int(hashlib.md5(seed_text.encode()).hexdigest(), 16)  # noqa: S324
    rng = seed
    result: list[float] = []
    for _ in range(1024):
        rng = (rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        result.append((rng / 0xFFFFFFFFFFFFFFFF) * 2 - 1)
    return result


@pytest.fixture
def retriever(qdrant_collection, sample_article_chunks):
    """QdrantRetriever pointed at a temp collection, pre-loaded with those chunks."""
    from adapters.qdrant import QdrantRetriever

    collection_name, client = qdrant_collection
    # Create retriever pointing at the right host
    import os
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))

    retr = QdrantRetriever(
        host=host,
        port=port,
        strategy_id=collection_name,
        embedder_id="bge_m3",
    )

    # Upsert sample chunks with deterministic vectors
    vectors = [_make_vector(c.text) for c in sample_article_chunks]
    retr.upsert(sample_article_chunks, vectors)

    # Refresh (Qdrant is immediately consistent)
    return retr


def test_upsert_and_retrieve(retriever, sample_article_chunks):
    query_vec = _make_vector(sample_article_chunks[0].text)
    results = retriever.retrieve(query="", k=3, query_vector=query_vec)
    assert len(results) >= 1
    assert results[0].chunk.text == sample_article_chunks[0].text


def test_score_ordering(retriever, sample_article_chunks):
    query_vec = _make_vector(sample_article_chunks[4].text)
    results = retriever.retrieve(query="", k=5, query_vector=query_vec)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Scores must be descending"


def test_namespace_isolation(qdrant_client, qdrant_available):
    """Two collections with different names must not share results."""
    import uuid as _uuid

    from qdrant_client.models import Distance, PointStruct, VectorParams

    if not qdrant_available:
        pytest.skip("Qdrant not reachable")

    col_a = f"ns_a_{uuid.uuid4().hex[:6]}"
    col_b = f"ns_b_{uuid.uuid4().hex[:6]}"
    id_a = str(_uuid.uuid4())
    id_b = str(_uuid.uuid4())

    for col in [col_a, col_b]:
        qdrant_client.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    vec_a = _make_vector("alpha unique content")
    vec_b = _make_vector("beta unique content")

    qdrant_client.upsert(
        collection_name=col_a,
        points=[PointStruct(id=id_a, vector=vec_a, payload={"text": "alpha"})],
    )
    qdrant_client.upsert(
        collection_name=col_b,
        points=[PointStruct(id=id_b, vector=vec_b, payload={"text": "beta"})],
    )

    results_a = qdrant_client.query_points(collection_name=col_a, query=vec_a, limit=1).points
    results_b = qdrant_client.query_points(collection_name=col_b, query=vec_a, limit=1).points

    assert str(results_a[0].id) == id_a
    assert str(results_b[0].id) == id_b  # col_b has no alpha — returns beta

    for col in [col_a, col_b]:
        qdrant_client.delete_collection(col)


def test_dense_retrieval_on_a_numbered_unit_query(retriever, sample_article_chunks):
    """A query about what a team may decide should surface article 44."""
    art44_idx = next(i for i, c in enumerate(sample_article_chunks) if "44" in c.structural_path)
    query_vec = _make_vector(sample_article_chunks[art44_idx].text)
    results = retriever.retrieve(query="", k=3, query_vector=query_vec)
    top_paths = [r.chunk.structural_path for r in results]
    assert any("44" in p for p in top_paths), f"Expected article 44 in the top 3, got {top_paths}"
