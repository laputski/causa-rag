"""Real integration test — Neo4jGraphRetriever.link_semantic against a live
Neo4j instance and a live Qdrant collection (no mocks). Seeds three chunks:
two near-duplicate vectors (should link) and one unrelated vector (should
not), then verifies the RELATED_SEMANTIC edges Neo4j actually wrote.

Always passes doc_ids=[doc_id] to link_semantic — this Neo4j instance is
shared with real corpus ingestion (no per-test isolation), so an unscoped
call scans every :Chunk node ever ingested, not just this test's three.
Found live: that turned a 3-chunk test into a 30+s hang against a
34k-chunk real corpus already sitting in the same graph.
"""
from __future__ import annotations

import uuid

import pytest
from qdrant_client.models import PointStruct

from core.models import Chunk

pytestmark = pytest.mark.integration


def _vector(base: float, noise: float = 0.0) -> list[float]:
    return [base + noise * (i % 7) for i in range(1024)]


class _QdrantStandIn:
    """Minimal shim exposing the ._client/._collection attributes
    link_semantic reads — the real QdrantRetriever wraps the same
    qdrant_client.QdrantClient, so this matches its actual shape."""

    def __init__(self, client: object, collection: str) -> None:
        self._client = client
        self._collection = collection


def test_link_semantic_connects_near_duplicates_not_unrelated_chunks(
    qdrant_collection, neo4j_graph,
) -> None:
    collection_name, qdrant_client = qdrant_collection
    retriever, prefix = neo4j_graph

    # Qdrant point IDs must be a UUID or unsigned int — chunk_id elsewhere in
    # the platform is already a uuid4 string (core/models.py:_chunk_id), so
    # real chunk_ids satisfy this. The test-only `prefix` is carried
    # separately (in Chunk.doc_id) purely so the Neo4j fixture can find and
    # clean up these nodes afterward without touching a real corpus.
    cid_a, cid_b, cid_c = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())

    # a and b are near-duplicate vectors (should land above any reasonable
    # cosine threshold); c is a deliberately distant vector.
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=cid_a, vector=_vector(0.5, noise=0.001)),
            PointStruct(id=cid_b, vector=_vector(0.5, noise=0.0011)),
            PointStruct(id=cid_c, vector=_vector(-0.5, noise=0.07)),
        ],
    )

    qdrant = _QdrantStandIn(qdrant_client, collection_name)
    doc_id = f"{prefix}doc"
    chunks = [
        Chunk(chunk_id=cid_a, doc_id=doc_id, text="a", structural_path="x"),
        Chunk(chunk_id=cid_b, doc_id=doc_id, text="b", structural_path="x"),
        Chunk(chunk_id=cid_c, doc_id=doc_id, text="q", structural_path="x"),
    ]
    retriever.add_chunks(chunks)

    written = retriever.link_semantic(qdrant, threshold=0.9, top_k=5, doc_ids=[doc_id])
    assert written >= 1

    driver = retriever._connect()
    with driver.session() as session:
        pairs = list(session.run(
            "MATCH (x:Chunk)-[r:RELATED_SEMANTIC]-(y:Chunk) "
            "WHERE x.doc_id = $doc_id "
            "RETURN x.chunk_id AS a, y.chunk_id AS b, r.score AS score",
            doc_id=doc_id,
        ))

    linked_pairs = {frozenset((p["a"], p["b"])) for p in pairs}
    assert frozenset((cid_a, cid_b)) in linked_pairs
    assert frozenset((cid_a, cid_c)) not in linked_pairs
    assert frozenset((cid_b, cid_c)) not in linked_pairs
    for p in pairs:
        assert p["score"] >= 0.9


def test_link_semantic_is_idempotent(qdrant_collection, neo4j_graph) -> None:
    """Re-running after edges already exist must MERGE, not duplicate."""
    collection_name, qdrant_client = qdrant_collection
    retriever, prefix = neo4j_graph
    cid_a, cid_b = str(uuid.uuid4()), str(uuid.uuid4())

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=cid_a, vector=_vector(0.5, noise=0.001)),
            PointStruct(id=cid_b, vector=_vector(0.5, noise=0.0011)),
        ],
    )

    qdrant = _QdrantStandIn(qdrant_client, collection_name)
    doc_id = f"{prefix}doc"
    chunks = [
        Chunk(chunk_id=cid_a, doc_id=doc_id, text="a", structural_path="x"),
        Chunk(chunk_id=cid_b, doc_id=doc_id, text="b", structural_path="x"),
    ]
    retriever.add_chunks(chunks)

    retriever.link_semantic(qdrant, threshold=0.9, top_k=5, doc_ids=[doc_id])
    retriever.link_semantic(qdrant, threshold=0.9, top_k=5, doc_ids=[doc_id])

    driver = retriever._connect()
    with driver.session() as session:
        count = session.run(
            "MATCH (x:Chunk)-[r:RELATED_SEMANTIC]-(y:Chunk) "
            "WHERE x.doc_id = $doc_id "
            "RETURN count(r) AS c",
            doc_id=doc_id,
        ).single()["c"]
    assert count == 2  # one undirected edge, counted from both endpoints
