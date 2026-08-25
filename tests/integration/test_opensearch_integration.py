"""Integration tests — OpenSearch BM25 retrieval against a live instance."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _index_chunks(client, index: str, chunks) -> None:
    from opensearchpy.helpers import bulk

    actions = [
        {
            "_index": index,
            "_id": str(chunk.chunk_id),
            "_source": {
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_id": str(chunk.chunk_id),
                "structural_path": chunk.structural_path,
            },
        }
        for chunk in chunks
    ]
    bulk(client, actions)
    client.indices.refresh(index=index)


def test_index_and_bm25_retrieve(opensearch_index, sample_article_chunks):
    index, client = opensearch_index
    _index_chunks(client, index, sample_article_chunks)

    resp = client.search(
        index=index,
        body={
            "query": {"match": {"text": "team decide budget headcount"}},
            "size": 5,
        },
    )
    hits = resp["hits"]["hits"]
    assert len(hits) > 0, "Expected BM25 results"
    texts = [h["_source"]["text"] for h in hits]
    assert any("a team may decide" in t.lower() for t in texts)


def test_bm25_score_ordering(opensearch_index, sample_article_chunks):
    index, client = opensearch_index
    _index_chunks(client, index, sample_article_chunks)

    resp = client.search(
        index=index,
        body={
            "query": {"match": {"text": "how a team is named"}},
            "size": 10,
        },
    )
    hits = resp["hits"]["hits"]
    scores = [h["_score"] for h in hits]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by score desc"


def test_exact_article_reference(opensearch_index, sample_article_chunks):
    """Navigational query with article number should find the specific article."""
    index, client = opensearch_index
    _index_chunks(client, index, sample_article_chunks)

    resp = client.search(
        index=index,
        body={
            "query": {"match": {"text": "Article 48"}},
            "size": 3,
        },
    )
    hits = resp["hits"]["hits"]
    assert len(hits) > 0
    # Top result should be article 48
    top_path = hits[0]["_source"].get("structural_path", "")
    assert "48" in top_path, f"Expected article 48, got {top_path}"


def test_ru_morphology_finds_variants(opensearch_index, sample_article_chunks):
    """BM25 should find a document containing "Agreement" when searching "agreement"."""
    index, client = opensearch_index

    from core.models import Chunk
    extra = Chunk(
        text="An Agreement is what creates an obligation between two teams.",
        doc_id="SRC001",
        structural_path="handbook/390",
    )
    _index_chunks(client, index, [extra])

    # Case-insensitive match works with the lowercase filter
    resp = client.search(
        index=index,
        body={"query": {"match": {"text": "agreement"}}, "size": 5},
    )
    texts = [h["_source"]["text"] for h in resp["hits"]["hits"]]
    assert any("agreement" in t.lower() for t in texts), 'Should find the document containing "Agreement"' 


def test_hybrid_retriever_integration(qdrant_collection, opensearch_index, sample_article_chunks):
    """HybridRetriever (RRF) should combine dense+sparse results."""
    import os

    from adapters.bge_m3 import BgeM3Embedder
    from adapters.opensearch import OpenSearchRetriever
    from adapters.qdrant import QdrantRetriever
    from core.retrieval.hybrid import HybridRetriever

    collection_name, qdrant_client = qdrant_collection
    os_index, os_client = opensearch_index

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))

    embedder = BgeM3Embedder(use_real_model=False)
    dense = QdrantRetriever(host=host, port=port, strategy_id=collection_name, embedder_id="bge_m3")
    sparse = OpenSearchRetriever(
        host=os.getenv("OPENSEARCH_HOST", "localhost"),
        port=int(os.getenv("OPENSEARCH_PORT", "9200")),
        strategy_id=os_index,
    )

    # Populate both
    vecs = embedder.embed([c.text for c in sample_article_chunks])
    dense.upsert(sample_article_chunks, vecs)
    _index_chunks(os_client, os_index, sample_article_chunks)

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, embedder=embedder)
    results = hybrid.retrieve("what a team may decide", k=5)

    assert len(results) >= 1
    # Hybrid returns results — with stub embedder the ranking is deterministic but not semantic.
    # Verify structural integrity: each result has a path and score.
    for r in results:
        assert r.chunk.structural_path, "structural_path must be non-empty"
        assert r.score >= 0.0, "score must be non-negative"


def test_opensearch_retriever_roundtrips_metadata(opensearch_index):
    """Eval Measurement Trustworthiness, Phase 0 — source_code/article_no
    must survive index_chunks() -> retrieve() through the REAL
    OpenSearchRetriever, not just OpenSearchRetrieverStub (which trivially
    preserves it by holding the Chunk object in memory). Without this,
    any chunk surfaced only via the sparse side of a hybrid merge
    silently loses the fields retrieval_recall_at_k/retrieval_precision_at_k
    need, undercounting retrieval quality for reasons unrelated to the
    pipeline being evaluated.
    """
    from adapters.opensearch import OpenSearchRetriever
    from core.models import Chunk

    index, _client = opensearch_index
    retriever = OpenSearchRetriever(strategy_id=index, corpus_id="default")

    chunk = Chunk(
        doc_id="SRC001", text="A team may decide anything within its own budget.",
        structural_path="handbook/44",
        metadata={"source_code": "SRC001", "article_no": "44"},
    )
    retriever.index_chunks([chunk])

    results = retriever.retrieve("what a team may decide", k=5)
    assert len(results) == 1
    assert results[0].chunk.metadata.get("source_code") == "SRC001"
    assert results[0].chunk.metadata.get("article_no") == "44"
