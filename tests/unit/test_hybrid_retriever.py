import pytest

from adapters.bge_m3 import BgeM3Embedder
from adapters.opensearch import OpenSearchRetrieverStub
from adapters.qdrant import QdrantRetrieverStub
from core.models import Chunk
from core.retrieval.hybrid import HybridRetriever, _rrf_score

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_hybrid(merge: str = "rrf") -> tuple[HybridRetriever, QdrantRetrieverStub, OpenSearchRetrieverStub]:
    emb = BgeM3Embedder()
    dense = QdrantRetrieverStub()
    sparse = OpenSearchRetrieverStub()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, embedder=emb, merge=merge)
    return hybrid, dense, sparse


def _chunk(text: str, doc_id: str = "d1") -> Chunk:
    return Chunk(doc_id=doc_id, text=text, strategy_id="fixed")


def _index_both(dense: QdrantRetrieverStub, sparse: OpenSearchRetrieverStub, chunks: list[Chunk], emb: BgeM3Embedder) -> None:
    vecs = emb.embed([c.text for c in chunks])
    dense.upsert(chunks, vecs)
    sparse.index_chunks(chunks)


# ── unit: RRF score formula ───────────────────────────────────────────────────

def test_rrf_score_decreases_with_rank():
    assert _rrf_score(1) > _rrf_score(2) > _rrf_score(10)


def test_rrf_score_always_positive():
    assert _rrf_score(100) > 0


# ── retriever tests ───────────────────────────────────────────────────────────

def test_hybrid_id():
    h, _, _ = _make_hybrid()
    assert h.retriever_id == "hybrid"


def test_invalid_merge():
    with pytest.raises(ValueError):
        HybridRetriever(None, None, None, merge="bad")  # type: ignore[arg-type]


def test_rrf_empty_returns_empty():
    h, _, _ = _make_hybrid("rrf")
    result = h.retrieve("a question", k=5)
    assert result == []


def test_rrf_deduplicates():
    emb = BgeM3Embedder()
    dense = QdrantRetrieverStub()
    sparse = OpenSearchRetrieverStub()
    h = HybridRetriever(dense, sparse, emb, merge="rrf")

    c = _chunk("a shared document agreement")
    vecs = emb.embed([c.text])
    dense.upsert([c], vecs)
    sparse.index_chunks([c])

    results = h.retrieve("agreement", k=10)
    ids = [r.chunk.chunk_id for r in results]
    assert len(ids) == len(set(ids)), "duplicates in result"


def test_rrf_boost_from_both_lists():
    emb = BgeM3Embedder()
    dense = QdrantRetrieverStub()
    sparse = OpenSearchRetrieverStub()
    h = HybridRetriever(dense, sparse, emb, merge="rrf")

    # chunk_a appears in both dense and sparse; chunk_b only in sparse
    chunk_a = _chunk("a shared lease agreement")
    chunk_b = _chunk("some other document text")

    vecs = emb.embed([chunk_a.text, chunk_b.text])
    dense.upsert([chunk_a], [vecs[0]])
    sparse.index_chunks([chunk_a, chunk_b])

    results = h.retrieve("agreement", k=5)
    top_id = results[0].chunk.chunk_id
    assert top_id == chunk_a.chunk_id, "chunk in both lists should rank higher"


def test_weighted_merge():
    emb = BgeM3Embedder()
    dense = QdrantRetrieverStub()
    sparse = OpenSearchRetrieverStub()
    h = HybridRetriever(dense, sparse, emb, merge="weighted", alpha=0.7)

    c = _chunk("a test document")
    dense.upsert([c], emb.embed([c.text]))
    sparse.index_chunks([c])

    results = h.retrieve("test", k=5)
    assert len(results) >= 1


def test_top_k_respected():
    emb = BgeM3Embedder()
    dense = QdrantRetrieverStub()
    sparse = OpenSearchRetrieverStub()
    h = HybridRetriever(dense, sparse, emb)

    chunks = [_chunk(f"document word{i}") for i in range(10)]
    _index_both(dense, sparse, chunks, emb)

    results = h.retrieve("document", k=3)
    assert len(results) <= 3
