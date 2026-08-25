from adapters.qdrant import QdrantRetrieverStub
from core.models import Chunk


def _vec(val: float, dim: int = 4) -> list[float]:
    return [val] * dim


def _chunk(doc_id: str = "d1", text: str = "hello") -> Chunk:
    return Chunk(doc_id=doc_id, text=text)


def test_retrieve_empty():
    r = QdrantRetrieverStub()
    assert r.retrieve("q", k=5, query_vector=_vec(1.0)) == []


def test_upsert_and_retrieve():
    r = QdrantRetrieverStub()
    c = _chunk()
    r.upsert([c], [_vec(1.0)])
    results = r.retrieve("q", k=1, query_vector=_vec(1.0))
    assert len(results) == 1
    assert results[0].chunk.doc_id == "d1"


def test_top_k_respected():
    r = QdrantRetrieverStub()
    chunks = [_chunk(text=f"t{i}") for i in range(5)]
    vecs = [_vec(float(i) / 10) for i in range(5)]
    r.upsert(chunks, vecs)
    results = r.retrieve("q", k=3, query_vector=_vec(1.0))
    assert len(results) == 3


def test_ranking_order():
    r = QdrantRetrieverStub()
    low = _chunk(text="low")
    high = _chunk(text="high")
    r.upsert([low, high], [_vec(0.1), _vec(0.9)])
    results = r.retrieve("q", k=2, query_vector=_vec(1.0))
    assert results[0].chunk.text == "high"
