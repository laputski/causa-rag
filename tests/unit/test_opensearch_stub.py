from adapters.opensearch import OpenSearchRetrieverStub
from core.interfaces import Retriever
from core.models import Chunk


def _chunk(text: str, doc_id: str = "d1") -> Chunk:
    return Chunk(doc_id=doc_id, text=text, strategy_id="fixed")


def test_satisfies_protocol():
    assert isinstance(OpenSearchRetrieverStub(), Retriever)


def test_empty_index_returns_nothing():
    r = OpenSearchRetrieverStub()
    assert r.retrieve("query", k=5) == []


def test_exact_term_match():
    r = OpenSearchRetrieverStub()
    r.index_chunks([_chunk("a lease agreement for premises"), _chunk("some other document")])
    results = r.retrieve("lease", k=5)
    assert len(results) == 1
    assert "lease" in results[0].chunk.text


def test_top_k_respected():
    r = OpenSearchRetrieverStub()
    r.index_chunks([_chunk(f"word{i} shared") for i in range(10)])
    results = r.retrieve("shared", k=3)
    assert len(results) == 3


def test_ranking_by_overlap():
    r = OpenSearchRetrieverStub()
    r.index_chunks([
        _chunk("one term"),
        _chunk("one term another term"),
    ])
    results = r.retrieve("one term", k=2)
    # second chunk has more overlap
    assert results[0].chunk.text == "one term another term"


def test_no_match_returns_empty():
    r = OpenSearchRetrieverStub()
    r.index_chunks([_chunk("an entirely different text")])
    results = r.retrieve("irrelevant query xyz", k=5)
    assert results == []
