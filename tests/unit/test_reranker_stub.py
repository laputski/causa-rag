from adapters.reranker import CrossEncoderRerankerStub
from core.interfaces import Reranker
from core.models import Chunk, ScoredChunk


def _sc(text: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(doc_id="d", text=text), score=score)


def test_satisfies_protocol():
    assert isinstance(CrossEncoderRerankerStub(), Reranker)


def test_empty_input():
    r = CrossEncoderRerankerStub()
    assert r.rerank("q", []) == []


def test_reranks_by_relevance():
    r = CrossEncoderRerankerStub()
    cands = [
        _sc("irrelevant document text"),
        _sc("lease agreement agreement"),
    ]
    result = r.rerank("lease agreement", cands)
    assert result[0].chunk.text == "lease agreement agreement"


def test_preserves_all_candidates():
    r = CrossEncoderRerankerStub()
    cands = [_sc(f"text {i}") for i in range(5)]
    result = r.rerank("some text", cands)
    assert len(result) == 5


def test_sorted_descending():
    r = CrossEncoderRerankerStub()
    cands = [_sc("a"), _sc("a b"), _sc("a b c")]
    result = r.rerank("a b c", cands)
    scores = [s.score for s in result]
    assert scores == sorted(scores, reverse=True)
