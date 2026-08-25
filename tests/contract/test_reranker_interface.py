"""Contract test: Reranker Protocol compliance."""

from core.interfaces import Reranker
from core.models import Chunk, ScoredChunk


class _StubReranker:
    reranker_id = "stub"

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        return sorted(candidates, key=lambda s: s.score, reverse=True)


def _sc(text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(doc_id="d", text=text), score=score)


def test_reranker_satisfies_protocol():
    assert isinstance(_StubReranker(), Reranker)


def test_reranker_preserves_count():
    r = _StubReranker()
    cands = [_sc("a", 0.3), _sc("b", 0.9), _sc("c", 0.1)]
    result = r.rerank("q", cands)
    assert len(result) == 3


def test_reranker_returns_scored_chunks():
    r = _StubReranker()
    result = r.rerank("q", [_sc("x", 0.5)])
    assert all(isinstance(s, ScoredChunk) for s in result)
