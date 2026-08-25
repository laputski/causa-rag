"""core/pipeline.py — pre-rerank snapshot (Funnel diagnosis, Phase 1).

Without capturing the ranked list before the reranker reorders/cuts it,
"the reranker dropped the right chunk" and "retrieval never found it" are
indistinguishable from stored data alone. Uses domain-neutral fakes,
matching the convention in test_pipeline_domain_hooks.py.
"""
from __future__ import annotations

from core.models import Chunk, QueryRequest, ScoredChunk
from core.pipeline import NaivePipeline


class _FakeRetriever:
    retriever_id = "fake"

    def __init__(self, chunks: list[ScoredChunk]):
        self._chunks = chunks

    def retrieve(self, query: str, k: int = 5, filters=None, **kwargs) -> list[ScoredChunk]:
        return list(self._chunks)


class _FakeEmbedder:
    embedder_id = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _FakeGenerator:
    generator_id = "fake"

    def generate(self, prompt: str) -> str:
        return "an answer"


class _ReversingReranker:
    """Deterministically reverses candidate order — simulates a reranker
    that demotes whatever retrieval ranked first."""

    reranker_id = "reversing"

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        return list(reversed(candidates))


def _chunk(article_no: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            doc_id="d1", chunk_id=f"c-{article_no}", text=f"text of article {article_no}",
            structural_path=f"handbook/{article_no}",
            metadata={"source_code": "SRC001", "article_no": article_no},
        ),
        score=1.0,
        retriever_id="fake",
    )


def test_no_reranker_leaves_pre_rerank_source_refs_empty():
    """source_refs IS the pre-rerank list when there's no reranker — no
    separate snapshot needed, nothing to distinguish."""
    p = NaivePipeline(
        retriever=_FakeRetriever([_chunk("5"), _chunk("44")]),
        embedder=_FakeEmbedder(),
        generator=_FakeGenerator(),
    )
    ans = p.run(QueryRequest(text="a question", top_k=5))
    assert ans.pre_rerank_source_refs == []
    assert len(ans.source_refs) == 2


def test_reranker_demotes_the_correct_chunk_out_of_top_k():
    """The exact scenario the instrumentation exists for: retrieval found
    the target article at rank 1, but the reranker demoted it to rank 2,
    and top_k=1 cuts it from the final context. Pre-rerank recall would
    show success; post-rerank recall must show failure — and only the
    pre_rerank_source_refs snapshot lets you tell the two apart.
    """
    p = NaivePipeline(
        retriever=_FakeRetriever([_chunk("5"), _chunk("44")]),  # "5" is the target, ranked first
        embedder=_FakeEmbedder(),
        generator=_FakeGenerator(),
        reranker=_ReversingReranker(),
    )
    ans = p.run(QueryRequest(text="a question", top_k=1))

    pre_rank_article_nos = [sr.article_no for sr in ans.pre_rerank_source_refs]
    post_rank_article_nos = [sr.article_no for sr in ans.source_refs]

    assert pre_rank_article_nos == ["5", "44"]  # full pre-rerank order preserved
    assert post_rank_article_nos == ["44"]  # target "5" was demoted and cut by top_k=1
    assert "5" in pre_rank_article_nos
    assert "5" not in post_rank_article_nos


def test_pre_rerank_source_refs_carries_same_fields_as_source_refs():
    p = NaivePipeline(
        retriever=_FakeRetriever([_chunk("5")]),
        embedder=_FakeEmbedder(),
        generator=_FakeGenerator(),
        reranker=_ReversingReranker(),
    )
    ans = p.run(QueryRequest(text="a question", top_k=5))
    sr = ans.pre_rerank_source_refs[0]
    assert sr.source_code == "SRC001"
    assert sr.article_no == "5"
    assert sr.chunk_text == "text of article 5"
