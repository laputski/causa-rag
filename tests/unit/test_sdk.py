"""core/sdk.py instrumentation SDK contract."""
from __future__ import annotations

from core.interfaces import Generator, Pipeline, Retriever
from core.models import QueryRequest
from core.sdk import build_pipeline, wrap_generator, wrap_retriever


def _naive_search(query: str, k: int, filters=None):
    corpus = ["the cat sits on the window", "the dog walks in the park", "the cat sleeps on the sofa"]
    scored = [(text, 1.0 / (i + 1)) for i, text in enumerate(corpus) if "cat" in text]
    return scored[:k]


def _naive_generate(prompt: str) -> str:
    return "an answer based on the context"


def test_wrap_retriever_satisfies_protocol():
    retriever = wrap_retriever(_naive_search)
    assert isinstance(retriever, Retriever)


def test_wrap_retriever_normalizes_tuples_to_scored_chunks():
    retriever = wrap_retriever(_naive_search)
    results = retriever.retrieve("cat", 5)
    assert len(results) == 2
    assert all(r.chunk.text for r in results)
    assert results[0].score >= results[1].score


def test_wrap_retriever_normalizes_dicts():
    def dict_search(query: str, k: int, filters=None):
        return [{"text": "the retrieved text", "score": 0.9, "doc_id": "doc-1"}]

    retriever = wrap_retriever(dict_search)
    results = retriever.retrieve("a question", 5)
    assert len(results) == 1
    assert results[0].chunk.text == "the retrieved text"
    assert results[0].chunk.doc_id == "doc-1"
    assert results[0].score == 0.9


def test_wrap_generator_satisfies_protocol():
    generator = wrap_generator(_naive_generate)
    assert isinstance(generator, Generator)
    assert generator.generate("any prompt") == "an answer based on the context"


def test_build_pipeline_satisfies_protocol_and_runs():
    retriever = wrap_retriever(_naive_search)
    generator = wrap_generator(_naive_generate)
    pipeline = build_pipeline(retriever, generator)
    assert isinstance(pipeline, Pipeline)

    answer = pipeline.run(QueryRequest(text="cat", top_k=5))

    assert answer.text == "an answer based on the context"
    assert len(answer.source_refs) == 2
    assert answer.stage_trace is not None
    assert answer.stage_trace.n_dense == 2
    assert answer.stage_trace.total_ms >= 0


def test_build_pipeline_applies_optional_reranker():
    class _ReverseReranker:
        reranker_id = "reverse"

        def rerank(self, query, candidates):
            return list(reversed(candidates))

    retriever = wrap_retriever(_naive_search)
    generator = wrap_generator(_naive_generate)
    pipeline = build_pipeline(retriever, generator, reranker=_ReverseReranker())

    answer = pipeline.run(QueryRequest(text="cat", top_k=5))
    assert answer.stage_trace.n_reranked == 2
    assert answer.stage_trace.rerank_ms is not None
