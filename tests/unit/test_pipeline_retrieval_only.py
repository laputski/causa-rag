"""core/pipeline.py, retrieve(): the search half, stopped before the model.

core/experiment/runner.py switches a run to retrieval-only by asking whether
its pipeline has a `retrieve` method. The in-process pipeline had none, so
`retrieval_only: true` was accepted and quietly ignored, so every run still
generated. These tests hold that shut with a generator that raises when
called, which is the only way the difference is visible from the outside:
a run that generates and one that does not return the same shape of Answer.

Fakes follow the convention in test_funnel_pipeline_instrumentation.py.
"""
from __future__ import annotations

import contextlib

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentRunner
from core.models import Chunk, QueryRequest, ScoredChunk
from core.pipeline import NaivePipeline
from core.registry import ComponentRegistry
from eval.dataset import EvalDataset


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
        return "an answer citing Fragment 1"


class _ExplodingGenerator:
    """Reddens the moment anything asks it for text.

    The bait for every claim below. Without it a passing test would only
    prove that retrieve() returns an empty string, which a generator
    returning nothing also does.
    """

    generator_id = "exploding"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        raise AssertionError("the generator was called on a retrieval-only run")


class _ReversingReranker:
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


def _pipeline(generator=None, **kw) -> NaivePipeline:
    return NaivePipeline(
        retriever=_FakeRetriever([_chunk("5"), _chunk("44"), _chunk("7")]),
        embedder=_FakeEmbedder(),
        generator=generator or _FakeGenerator(),
        **kw,
    )


def test_retrieve_never_reaches_the_generator():
    gen = _ExplodingGenerator()
    ans = _pipeline(gen).retrieve(QueryRequest(text="a question", top_k=2))
    assert gen.calls == 0
    assert ans.text == ""


def test_run_still_reaches_the_generator():
    """The bait, proving the test above measures the flag and not the fake:
    the same pipeline through run() does call the generator. Counted rather
    than caught, because the Langfuse span wrapper re-raises as a
    RuntimeError and an exception-type assertion would measure the wrapper.
    """
    gen = _ExplodingGenerator()
    with contextlib.suppress(Exception):
        _pipeline(gen).run(QueryRequest(text="a question", top_k=2))
    assert gen.calls == 1


def test_retrieve_returns_the_same_sources_as_run():
    """The whole worth of the measurement. If the two paths could return
    different sources, a retrieval-only report would describe a search the
    platform never performs for a real question."""
    req = lambda: QueryRequest(text="a question", top_k=2)  # noqa: E731
    from_run = _pipeline().run(req())
    from_retrieve = _pipeline().retrieve(req())

    assert ([s.article_no for s in from_retrieve.source_refs]
            == [s.article_no for s in from_run.source_refs])
    assert from_retrieve.stage_trace.context_chars == from_run.stage_trace.context_chars


def test_retrieve_applies_the_reranker_and_keeps_the_pre_rerank_snapshot():
    """Funnel diagnosis has to survive the short path, or the report can
    say a configuration lost a document but not where it lost it."""
    from core.pipeline import ConfigurablePipeline

    p = ConfigurablePipeline(
        retriever=_FakeRetriever([_chunk("5"), _chunk("44"), _chunk("7")]),
        embedder=_FakeEmbedder(),
        generator=_ExplodingGenerator(),
        reranker=_ReversingReranker(),
        top_k=3,
    )
    ans = p.retrieve(QueryRequest(text="a question"))

    assert [s.article_no for s in ans.source_refs] == ["7", "44", "5"]
    assert [s.article_no for s in ans.pre_rerank_source_refs] == ["5", "44", "7"]


def test_retrieve_keeps_the_candidate_window_when_fetch_k_widens_it():
    p = _pipeline(_ExplodingGenerator(), fetch_k=3)
    ans = p.retrieve(QueryRequest(text="a question", top_k=1))

    assert len(ans.source_refs) == 1
    assert len(ans.candidate_source_refs) == 3


def test_retrieve_honours_top_k():
    ans = _pipeline(_ExplodingGenerator()).retrieve(QueryRequest(text="a question", top_k=2))
    assert len(ans.source_refs) == 2


def test_answer_says_it_was_retrieval_only():
    """A stored run with an empty text must not be mistaken later for a
    generation run whose model returned nothing."""
    ans = _pipeline(_ExplodingGenerator()).retrieve(QueryRequest(text="a question"))
    assert ans.metadata["retrieval_only"] is True
    assert ans.refused is False
    assert ans.computed_citations == []
    assert ans.rendered_prompt_preview == ""


def test_retrieve_records_timing():
    """total_ms is set inside the run; skipping generation must not leave
    the trace looking like a run that never happened."""
    ans = _pipeline(_ExplodingGenerator()).retrieve(QueryRequest(text="a question"))
    assert ans.stage_trace.total_ms >= 0.0
    assert ans.stage_trace.context_chars > 0


def test_runner_switch_sees_the_method():
    """The exact expression at core/experiment/runner.py, where the switch is a
    hasattr, so an in-process pipeline without this method would take the
    generating path with no error and no log line."""
    assert hasattr(_pipeline(), "retrieve")


def test_retrieval_only_run_through_the_runner_never_generates():
    """The end-to-end gate: a config with retrieval_only=true, driven the
    way the gateway drives it, must not reach the generator once."""
    gen = _ExplodingGenerator()
    registry = ComponentRegistry()
    registry.register("pipeline", "naive", _pipeline(gen))
    runner = ExperimentRunner(registry=registry)

    config = ExperimentConfig(
        name="retrieval-only",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="fake"),
        generator=ComponentRef(kind="generator", component_id="exploding"),
        pipeline_id="naive",
        retrieval_only=True,
        top_k=2,
    )
    dataset = EvalDataset(
        name="t", version="v1", speed="fast",
        questions=[{"id": "q1", "question": "a question"}, {"id": "q2", "question": "another"}],
    )

    result = runner.run(config, dataset)

    assert gen.calls == 0
    assert [r.error for r in result.question_results] == [None, None]
    assert all(r.generated_answer == "" for r in result.question_results)
    assert all(len(r.source_refs) == 2 for r in result.question_results)


def test_runner_without_the_flag_does_generate():
    """The bait for the gate above: the identical setup with
    retrieval_only left off reaches the exploding generator."""
    gen = _ExplodingGenerator()
    registry = ComponentRegistry()
    registry.register("pipeline", "naive", _pipeline(gen))
    runner = ExperimentRunner(registry=registry)

    config = ExperimentConfig(
        name="generating",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="fake"),
        generator=ComponentRef(kind="generator", component_id="exploding"),
        pipeline_id="naive",
        top_k=2,
    )
    dataset = EvalDataset(
        name="t", version="v1", speed="fast",
        questions=[{"id": "q1", "question": "a question"}],
    )

    result = runner.run(config, dataset)

    assert gen.calls == 1
    assert result.question_results[0].error is not None
