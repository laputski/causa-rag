"""core/pipeline.py optional scorer/mask_engine/refusal_policy
hooks. Uses domain-neutral fakes (not legal-specific), matching the existing
convention for the reranker/grounder/route_policy hook tests.
"""
from __future__ import annotations

from core.models import Answer, Chunk, GroundingResult, QueryRequest, ScoredChunk
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
        return "hello world. another sentence."


def _chunk(text: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(doc_id="d1", chunk_id=f"c-{text[:4]}", text=text, structural_path="root"),
        score=score,
        retriever_id="fake",
    )


def _make_pipeline(**kwargs) -> NaivePipeline:
    return NaivePipeline(
        retriever=kwargs.pop("retriever", _FakeRetriever([_chunk("alpha", 1.0), _chunk("beta", 2.0)])),
        embedder=_FakeEmbedder(),
        generator=_FakeGenerator(),
        **kwargs,
    )


def test_all_three_hooks_are_noop_when_none() -> None:
    pipeline = _make_pipeline()
    answer = pipeline.run(QueryRequest(text="hi", top_k=5))
    assert not answer.refused
    assert answer.text == "hello world. another sentence."


def test_scorer_reorders_results() -> None:
    class _ReverseScorer:
        def score(self, query, candidates):
            return list(reversed(candidates))

    captured: list[list[ScoredChunk]] = []

    class _CapturingRetriever(_FakeRetriever):
        pass

    class _Spy:
        def score(self, query, candidates):
            captured.append(list(candidates))
            return list(reversed(candidates))

    pipeline = _make_pipeline(scorer=_Spy())
    pipeline.run(QueryRequest(text="hi", top_k=5))
    assert [sc.chunk.text for sc in captured[0]] == ["alpha", "beta"]


def test_mask_engine_wraps_answer_text() -> None:
    class _FakeMaskEngine:
        def render(self, question_type, answer_text, source_paths=None, extra=None) -> str:
            return f"[{question_type}] {answer_text}"

    pipeline = _make_pipeline(mask_engine=_FakeMaskEngine())
    answer = pipeline.run(QueryRequest(text="hi", top_k=5))
    assert answer.text.startswith("[open] ")


def test_refusal_policy_fires_when_grounder_flags_low_confidence() -> None:
    class _AlwaysUngroundedGrounder:
        grounder_id = "fake"

        def check(self, answer_text, context_chunks) -> GroundingResult:
            return GroundingResult(is_grounded=False, unsupported_claims=["x"], confidence=0.0)

    class _FakeRefusalPolicy:
        policy_id = "fake_refusal"

        def should_refuse(self, grounding: GroundingResult) -> bool:
            return not grounding.is_grounded

        def build_refusal(self, reason: str, original_query: str) -> Answer:
            return Answer(text="I refuse.", refused=True, refusal_reason=reason)

    pipeline = _make_pipeline(grounder=_AlwaysUngroundedGrounder(), refusal_policy=_FakeRefusalPolicy())
    answer = pipeline.run(QueryRequest(text="hi", top_k=5))
    assert answer.refused
    assert answer.text == "I refuse."
    assert answer.refusal_reason == "low_confidence"


def test_refusal_policy_does_not_fire_without_a_grounder() -> None:
    class _FakeRefusalPolicy:
        policy_id = "fake_refusal"

        def should_refuse(self, grounding: GroundingResult) -> bool:
            return True  # would always refuse, but there's no grounding to check

        def build_refusal(self, reason: str, original_query: str) -> Answer:
            return Answer(text="I refuse.", refused=True, refusal_reason=reason)

    pipeline = _make_pipeline(refusal_policy=_FakeRefusalPolicy())  # no grounder
    answer = pipeline.run(QueryRequest(text="hi", top_k=5))
    assert not answer.refused


def test_mask_engine_skipped_when_refusal_fires() -> None:
    class _AlwaysUngroundedGrounder:
        grounder_id = "fake"

        def check(self, answer_text, context_chunks) -> GroundingResult:
            return GroundingResult(is_grounded=False, confidence=0.0)

    class _FakeRefusalPolicy:
        def should_refuse(self, grounding: GroundingResult) -> bool:
            return True

        def build_refusal(self, reason: str, original_query: str) -> Answer:
            return Answer(text="I refuse.", refused=True, refusal_reason=reason)

    class _FakeMaskEngine:
        def render(self, question_type, answer_text, source_paths=None, extra=None) -> str:
            return f"MASKED({answer_text})"

    pipeline = _make_pipeline(
        grounder=_AlwaysUngroundedGrounder(),
        refusal_policy=_FakeRefusalPolicy(),
        mask_engine=_FakeMaskEngine(),
    )
    answer = pipeline.run(QueryRequest(text="hi", top_k=5))
    assert answer.text == "I refuse."  # not "MASKED(I refuse.)"
