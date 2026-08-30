"""_CompositeEvaluator on a run that never generated anything.

A retrieval-only run returns an empty answer text by design. Every metric
computed from that text scores it anyway: correct_refusal reads emptiness
as a refusal, answer_relevance and context_support compare an empty string
against the question and the chunks. Averaged over a dataset that is three
columns of zeros describing a generator that was never asked to speak.

The Configuration Report is measured this way on every one of its rows, so
these hold the columns out rather than explaining them in a footnote.
"""
from __future__ import annotations

from core.models import Answer, SourceRef
from services.api_gateway.routers.experiments import _CompositeEvaluator


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


_QUESTION = {
    "id": "q1",
    "question": "Which article covers the deposit?",
    "article_refs": ["151236/1"],
}


def _answer(text: str, *, retrieval_only: bool) -> Answer:
    meta: dict = {"pipeline": "naive"}
    if retrieval_only:
        meta["retrieval_only"] = True
    return Answer(
        text=text,
        source_refs=[SourceRef(
            doc_id="d", chunk_id="c", chunk_text="the deposit is returned",
            source_code="151236", article_no="1",
        )],
        metadata=meta,
    )


def _ev() -> _CompositeEvaluator:
    return _CompositeEvaluator(embedder=_FakeEmbedder(), top_k=5)


def test_retrieval_metrics_are_still_reported():
    m = _ev().evaluate(_QUESTION, _answer("", retrieval_only=True))
    assert m["retrieval_recall_at_k"] == 1.0
    assert "retrieval_precision_at_k" in m
    assert "retrieval_average_precision" in m


def test_generation_metrics_are_absent_not_zero():
    m = _ev().evaluate(_QUESTION, _answer("", retrieval_only=True))
    for key in ("correct_refusal", "answer_relevance", "context_support",
                "answer_similarity", "grounded_in_correct_source"):
        assert key not in m, f"{key} was scored against an answer that was never generated"


def test_the_same_empty_answer_outside_a_retrieval_only_run_is_still_judged():
    """The bait. An empty text from a run that did call the generator is a
    real refusal and must keep being counted as one."""
    m = _ev().evaluate(_QUESTION, _answer("", retrieval_only=False))
    assert m["correct_refusal"] == 0.0
    assert m["answer_relevance"] == 0.0


def test_a_generated_answer_is_unaffected():
    m = _ev().evaluate(_QUESTION, _answer("The deposit is returned.", retrieval_only=False))
    assert m["correct_refusal"] == 1.0
    assert "answer_relevance" in m
    assert m["retrieval_recall_at_k"] == 1.0
