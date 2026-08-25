"""eval/trulens_runner.py — unit-tested with a fake LiteLLM provider (no
real Ollama/network needed).

Regression: trulens's context_relevance/relevance/
groundedness_measure_with_cot_reasons all return (score, reasons_dict)
tuples in the installed version, not bare floats as their type hints
suggest — confirmed live against the real package. Forgetting to unpack
any one of them broke aggregation with "unsupported operand type(s) for +:
'int' and 'tuple'" inside sum().
"""
from __future__ import annotations

from unittest.mock import MagicMock

from eval.dataset import EvalDataset
from eval.trulens_runner import TruLensRunner


class _FakeAnswer:
    def __init__(self, text, source_refs=None):
        self.text = text
        self.source_refs = source_refs or []


class _FakePipeline:
    def run(self, request):
        return _FakeAnswer("an answer to " + request.text)


class _FakeProvider:
    """Mirrors the real trulens LiteLLM provider's tuple-returning shape."""

    def context_relevance(self, question, context):
        return (0.8, {"reason": {"score": 2}})

    def groundedness_measure_with_cot_reasons(self, source, statement):
        return (0.7, {"reason": {"score": 2}})

    def relevance(self, prompt, response):
        return (0.9, {"reason": {"score": 3}})


def _dataset() -> EvalDataset:
    return EvalDataset(
        name="test", version="v1", speed="fast",
        questions=[{"id": "q1", "question": "What is a rule?"}, {"id": "q2", "question": "What is a review?"}],
    )


def test_run_unpacks_tuple_returns_and_aggregates_means(tmp_path):
    runner = TruLensRunner(pipeline=_FakePipeline(), results_dir=tmp_path)
    runner._build_provider = MagicMock(return_value=_FakeProvider())

    result = runner.run(_dataset())

    assert result.metrics == {"context_relevance": 0.8, "groundedness": 0.7, "answer_relevance": 0.9}
    assert len(result.per_question) == 2


def test_run_respects_max_questions(tmp_path):
    runner = TruLensRunner(pipeline=_FakePipeline(), results_dir=tmp_path)
    runner._build_provider = MagicMock(return_value=_FakeProvider())

    result = runner.run(_dataset(), max_questions=1)

    assert len(result.per_question) == 1


def test_run_writes_results_to_disk(tmp_path):
    runner = TruLensRunner(pipeline=_FakePipeline(), results_dir=tmp_path)
    runner._build_provider = MagicMock(return_value=_FakeProvider())

    runner.run(_dataset())

    saved = list(tmp_path.glob("trulens_test_*.json"))
    assert saved
