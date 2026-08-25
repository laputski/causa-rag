"""eval/deepeval_runner.py — result/report metadata.

Found live: GET /report/deepeval showed a raw epoch float next to the title
and no way to tell which judge/generator model, or how many questions, a
given saved run actually used — with multiple historical runs sitting side
by side and the judge model now swappable (eval/judge_model.py), "which run
is which" became a real question. Pins that DeepEvalRunner.run() actually
persists judge_model/generator_model/n_questions, not just computes them.
"""
from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import patch

# The runner imports deepeval's metrics at call time, so without the `judges`
# extra this raises instead of exercising anything. A fresh clone carries the
# `dev` extra alone and `pytest` there must be green.
import pytest

from eval.dataset import EvalDataset

_needs_judges = pytest.mark.skipif(
    importlib.util.find_spec("deepeval") is None,
    reason="needs the judges: pip install -e '.[judges]'",
)


def _fake_metric_data(name: str, score: float):
    return SimpleNamespace(name=name, score=score)


def _fake_test_result():
    return SimpleNamespace(metrics_data=[
        _fake_metric_data("Answer Relevancy", 0.8),
        _fake_metric_data("Faithfulness", 0.9),
        _fake_metric_data("Contextual Precision", 0.7),
        _fake_metric_data("Contextual Recall", 0.6),
    ])


@_needs_judges
def test_run_persists_judge_generator_model_and_question_count(tmp_path):
    from eval.deepeval_runner import DeepEvalRunner

    dataset = EvalDataset(name="handbook", version="v0", speed="fast", questions=[
        {"id": "q1", "question": "Question 1?", "ground_truth": "Answer 1"},
        {"id": "q2", "question": "Question 2?", "ground_truth": "Answer 2"},
    ])

    pipeline = SimpleNamespace(
        _generator=SimpleNamespace(_model="qwen3:8b"),  # deliberately != judge model below
        run=lambda req: SimpleNamespace(text="The model's answer", source_refs=[]),
    )

    runner = DeepEvalRunner(pipeline=pipeline, ollama_model="qwen2.5:7b-instruct-q4_K_M", results_dir=tmp_path)

    fake_deepeval = SimpleNamespace(
        evaluate=lambda test_cases, metrics: SimpleNamespace(
            test_results=[_fake_test_result() for _ in test_cases],
        ),
    )
    with patch.dict("sys.modules", {
        "deepeval": SimpleNamespace(
            evaluate=fake_deepeval.evaluate,
            models=SimpleNamespace(OllamaModel=lambda **kw: SimpleNamespace()),
            metrics=SimpleNamespace(
                AnswerRelevancyMetric=lambda **kw: SimpleNamespace(),
                FaithfulnessMetric=lambda **kw: SimpleNamespace(),
                ContextualPrecisionMetric=lambda **kw: SimpleNamespace(),
                ContextualRecallMetric=lambda **kw: SimpleNamespace(),
            ),
            test_case=SimpleNamespace(LLMTestCase=lambda **kw: SimpleNamespace(**kw)),
        ),
    }):
        result = runner.run(dataset)

    assert result.judge_model == "qwen2.5:7b-instruct-q4_K_M"
    assert result.generator_model == "qwen3:8b"
    assert result.n_questions == 2

    saved = sorted(tmp_path.glob("deepeval_handbook_*.json"))
    assert saved, "expected a results file to be written"
    content = json.loads(saved[-1].read_text(encoding="utf-8"))
    assert content["judge_model"] == "qwen2.5:7b-instruct-q4_K_M"
    assert content["n_questions"] == 2
    assert "generator_model" in content
