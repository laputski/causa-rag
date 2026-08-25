"""services/api_gateway/routers/experiments.py:_resample_flip_verdicts —
the helper that re-runs each flipped question a couple more times
against a rebuilt pipeline, so `confirm_flips` can tell a genuine
per-question regression apart from generation-metric noise. Mocks the
pipeline rebuild / dataset load / evaluator boundary (all real I/O or
disk-dependent), leaving the actual sample-aggregation logic exercised for
real — the same "mock the boundary, keep the logic real" split
test_paired_diff.py and test_compare_paired_diff.py already use.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult
from core.models import Answer
from services.api_gateway.routers.experiments import _resample_flip_verdicts


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="after",
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="stub"),
    )


def _after(dataset_name: str = "handbook.v1.fast.jsonl") -> ExperimentResult:
    return ExperimentResult(config=_cfg(), run_id="run-b", dataset_name=dataset_name)


class _FakeDataset:
    def __init__(self, questions: list[dict]) -> None:
        self.questions = questions


@pytest.fixture
def mocked_boundary():
    """Patches everything `_resample_flip_verdicts` treats as an I/O
    boundary: pipeline rebuild, dataset load, embedder resolution, and the
    evaluator's own metric computation. Yields the fake pipeline's `.run`
    mock so a test can control what answer.text/metrics come back per call."""
    fake_pipeline = MagicMock()
    with patch(
        "core.experiment.runner.ExperimentRunner._build_pipeline", return_value=fake_pipeline,
    ), patch(
        "services.api_gateway.routers.experiments._load_dataset",
        AsyncMock(return_value=_FakeDataset([
            {"id": "q1", "question": "q1 text?", "article_refs": ["art-1"], "reference_answer": "ref"},
        ])),
    ), patch(
        "services.api_gateway.routers.experiments.registry.resolve", return_value=MagicMock(),
    ):
        yield fake_pipeline


class TestResampleFlipVerdicts:
    async def test_returns_none_when_pipeline_cannot_be_rebuilt(self) -> None:
        with patch(
            "core.experiment.runner.ExperimentRunner._build_pipeline",
            side_effect=RuntimeError("no realm resources"),
        ):
            result = await _resample_flip_verdicts(_after(), ["q1"])
        assert result is None

    async def test_returns_none_when_dataset_cannot_load(self) -> None:
        with patch(
            "core.experiment.runner.ExperimentRunner._build_pipeline", return_value=MagicMock(),
        ), patch(
            "services.api_gateway.routers.experiments._load_dataset",
            AsyncMock(side_effect=RuntimeError("dataset gone")),
        ):
            result = await _resample_flip_verdicts(_after(), ["q1"])
        assert result is None

    async def test_question_missing_from_dataset_is_skipped_not_erroring(self, mocked_boundary) -> None:
        result = await _resample_flip_verdicts(_after(), ["q-does-not-exist"])
        assert result == {}

    async def test_collects_two_resamples_by_default(self, mocked_boundary) -> None:
        with patch(
            "services.api_gateway.routers.experiments._CompositeEvaluator.evaluate",
            return_value={"retrieval_recall_at_k": 0.9, "answer_similarity": 0.9, "grounded_in_correct_source": 0.9},
        ), patch(
            "services.api_gateway.routers.experiments._CompositeEvaluator.resolve_answerability",
            return_value="answerable",
        ):
            mocked_boundary.run.return_value = Answer(text="a good answer")
            result = await _resample_flip_verdicts(_after(), ["q1"])
        assert result is not None
        assert len(result["q1"]) == 2
        assert mocked_boundary.run.call_count == 2

    async def test_ok_metrics_resample_as_ok_true(self, mocked_boundary) -> None:
        with patch(
            "services.api_gateway.routers.experiments._CompositeEvaluator.evaluate",
            return_value={"retrieval_recall_at_k": 0.9, "answer_similarity": 0.9, "grounded_in_correct_source": 0.9},
        ), patch(
            "services.api_gateway.routers.experiments._CompositeEvaluator.resolve_answerability",
            return_value="answerable",
        ):
            mocked_boundary.run.return_value = Answer(text="a good answer")
            result = await _resample_flip_verdicts(_after(), ["q1"])
        assert result == {"q1": [True, True]}

    async def test_bad_metrics_resample_as_ok_false(self, mocked_boundary) -> None:
        with patch(
            "services.api_gateway.routers.experiments._CompositeEvaluator.evaluate",
            return_value={"retrieval_recall_at_k": 0.0, "answer_similarity": 0.1},
        ), patch(
            "services.api_gateway.routers.experiments._CompositeEvaluator.resolve_answerability",
            return_value="answerable",
        ):
            mocked_boundary.run.return_value = Answer(text="")
            result = await _resample_flip_verdicts(_after(), ["q1"])
        assert result == {"q1": [False, False]}

    async def test_a_pipeline_exception_on_one_sample_counts_as_not_ok(self, mocked_boundary) -> None:
        # Conservative: a hard failure to even answer isn't silently
        # dropped from the sample set — it counts against the flip being
        # confirmed as noise, same as a real run records QuestionResult.error
        # rather than pretending the question never ran.
        mocked_boundary.run.side_effect = RuntimeError("generator timed out")
        with patch(
            "services.api_gateway.routers.experiments._CompositeEvaluator.evaluate",
            return_value={},
        ), patch(
            "services.api_gateway.routers.experiments._CompositeEvaluator.resolve_answerability",
            return_value="answerable",
        ):
            result = await _resample_flip_verdicts(_after(), ["q1"])
        assert result == {"q1": [False, False]}
