"""POST /experiments/compare — per-question paired diff attached alongside
the existing aggregate metric_deltas. Mocks `_get_results`
directly (module-level function, same shape as production: dict[run_id ->
ExperimentResult]) rather than Mongo, mirroring test_feedback_router.py's
mocking style but one level higher since this endpoint's data source is an
already-parsed ExperimentResult, not a raw Mongo doc.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.experiment.config import ComponentRef, ExperimentConfig
from core.experiment.runner import ExperimentResult, QuestionResult
from services.api_gateway.routers.experiments import (
    _progress,
    router,
)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _cfg(name: str = "test") -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        chunking_strategy=ComponentRef(kind="chunker", component_id="fixed"),
        embedder=ComponentRef(kind="embedder", component_id="bge_m3"),
        generator=ComponentRef(kind="generator", component_id="stub"),
    )


def _qr(question_id: str, recall: float, answerability: str = "answerable") -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
        question=f"question {question_id}",
        reference_answer="ref",
        generated_answer="gen",
        metrics={"retrieval_recall_at_k": recall},
        answerability=answerability,
    )


def test_compare_response_includes_paired_diff(client: TestClient) -> None:
    before = ExperimentResult(
        config=_cfg("before"), run_id="run-a",
        question_results=[_qr("q1", 0.1), _qr("q2", 0.9)],
    )
    after = ExperimentResult(
        config=_cfg("after"), run_id="run-b",
        question_results=[_qr("q1", 0.9), _qr("q2", 0.1)],
    )
    with patch(
        "services.api_gateway.routers.experiments._get_results",
        AsyncMock(return_value={"run-a": before, "run-b": after}),
    ):
        resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

    assert resp.status_code == 200
    body = resp.json()
    assert "paired_diff" in body
    assert body["paired_diff"]["fixed"] == ["q1"]
    assert body["paired_diff"]["flips"] == ["q2"]


def test_paired_diff_questions_lookup_has_text_and_funnel_transition(client: TestClient) -> None:
    """A bare question_id (e.g. "19349990") says nothing on its own — the
    `questions` lookup is what lets the UI show the actual question text
    and which funnel layer it moved between, for every id in fixed/flips."""
    before = ExperimentResult(
        config=_cfg("before"), run_id="run-a",
        question_results=[_qr("q1", 0.1), _qr("q2", 0.9), _qr("q3", 0.9)],
    )
    after = ExperimentResult(
        config=_cfg("after"), run_id="run-b",
        question_results=[_qr("q1", 0.9), _qr("q2", 0.1), _qr("q3", 0.9)],
    )
    with patch(
        "services.api_gateway.routers.experiments._get_results",
        AsyncMock(return_value={"run-a": before, "run-b": after}),
    ):
        resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

    questions = resp.json()["paired_diff"]["questions"]
    assert questions["q1"] == {"question": "question q1", "funnel_before": "retrieval", "funnel_after": "ok"}
    assert questions["q2"] == {"question": "question q2", "funnel_before": "ok", "funnel_after": "retrieval"}
    # q3 stayed "ok" both times (unchanged) — no text lookup needed for it,
    # keeps the payload from growing with every unchanged question.
    assert "q3" not in questions


def test_compare_still_returns_aggregate_fields(client: TestClient) -> None:
    before = ExperimentResult(config=_cfg("before"), run_id="run-a", aggregate_metrics={"faithfulness": 0.8})
    after = ExperimentResult(config=_cfg("after"), run_id="run-b", aggregate_metrics={"faithfulness": 0.9})
    with patch(
        "services.api_gateway.routers.experiments._get_results",
        AsyncMock(return_value={"run-a": before, "run-b": after}),
    ):
        resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

    body = resp.json()
    assert "config_diff" in body
    assert "metric_deltas" in body
    assert "summary" in body


class TestResampleNoiseFiltering:
    """the own left-open acceptance criterion: a flip reported from a
    single before/after sample can be generation-metric noise; the endpoint
    resamples each flipped question before trusting it (see
    _resample_flip_verdicts / core.eval.regression.confirm_flips)."""

    def test_flip_confirmed_by_resample_stays_in_flips_and_is_marked_attempted(
        self, client: TestClient,
    ) -> None:
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("q1", 0.9)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("q1", 0.1)],
        )
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ), patch(
            "services.api_gateway.routers.experiments._resample_flip_verdicts",
            AsyncMock(return_value={"q1": [False, False]}),  # both resamples still not-ok
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        paired = resp.json()["paired_diff"]
        assert paired["flips"] == ["q1"]
        assert paired["unchanged"] == []
        assert paired["resample_attempted"] is True
        assert paired["noise_filtered"] == []

    def test_flip_demoted_to_noise_moves_to_unchanged(self, client: TestClient) -> None:
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("q1", 0.9)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("q1", 0.1)],
        )
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ), patch(
            "services.api_gateway.routers.experiments._resample_flip_verdicts",
            AsyncMock(return_value={"q1": [True, True]}),  # both resamples came back ok — noise
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        paired = resp.json()["paired_diff"]
        assert paired["flips"] == []
        assert paired["unchanged"] == ["q1"]
        assert paired["resample_attempted"] is True
        assert paired["noise_filtered"] == ["q1"]

    def test_resample_failure_degrades_to_original_single_sample_verdict(
        self, client: TestClient,
    ) -> None:
        """Pipeline/dataset can't be rebuilt (e.g. Realm resources missing) —
        _resample_flip_verdicts returns None; the comparison must still
        succeed with the original paired_diff verdict, not fail or hang."""
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("q1", 0.9)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("q1", 0.1)],
        )
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ), patch(
            "services.api_gateway.routers.experiments._resample_flip_verdicts",
            AsyncMock(return_value=None),
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        paired = resp.json()["paired_diff"]
        assert paired["flips"] == ["q1"]
        assert paired["resample_attempted"] is False
        assert paired["noise_filtered"] == []

    def test_no_flips_never_attempts_resampling(self, client: TestClient) -> None:
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("q1", 0.9)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("q1", 0.95)],
        )
        mock_resample = AsyncMock(return_value=None)
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ), patch(
            "services.api_gateway.routers.experiments._resample_flip_verdicts", mock_resample,
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        assert resp.json()["paired_diff"]["resample_attempted"] is False
        mock_resample.assert_not_called()

    def test_too_many_flips_skips_resampling_entirely(self, client: TestClient) -> None:
        # _MAX_RESAMPLE_FLIPS is 20 — a fix with more flips than that is
        # already unambiguously bad; save the compute rather than resample.
        many_before = [_qr(f"q{i}", 0.9) for i in range(25)]
        many_after = [_qr(f"q{i}", 0.1) for i in range(25)]
        before = ExperimentResult(config=_cfg("before"), run_id="run-a", question_results=many_before)
        after = ExperimentResult(config=_cfg("after"), run_id="run-b", question_results=many_after)
        mock_resample = AsyncMock(return_value=None)
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ), patch(
            "services.api_gateway.routers.experiments._resample_flip_verdicts", mock_resample,
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        assert len(resp.json()["paired_diff"]["flips"]) == 25
        mock_resample.assert_not_called()


class TestCompareProgress:
    """`progress_id` — the counter that makes a minutes-long comparison legible.

    Resampling re-answers up to `_MAX_RESAMPLE_FLIPS` questions twice each, and
    until a later version the screen showed one spinner for all of it. The client now
    names a channel, and the endpoint publishes into the same `_progress` dict
    the run's own WebSocket already serves.
    """

    def test_progress_events_are_published_under_the_client_s_id(
        self, client: TestClient,
    ) -> None:
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("q1", 0.9)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("q1", 0.1)],
        )

        async def _resample(_after, flip_ids, on_progress=None):
            assert on_progress is not None
            on_progress(0, len(flip_ids))
            on_progress(1, len(flip_ids))
            return {qid: [False, False] for qid in flip_ids}

        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ), patch(
            "services.api_gateway.routers.experiments._resample_flip_verdicts", _resample,
        ):
            resp = client.post(
                "/experiments/compare",
                json={"ids": ["run-a", "run-b"], "progress_id": "cmp-1"},
            )

        assert resp.status_code == 200
        events = _progress["cmp-1"]
        assert events[0] == {"type": "progress", "processed": 0, "total": 1}
        assert events[1] == {"type": "progress", "processed": 1, "total": 1}
        # Without a closing event the socket keeps a reader watching a
        # connection that has already said everything it will say.
        assert events[-1] == {"type": "done"}

    def test_a_comparison_with_nothing_to_resample_still_closes_the_channel(
        self, client: TestClient,
    ) -> None:
        # Identical runs — no flips, so resampling never starts. The reader is
        # still listening, and only `done` gets them off the socket.
        same = [_qr("q1", 0.9)]
        before = ExperimentResult(config=_cfg("before"), run_id="run-a", question_results=same)
        after = ExperimentResult(config=_cfg("after"), run_id="run-b", question_results=same)
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ):
            resp = client.post(
                "/experiments/compare",
                json={"ids": ["run-a", "run-b"], "progress_id": "cmp-2"},
            )

        assert resp.status_code == 200
        assert _progress["cmp-2"] == [{"type": "done"}]

    def test_without_a_progress_id_nothing_is_published_and_no_callback_passed(
        self, client: TestClient,
    ) -> None:
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("q1", 0.9)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("q1", 0.1)],
        )
        seen: list[object] = []

        async def _resample(_after, flip_ids, on_progress=None):
            seen.append(on_progress)
            return {qid: [False, False] for qid in flip_ids}

        before_keys = set(_progress)
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ), patch(
            "services.api_gateway.routers.experiments._resample_flip_verdicts", _resample,
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        assert resp.status_code == 200
        assert seen == [None]
        assert set(_progress) == before_keys
