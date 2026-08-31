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


class TestComparability:
    """The pair may not be measuring the same thing, and the report says so.

    Assertions are on the stable warning ids, never on the rendered sentence
    (CONTRIBUTING: "Assert on the stable identifier a diagnostic carries").
    """

    def _ids(self, body: dict) -> set[str]:
        return {w["id"] for w in body["compatibility"]["warnings"]}

    def test_compare_reports_a_comparable_pair_with_no_warnings(
        self, client: TestClient,
    ) -> None:
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("q1", 0.1)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("q1", 0.9)],
        )
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        body = resp.json()
        assert body["compatibility"]["comparable"] is True
        assert body["compatibility"]["matched"] == 1
        assert body["compatibility"]["warnings"] == []

    def test_runs_with_no_shared_question_are_reported_as_incomparable(
        self, client: TestClient,
    ) -> None:
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("a1", 0.1)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("b1", 0.9)],
        )
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        body = resp.json()
        assert body["compatibility"]["comparable"] is False
        assert "no_overlap" in self._ids(body)
        assert body["compatibility"]["matched"] == 0
        assert body["compatibility"]["only_in_before"] == 1
        assert body["compatibility"]["only_in_after"] == 1

    def test_the_exported_summary_carries_the_warnings_too(
        self, client: TestClient,
    ) -> None:
        # The .txt export is what gets pasted into tickets, so a caveat that
        # lives only on screen is a caveat the ticket never sees.
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("a1", 0.1)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("b1", 0.9)],
        )
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        assert "Comparability" in resp.json()["summary"]

    def test_run_facts_count_answered_questions_for_both_sides(
        self, client: TestClient,
    ) -> None:
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a",
            question_results=[_qr("q1", 0.1)], n_questions=50, stopped=True,
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b",
            question_results=[_qr("q1", 0.9), _qr("q2", 0.5)],
        )
        with patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(return_value={"run-a": before, "run-b": after}),
        ):
            resp = client.post("/experiments/compare", json={"ids": ["run-a", "run-b"]})

        compat = resp.json()["compatibility"]
        assert compat["before"]["n_questions"] == 1
        assert compat["after"]["n_questions"] == 2


class TestRequestValidation:
    """`ids` used to be an unbounded list, so one id raised IndexError and
    surfaced as a 500 while a third id was dropped without a word."""

    @pytest.mark.parametrize("ids", [[], ["only-one"], ["a", "b", "c"]])
    def test_anything_other_than_two_ids_is_rejected_by_validation(
        self, client: TestClient, ids: list[str],
    ) -> None:
        resp = client.post("/experiments/compare", json={"ids": ids})
        assert resp.status_code == 422


class TestPreflight:
    """The same verdict, delivered while the pair is still being chosen."""

    def test_preflight_names_what_is_incomparable_without_comparing(
        self, client: TestClient,
    ) -> None:
        before = ExperimentResult(
            config=_cfg("before"), run_id="run-a", question_results=[_qr("a1", 0.1)],
        )
        after = ExperimentResult(
            config=_cfg("after"), run_id="run-b", question_results=[_qr("b1", 0.9)],
        )
        after.config.dataset_name = "other.jsonl"

        async def _one(run_id: str):
            return {"run-a": before, "run-b": after}.get(run_id)

        with patch(
            "services.api_gateway.routers.experiments._get_one_result", _one,
        ), patch(
            "services.api_gateway.routers.experiments._resample_flip_verdicts",
            AsyncMock(side_effect=AssertionError("preflight must never resample")),
        ):
            resp = client.get("/experiments/compare/preflight?a=run-a&b=run-b")

        assert resp.status_code == 200
        body = resp.json()
        assert body["comparable"] is False
        assert {"different_dataset", "no_overlap"} <= {w["id"] for w in body["warnings"]}

    def test_preflight_reads_only_the_two_runs_it_was_asked_about(
        self, client: TestClient,
    ) -> None:
        # Reading the whole store on every change of a selector would make
        # choosing a pair cost more than comparing one.
        asked: list[str] = []
        run = ExperimentResult(
            config=_cfg("r"), run_id="run-a", question_results=[_qr("q1", 0.1)],
        )

        async def _one(run_id: str):
            asked.append(run_id)
            return run

        with patch(
            "services.api_gateway.routers.experiments._get_one_result", _one,
        ), patch(
            "services.api_gateway.routers.experiments._get_results",
            AsyncMock(side_effect=AssertionError("preflight must not read every run")),
        ):
            resp = client.get("/experiments/compare/preflight?a=run-a&b=run-b")

        assert resp.status_code == 200
        assert asked == ["run-a", "run-b"]

    def test_an_unknown_run_is_a_404_naming_it(self, client: TestClient) -> None:
        async def _one(run_id: str):
            return None

        with patch("services.api_gateway.routers.experiments._get_one_result", _one):
            resp = client.get("/experiments/compare/preflight?a=nope&b=also-nope")

        assert resp.status_code == 404
        assert "nope" in resp.json()["detail"]

    def test_the_preflight_route_is_not_swallowed_by_the_run_id_route(
        self, client: TestClient,
    ) -> None:
        # `GET /experiments/{run_id}` is declared earlier in the router, and a
        # two-segment path cannot match its single-segment pattern.
        async def _one(run_id: str):
            return None

        with patch("services.api_gateway.routers.experiments._get_one_result", _one):
            resp = client.get("/experiments/compare/preflight?a=x&b=y")

        assert resp.json()["detail"].startswith("Runs not found")


def test_the_compatibility_block_keeps_the_field_names_the_interface_reads(
    client: TestClient,
) -> None:
    """The TypeScript interface is written from these names by hand.

    Nothing else connects the two, so renaming a field here would leave the
    interface type-checking cleanly against a shape that no longer arrives.
    """
    before = ExperimentResult(
        config=_cfg("before"), run_id="run-a", question_results=[_qr("a1", 0.1)],
    )
    after = ExperimentResult(
        config=_cfg("after"), run_id="run-b", question_results=[_qr("b1", 0.9)],
    )
    with patch(
        "services.api_gateway.routers.experiments._get_results",
        AsyncMock(return_value={"run-a": before, "run-b": after}),
    ):
        body = client.post(
            "/experiments/compare", json={"ids": ["run-a", "run-b"]},
        ).json()

    compat = body["compatibility"]
    assert set(compat) == {
        "comparable", "matched", "only_in_before", "only_in_after",
        "before", "after", "warnings",
    }
    assert set(compat["before"]) == {
        "run_id", "dataset_name", "n_questions", "realm_id", "corpus_id", "stopped",
    }
    assert set(compat["warnings"][0]) == {
        "id", "severity", "title", "detail", "params", "action",
    }
    assert set(body["paired_diff"]) >= {"only_in_before", "only_in_after"}


def test_preflight_and_compare_agree_about_one_pair(client: TestClient) -> None:
    """One rule set, so the picker and the report cannot say different things."""
    before = ExperimentResult(
        config=_cfg("before"), run_id="run-a", question_results=[_qr("a1", 0.1)],
    )
    after = ExperimentResult(
        config=_cfg("after"), run_id="run-b", question_results=[_qr("b1", 0.9)],
    )

    async def _one(run_id: str):
        return {"run-a": before, "run-b": after}.get(run_id)

    with patch(
        "services.api_gateway.routers.experiments._get_results",
        AsyncMock(return_value={"run-a": before, "run-b": after}),
    ), patch("services.api_gateway.routers.experiments._get_one_result", _one):
        full = client.post(
            "/experiments/compare", json={"ids": ["run-a", "run-b"]},
        ).json()["compatibility"]
        pre = client.get("/experiments/compare/preflight?a=run-a&b=run-b").json()

    assert pre == full
